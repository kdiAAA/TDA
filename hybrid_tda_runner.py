import argparse
import copy
import math
import random
import time
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import wandb
import yaml
from tqdm import tqdm

import clip
from utils import build_test_data_loader, cls_acc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', required=True, help='Path to a YAML config file.')
    return parser.parse_args()


def load_yaml(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def deep_update(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


@torch.no_grad()
def build_prompt_embeddings(classnames: List[str], templates: List[str], clip_model) -> torch.Tensor:
    class_prompt_embeddings = []
    for classname in classnames:
        classname = classname.replace('_', ' ')
        texts = [t.format(classname) for t in templates]
        tokens = clip.tokenize(texts).cuda()
        text_embeddings = clip_model.encode_text(tokens)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        class_prompt_embeddings.append(text_embeddings)
    return torch.stack(class_prompt_embeddings, dim=0).cuda()


@torch.no_grad()
def encode_image(images, clip_model) -> torch.Tensor:
    if isinstance(images, list):
        images = torch.cat(images, dim=0).cuda()
    else:
        images = images.cuda()
    features = clip_model.encode_image(images)
    features = features / features.norm(dim=-1, keepdim=True)
    return features


@torch.no_grad()
def adaptive_prompt_logits(
    image_features: torch.Tensor,
    prompt_embeddings: torch.Tensor,
    prompt_cfg: dict,
    logit_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    c, t, d = prompt_embeddings.shape
    sims = torch.einsum('bd,ctd->bct', image_features, prompt_embeddings)

    if not prompt_cfg.get('enabled', True) or t == 1:
        class_embeddings = prompt_embeddings.mean(dim=1)
        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ class_embeddings.t()
        return logits, class_embeddings

    selection = prompt_cfg.get('selection', 'topk')
    if selection == 'all':
        keep_mask = torch.ones_like(sims, dtype=torch.bool)
    elif selection == 'percentile':
        percentile = float(prompt_cfg.get('percentile', 0.5))
        k = max(1, int(math.ceil(t * percentile)))
        top_idx = sims.topk(k=k, dim=-1).indices
        keep_mask = torch.zeros_like(sims, dtype=torch.bool)
        keep_mask.scatter_(-1, top_idx, True)
    else:
        k = max(1, min(int(prompt_cfg.get('topk', 1)), t))
        top_idx = sims.topk(k=k, dim=-1).indices
        keep_mask = torch.zeros_like(sims, dtype=torch.bool)
        keep_mask.scatter_(-1, top_idx, True)

    masked = prompt_embeddings.unsqueeze(0) * keep_mask.unsqueeze(-1)
    class_embeddings = masked.sum(dim=2)
    norm = class_embeddings.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    class_embeddings = class_embeddings / norm
    logits = logit_scale * torch.einsum('bd,bcd->bc', image_features, class_embeddings)
    return logits, class_embeddings.squeeze(0)


class ClasswiseCache:
    def __init__(self, enabled: bool, shot_capacity: int, replacement_policy: str):
        self.enabled = enabled
        self.shot_capacity = shot_capacity
        self.replacement_policy = replacement_policy
        self.store: Dict[int, List[dict]] = {}

    def __len__(self):
        return sum(len(v) for v in self.store.values())

    def add(self, pred: int, feature: torch.Tensor, score: float, label_vector: torch.Tensor):
        if not self.enabled:
            return
        entry = {
            'feature': feature.detach(),
            'score': float(score),
            'label': label_vector.detach(),
        }
        bucket = self.store.setdefault(pred, [])
        if len(bucket) < self.shot_capacity:
            bucket.append(entry)
            if self.replacement_policy == 'loss_sorted':
                bucket.sort(key=lambda x: x['score'])
            return

        if self.replacement_policy == 'fifo':
            bucket.pop(0)
            bucket.append(entry)
            return

        bucket.sort(key=lambda x: x['score'])
        if entry['score'] < bucket[-1]['score']:
            bucket[-1] = entry
            bucket.sort(key=lambda x: x['score'])

    def logits(self, image_features: torch.Tensor, alpha: float, beta: float, num_classes: int) -> torch.Tensor:
        if len(self) == 0:
            return torch.zeros((image_features.size(0), num_classes), device=image_features.device, dtype=image_features.dtype)
        keys = []
        values = []
        for class_index in sorted(self.store.keys()):
            for entry in self.store[class_index]:
                keys.append(entry['feature'])
                values.append(entry['label'])
        cache_keys = torch.cat(keys, dim=0).t()
        cache_values = torch.cat(values, dim=0)
        affinity = image_features @ cache_keys
        weights = (-1.0 * (beta - beta * affinity)).exp()
        return alpha * (weights @ cache_values)


class PrototypeMemory:
    def __init__(self, enabled: bool, init_embeddings: torch.Tensor):
        self.enabled = enabled
        self.embeddings = init_embeddings.detach().clone()
        self.embeddings = self.embeddings / self.embeddings.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        self.mass = torch.full((init_embeddings.size(0),), 1e-6, device=init_embeddings.device, dtype=init_embeddings.dtype)

    def logits(self, image_features: torch.Tensor, scale: float) -> torch.Tensor:
        if not self.enabled:
            return torch.zeros((image_features.size(0), self.embeddings.size(0)), device=image_features.device, dtype=image_features.dtype)
        return scale * (image_features @ self.embeddings.t())

    def update(self, image_feature: torch.Tensor, target_dist: torch.Tensor, cfg: dict):
        if not self.enabled:
            return
        rule = cfg.get('update_rule', 'topk_soft')
        topk = int(cfg.get('topk', 2))
        forgetting = float(cfg.get('forgetting_factor', 1.0))
        weights = target_dist.squeeze(0).detach().clone()

        if rule == 'top1_hard':
            idx = int(weights.argmax().item())
            weights.zero_()
            weights[idx] = 1.0
        elif rule == 'topk_soft':
            k = max(1, min(topk, weights.numel()))
            idx = weights.topk(k=k).indices
            mask = torch.zeros_like(weights)
            mask[idx] = 1.0
            weights = weights * mask
            weights = weights / weights.sum().clamp(min=1e-12)
        elif rule == 'all_soft':
            weights = weights / weights.sum().clamp(min=1e-12)
        else:
            raise ValueError(f'Unsupported prototype update rule: {rule}')

        feat = image_feature.squeeze(0)
        for c in range(weights.numel()):
            w = float(weights[c].item())
            old_mass = float(self.mass[c].item())
            new_mass = forgetting * old_mass + w
            if new_mass <= 0:
                continue
            if w > 0:
                updated = (forgetting * old_mass * self.embeddings[c] + w * feat) / new_mass
            else:
                updated = self.embeddings[c]
            updated = updated / updated.norm().clamp(min=1e-12)
            self.embeddings[c] = updated
            self.mass[c] = new_mass


def normalized_entropy(prob: torch.Tensor) -> torch.Tensor:
    entropy = -(prob * prob.clamp(min=1e-12).log()).sum(dim=1)
    return entropy / math.log(prob.size(1))



def confidence_from_logits(logits: torch.Tensor) -> float:
    probs = logits.softmax(dim=1)
    return float((1.0 - normalized_entropy(probs)).mean().item())



def maturity_weight(step_idx: int, cfg: dict) -> float:
    rho = float(cfg.get('maturity_rho', 0.01))
    cap = float(cfg.get('maturity_cap', 1.0))
    return float(min(rho * step_idx, cap))



def make_positive_label(prob: torch.Tensor, cfg: dict) -> torch.Tensor:
    label_type = cfg.get('label_type', 'topk_soft')
    p = prob.squeeze(0).detach().clone()
    if label_type == 'one_hot':
        out = torch.zeros_like(p)
        out[int(p.argmax().item())] = 1.0
        return out.unsqueeze(0)
    if label_type == 'full_soft':
        return (p / p.sum().clamp(min=1e-12)).unsqueeze(0)
    k = max(1, min(int(cfg.get('topk', 2)), p.numel()))
    idx = p.topk(k=k).indices
    mask = torch.zeros_like(p)
    mask[idx] = 1.0
    out = p * mask
    out = out / out.sum().clamp(min=1e-12)
    return out.unsqueeze(0)



def make_negative_label(prob: torch.Tensor, cfg: dict) -> torch.Tensor:
    label_type = cfg.get('label_type', 'bottomk_complement')
    p = prob.squeeze(0).detach().clone()
    if label_type == 'threshold_mask':
        lower = float(cfg.get('mask_threshold', {}).get('lower', 0.03))
        upper = float(cfg.get('mask_threshold', {}).get('upper', 1.0))
        out = ((p > lower) & (p < upper)).float()
        if out.sum() <= 0:
            out[int(p.argmin().item())] = 1.0
        out = out / out.sum().clamp(min=1e-12)
        return out.unsqueeze(0)
    if label_type == 'one_hot_bottom1':
        out = torch.zeros_like(p)
        out[int(p.argmin().item())] = 1.0
        return out.unsqueeze(0)
    k = max(1, min(int(cfg.get('bottomk', 3)), p.numel()))
    idx = torch.topk(-p, k=k).indices
    mask = torch.zeros_like(p)
    mask[idx] = 1.0
    out = (1.0 - p) * mask
    out = out / out.sum().clamp(min=1e-12)
    return out.unsqueeze(0)



def make_update_target(update_type: str, p_text: torch.Tensor, p_final: torch.Tensor, maturity: float) -> torch.Tensor:
    if update_type == 'text_only':
        return p_text.detach()
    if update_type == 'fused_only':
        return p_final.detach()
    return ((1.0 - maturity) * p_text + maturity * p_final).detach()


@torch.no_grad()
def run_hybrid(cfg: dict, loader, clip_model, prompt_embeddings: torch.Tensor, dataset_name: str):
    method_cfg = cfg['method']
    runtime_cfg = cfg.get('logging', {})
    logit_scale = float(method_cfg.get('logit_scale', 100.0))
    num_classes = prompt_embeddings.size(0)

    init_proto = prompt_embeddings.mean(dim=1)
    init_proto = init_proto / init_proto.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    pos_cfg = method_cfg['cache']['positive']
    neg_cfg = method_cfg['cache']['negative']
    proto_cfg = method_cfg['prototype']
    fusion_cfg = method_cfg['fusion']

    pos_cache = ClasswiseCache(pos_cfg.get('enabled', True), int(pos_cfg.get('shot_capacity', 3)), method_cfg['cache'].get('replacement_policy', 'loss_sorted'))
    neg_cache = ClasswiseCache(neg_cfg.get('enabled', True), int(neg_cfg.get('shot_capacity', 2)), method_cfg['cache'].get('replacement_policy', 'loss_sorted'))
    prototypes = PrototypeMemory(proto_cfg.get('enabled', True), init_proto)

    accuracies = []
    timings = []
    peak_memory = 0.0

    for step_idx, (images, target) in enumerate(tqdm(loader, desc=f'Processed test images [{dataset_name}]')):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        tic = time.perf_counter()

        target = target.cuda()
        image_features = encode_image(images, clip_model)
        text_logits, _ = adaptive_prompt_logits(image_features, prompt_embeddings, method_cfg['prompt_filtering'], logit_scale)
        p_text = text_logits.softmax(dim=1)
        entropy = float(normalized_entropy(p_text).mean().item())
        pred = int(p_text.argmax(dim=1).item())

        cache_logits = torch.zeros_like(text_logits)
        if method_cfg['cache'].get('enabled', True):
            if len(pos_cache) > 0 and pos_cfg.get('enabled', True):
                cache_logits = cache_logits + pos_cache.logits(image_features, float(pos_cfg.get('alpha', 1.0)), float(pos_cfg.get('beta', 1.0)), num_classes)
            if len(neg_cache) > 0 and neg_cfg.get('enabled', True):
                neg_penalty = neg_cache.logits(image_features, float(neg_cfg.get('alpha', 1.0)), float(neg_cfg.get('beta', 1.0)), num_classes)
                cache_logits = cache_logits - float(neg_cfg.get('lambda', 1.0)) * neg_penalty

        proto_logits = prototypes.logits(image_features, float(proto_cfg.get('logit_scale', logit_scale)))

        maturity = maturity_weight(step_idx + 1, fusion_cfg)
        fusion_design = fusion_cfg.get('design', 'maturity_confidence')
        cache_weight = 0.0
        proto_weight = 0.0

        if fusion_design == 'fixed':
            cache_weight = float(fusion_cfg.get('fixed_cache_weight', 0.5)) if pos_cfg.get('enabled', True) or neg_cfg.get('enabled', True) else 0.0
            proto_weight = float(fusion_cfg.get('fixed_proto_weight', 0.5)) if proto_cfg.get('enabled', True) else 0.0
        else:
            cache_conf = confidence_from_logits(cache_logits) if (len(pos_cache) > 0 or len(neg_cache) > 0) else 0.0
            proto_conf = confidence_from_logits(proto_logits) if proto_cfg.get('enabled', True) else 0.0
            denom = cache_conf + proto_conf + 1e-12
            if fusion_design == 'confidence_only':
                cache_weight = cache_conf / denom
                proto_weight = proto_conf / denom
            elif fusion_design == 'maturity_only':
                base_cache = float(fusion_cfg.get('fixed_cache_weight', 0.5))
                base_proto = float(fusion_cfg.get('fixed_proto_weight', 0.5))
                base_denom = base_cache + base_proto + 1e-12
                cache_weight = maturity * (base_cache / base_denom)
                proto_weight = maturity * (base_proto / base_denom)
            else:
                cache_weight = maturity * (cache_conf / denom if denom > 0 else 0.0)
                proto_weight = maturity * (proto_conf / denom if denom > 0 else 0.0)

        final_logits = text_logits.clone()
        if method_cfg['cache'].get('enabled', True) and (len(pos_cache) > 0 or len(neg_cache) > 0):
            final_logits = final_logits + cache_weight * cache_logits
        if proto_cfg.get('enabled', True):
            final_logits = final_logits + proto_weight * proto_logits

        p_final = final_logits.softmax(dim=1)
        update_target = make_update_target(method_cfg['update_target'].get('type', 'mixed'), p_text, p_final, maturity)

        if pos_cfg.get('enabled', True):
            pos_entropy_thresh = float(pos_cfg.get('entropy_threshold', 0.2))
            if entropy <= pos_entropy_thresh:
                pos_label = make_positive_label(update_target, pos_cfg)
                pos_cache.add(pred, image_features, entropy, pos_label)

        if neg_cfg.get('enabled', True):
            lower = float(neg_cfg.get('entropy_threshold', {}).get('lower', 0.2))
            upper = float(neg_cfg.get('entropy_threshold', {}).get('upper', 0.5))
            if lower < entropy < upper:
                neg_label = make_negative_label(update_target, neg_cfg)
                neg_cache.add(pred, image_features, entropy, neg_label)

        prototypes.update(image_features, update_target, proto_cfg)

        acc = cls_acc(final_logits, target)
        accuracies.append(acc)

        elapsed = time.perf_counter() - tic
        timings.append(elapsed)
        if torch.cuda.is_available():
            peak_memory = max(peak_memory, torch.cuda.max_memory_allocated() / (1024 ** 2))

        if cfg['experiment'].get('wandb_log', False):
            payload = {
                'dataset_accuracy_running': sum(accuracies) / len(accuracies),
                'sample_entropy': entropy,
            }
            if runtime_cfg.get('track_runtime', True):
                payload['sample_runtime_sec'] = elapsed
                payload['avg_runtime_sec'] = sum(timings) / len(timings)
            if runtime_cfg.get('track_memory', True) and torch.cuda.is_available():
                payload['peak_memory_mb'] = peak_memory
            wandb.log(payload, commit=True)

        every = int(runtime_cfg.get('print_every', 1000))
        if step_idx % every == 0:
            print(f"---- Hybrid-TDA test accuracy: {sum(accuracies)/len(accuracies):.2f}. ----")

    metrics = {
        'accuracy': sum(accuracies) / len(accuracies),
        'avg_runtime_sec': sum(timings) / max(len(timings), 1),
        'peak_memory_mb': peak_memory,
    }
    print(f"---- Hybrid-TDA final test accuracy on {dataset_name}: {metrics['accuracy']:.2f}. ----")
    return metrics



def resolve_dataset_cfg(config: dict, dataset_name: str) -> dict:
    defaults = copy.deepcopy(config)
    overrides = defaults.pop('dataset_overrides', {})
    if dataset_name in overrides:
        defaults = deep_update(defaults, overrides[dataset_name])
    return defaults



def main():
    args = parse_args()
    config = load_yaml(args.config_file)
    experiment = config['experiment']

    random.seed(int(experiment.get('seed', 1)))
    torch.manual_seed(int(experiment.get('seed', 1)))

    clip_model, preprocess = clip.load(experiment['backbone'])
    clip_model.eval()

    if experiment.get('wandb_log', False):
        date = datetime.now().strftime('%b%d_%H-%M-%S')
        group_name = f"{experiment['backbone']}_{experiment['datasets']}_{date}"

    for dataset_name in experiment['datasets'].split('/'):
        dataset_cfg = resolve_dataset_cfg(config, dataset_name)
        print(f'Processing {dataset_name} dataset.')
        print('\nRunning hybrid dataset configuration:')
        print(dataset_cfg, '\n')

        test_loader, classnames, template = build_test_data_loader(dataset_name, experiment.get('data_root', './dataset/'), preprocess)
        prompt_embeddings = build_prompt_embeddings(classnames, template, clip_model)

        if experiment.get('wandb_log', False):
            run = wandb.init(
                project=experiment.get('project', 'Hybrid-TDA'),
                config=dataset_cfg,
                group=group_name,
                name=f"{experiment.get('run_name_prefix', 'hybrid')}_{dataset_name}",
            )

        metrics = run_hybrid(dataset_cfg, test_loader, clip_model, prompt_embeddings, dataset_name)

        if experiment.get('wandb_log', False):
            log_payload = {f'{dataset_name}/accuracy': metrics['accuracy']}
            if dataset_cfg.get('logging', {}).get('track_runtime', True):
                log_payload[f'{dataset_name}/avg_runtime_sec'] = metrics['avg_runtime_sec']
            if dataset_cfg.get('logging', {}).get('track_memory', True):
                log_payload[f'{dataset_name}/peak_memory_mb'] = metrics['peak_memory_mb']
            wandb.log(log_payload)
            run.finish()


if __name__ == '__main__':
    main()
