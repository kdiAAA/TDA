# Efficient Test-Time Adaptation of Vision-Language Models
> [**Efficient Test-Time Adaptation of Vision-Language Models**]()<br>
> [Adilbek Karmanov](https://scholar.google.com/citations?user=R7GNWu0AAAAJ&hl=en), [Dayan Guan](https://dayan-guan.github.io/), [Shijian Lu](https://scholar.google.com/citations?hl=en&user=uYmK-A0AAAAJ&view=en), [Abdulmotaleb El Saddik](https://scholar.google.ca/citations?user=VcOjgngAAAAJ&hl=en), [Eric Xing](https://scholar.google.com/citations?user=5pKTRxEAAAAJ&hl=en)

Official implementation of the paper: "Efficient Test-Time Adaptation of Vision-Language Models" [CVPR 2024].

## Overview
![abstract figure](docs/main_figure.png)
> **<p align="justify"> Abstract:** Test-time adaptation with pre-trained vision-language models has attracted increasing attention for tackling distribution shifts during the test time. Though prior studies have achieved very promising performance, they involve intensive computation which is severely unaligned with test-time adaptation. We design TDA, a training-free dynamic adapter that enables effective and efficient test-time adaptation with vision-language models. TDA works with a lightweight key-value cache that maintains a dynamic queue with few-shot pseudo labels as values and the corresponding test-sample features as keys. Leveraging the key-value cache, TDA allows adapting to test data gradually via progressive pseudo label refinement which is super-efficient without incurring any backpropagation. In addition, we introduce negative pseudo labeling that alleviates the adverse impact of pseudo label noises by assigning pseudo labels to certain negative classes when the model is uncertain about its pseudo label predictions. Extensive experiments over two benchmarks demonstrate TDA’s superior effectiveness and efficiency as compared with the state-of-the-art.

## Main Contributions
In summary, the contributions of this work are threefold: </br>

* **First**, we design a training-free dynamic adapter (TDA) that can achieve test-time adaptation of vision-language models efficiently and effectively. To the best of our knowledge, this is the first work that investigates the efficiency issue of test-time adaptation of vision-language models. </br>
* **Second**, we introduce negative pseudo labeling to alleviate the adverse impact of pseudo label noises which makes TDA more robust to pseudo label noises and generalizable to testing data. </br>
* **Third**, we evaluate TDA extensively over two benchmarks, and experiments show that TDA achieves superior accuracy and efficiency compared with the state-of-the-art. </br>

## Requirements 
### Installation
Follow these steps to set up a conda environment and ensure all necessary packages are installed:

```bash
git clone https://github.com/kdiAAA/TDA.git
cd TDA

conda create -n tda python=3.8
conda activate tda

pip install -r requirements.txt

# Install specified PyTorch, torchvision, and matching CUDA toolkit versions
conda install pytorch torchvision cudatoolkit
```

### Dataset
To set up all required datasets, kindly refer to the guidance in DATASETS.md, which incorporates steps for two benchmarks.

## Run TDA
### Configs
The configuration for TDA hyperparameters in `configs/dataset.yaml` can be tailored within the provided file to meet the needs of various datasets. This customization includes settings for both the positive and negative caches as outlined below:
* **Positive Cache Configuration:** Adjustments can be made to the `shot_capacity`, `alpha`, and `beta` values to optimize performance.

* **Negative Cache Configuration:** Similar to the positive cache, the negative cache can also be fine-tuned by modifying the `shot_capacity`, `alpha`, `beta`, as well as the `entropy_threshold` and `mask_threshold` parameters.

For ease of reference, the configurations provided aim to achieve optimal performance across datasets on two benchmarks, consistent with the results documented in our paper. However, specific tuning of these parameters for negative cache could potentially unlock further enhancements in performance. Adjusting parameters like `alpha` and `beta` within the positive cache lets you fine-tune things to match the unique needs of each dataset.


