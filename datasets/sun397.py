import os

from .utils import DatasetBase

from .oxford_pets import OxfordPets


template = ["itap of a {}.",
            "a bad photo of the {}.",
            "a origami {}.",
            "a photo of the large {}.",
            "a {} in a video game.",
            "art of the {}.",
            "a photo of the small {}."]

class SUN397(DatasetBase):

    dataset_dir = 'sun397'

    def __init__(self, root):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'SUN397')
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_SUN397.json')

        self.template = template

        test = OxfordPets.read_split(self.split_path, self.image_dir)

        super().__init__(test=test)
