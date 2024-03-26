import os

from .oxford_pets import OxfordPets
from .utils import DatasetBase


template = ["itap of a {}.",
            "a bad photo of the {}.",
            "a origami {}.",
            "a photo of the large {}.",
            "a {} in a video game.",
            "art of the {}.",
            "a photo of the small {}."]

class StanfordCars(DatasetBase):

    dataset_dir = 'stanford_cars'

    def __init__(self, root):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_StanfordCars.json')

        self.template = template

        test = OxfordPets.read_split(self.split_path, self.dataset_dir)

        super().__init__(test=test)