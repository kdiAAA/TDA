import os

from .utils import DatasetBase
from .oxford_pets import OxfordPets

template = ['a photo of {}, a type of food.']

class Food101(DatasetBase):

    dataset_dir = 'food-101'

    def __init__(self, root):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'images')
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_Food101.json')
        
        self.template = template

        test = OxfordPets.read_split(self.split_path, self.image_dir)

        super().__init__(test=test)