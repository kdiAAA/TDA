import os

from .oxford_pets import OxfordPets
from .utils import DatasetBase


template = ['a photo of a {}, a type of flower.']

class OxfordFlowers(DatasetBase):

    dataset_dir = 'oxford_flowers'

    def __init__(self, root):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'jpg')
        self.label_file = os.path.join(self.dataset_dir, 'imagelabels.mat')
        self.lab2cname_file = os.path.join(self.dataset_dir, 'cat_to_name.json')
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_OxfordFlowers.json')

        self.template = template

        test = OxfordPets.read_split(self.split_path, self.image_dir)
        
        super().__init__(test=test)