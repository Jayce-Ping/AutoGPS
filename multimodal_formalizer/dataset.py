"""
    This file contains the dataset class for the Geometry3K dataset
"""
import numpy as np
import os
import json

class Geometry3KDataset:
    """
        Dataset class for the Geometry3K dataset
    """
    def __init__(self, root_dir, phase = 'test', annotation_file=None):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images')

        if annotation_file:
            self.annotation_filepath = annotation_file
        else:
            self.annotation_filepath = os.path.join(root_dir, f'geometry3k_{phase}.json')
    
        if not os.path.exists(self.annotation_filepath):
            raise ValueError(f'Annotation file not found at {self.annotation_filepath}')
    
        with open(self.annotation_filepath, 'r') as f:
            self.annotation = json.load(f)


    def __getitem__(self, index):
        if str(index) not in self.annotation:
            raise ValueError(f'Index {index} not found in annotation file {self.annotation_filepath}.')
        
        data = self.annotation[str(index)]
        
        data['image_path'] = os.path.join(self.root_dir, 'images', f"{index}_diagram.png")
        return data
    
    def get_problem_ids(self):
        """
            Returns the list of problem ids that are present in the dataset
        """
        return list(self.annotation.keys())

    def __len__(self):
        return 3002


class PGPS9KDataset:
    """
        Dataset class for the PGPS9K dataset
    """
    def __init__(self, root_dir, phase = 'test', annotation_file = None):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images')
        self.phase = phase

        if annotation_file:
            self.annotation_filepath = annotation_file
        else:
            self.annotation_filepath = os.path.join(root_dir, f'pgps9k_{phase}.json')

        if not os.path.exists(self.annotation_filepath):
            raise ValueError(f'Annotation file not found at {self.annotation_filepath}')
        
        with open(self.annotation_filepath, 'r') as f:
            self.annotation = json.load(f)

    def __getitem__(self, index):
        if str(index) not in self.annotation:
            raise ValueError(f'Index {index} not found in annotation file {self.annotation_filepath}.')
        
        data = self.annotation[str(index)]
        
        data['image_path'] = os.path.join(self.root_dir, 'images', f"{index}_diagram.png")
        return data

    def get_problem_ids(self):
        """
            Returns the list of problem ids that are present in the dataset
        """
        return list(self.annotation.keys())

    def __len__(self):
        if self.phase == 'train':
            return 8021
        elif self.phase == 'test':
            return 1000
        else:
            return 9021