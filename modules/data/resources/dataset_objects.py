from abc import abstractmethod
import numpy as np
import pandas as pd
import os 
from . import config as cfg
# import config as cfg


'''
Abstract Class for Dataset. Used as an interface for the different datasets
'''
class IDataset:
    def __init__(self, name: str, dataset_path: str, nc_label: str, description: str):
        self.name = name
        self.path = dataset_path
        self.image_paths = []
        self.labels = []
        self.description = description

    def __str__(self):
        return f"{self.name} Dataset"

    @abstractmethod
    def get_filepaths_labels(self, return_binary: bool = False):
        pass

    def _wrap_images_and_labels(self):
        return (np.array(self.image_paths), np.array(self.labels))

    def check_exists(self):
        return self.path is not None

    def to_markdown(self):
        return f"{self.name} | {self.description}"

    def add_image_paths(self, image_paths):
        self.image_paths = image_paths

    def add_labels(self, labels):
        self.labels = labels


'''
Specific Dataset Interfaces for Specific Datasets
'''
class CRICDataset(IDataset):
    def __init__(self):
        if os.path.exists(cfg.CRIC_DATASET_PATH):
            dataset_path = cfg.CRIC_DATASET_PATH
        else:
            dataset_path = None

        super().__init__(
            name = "CRIC",
            dataset_path=dataset_path,
            nc_label = cfg.CRIC_NC_LABEL,
            description = "CRIC Full Dataset"
        )
    

    def get_filepaths_labels(self):
        for root, dirs, files in os.walk(self.path, topdown=True):
            for d in dirs:
                for file in os.listdir(os.path.join(root, d)):
                    if file == 'Results.csv':
                        continue
                    self.image_paths.append(os.path.join(root, d, file))
                    self.labels.append(file.split('_')[0])

        # Post Processing
        return self._wrap_images_and_labels()


class MENDDataset(IDataset):
    def __init__(self):
        if os.path.exists(cfg.MEND_DATASET_PATH):
            dataset_path = cfg.MEND_DATASET_PATH
        else: 
            dataset_path = None

        super().__init__(
            name = "MEND",
            dataset_path=dataset_path,
            nc_label = cfg.MEND_NC_LABEL,
            description = "Mendeley-Hussain Dataset"
        )

    def get_filepaths_labels(self):
        for root, dirs, files in os.walk(self.path, topdown=True):
            for d in dirs:
                for file in os.listdir(os.path.join(root, d)):
                    if file == 'Results.csv':
                        continue
                    self.image_paths.append(os.path.join(root, d, file))
                    self.labels.append(file.split('_')[0])

        # Post Processing
        return self._wrap_images_and_labels()


class CDetectorDataset(IDataset):
    def __init__(self):
        if os.path.exists(cfg.CDET_DATASET_PATH):
            dataset_path = cfg.CDET_DATASET_PATH
        else: 
            print('not found')
            dataset_path = None

        self.train_path = os.path.join(dataset_path,'Train')
        self.test_path = os.path.join(dataset_path,'Test')

        super().__init__(
            name = "CDET",
            dataset_path=dataset_path,
            nc_label = cfg.CDET_NC_LABEL,
            description = "CDET Full Dataset"
        )
        
    def get_filepaths_labels(self):
        for root, dirs, files in os.walk(self.train_path, topdown=True):
            for d in dirs:
                label = d
                for file in os.listdir(os.path.join(root, d)):
                    if file == 'Results.csv':
                        continue
                    self.image_paths.append(os.path.join(root, d, file))
                    #self.labels.append(file.split('_')[0])
                    self.labels.append(label)
        
        for root, dirs, files in os.walk(self.test_path, topdown=True):
            for d in dirs:
                label = d
                for file in os.listdir(os.path.join(root, d)):
                    if file == 'Results.csv':
                        continue
                    self.image_paths.append(os.path.join(root, d, file))
                    #self.labels.append(file.split('_')[0])
                    self.labels.append(label)

        # Post Processing
        return self._wrap_images_and_labels()
        
class ASCYDataset(IDataset):
    def __init__(self):
        if os.path.exists(cfg.ASCY_DATASET_PATH):
            dataset_path = cfg.ASCY_DATASET_PATH
        else: 
            print('not found')
            dataset_path = None


        super().__init__(
            name = "ASCY",
            dataset_path=dataset_path,
            nc_label = cfg.ASCY_NC_LABEL,
            description = "ASCY"
        )
        
    def get_filepaths_labels(self):
        for root, dirs, files in os.walk(self.path, topdown=True):
            for d in dirs:
                label = d
                for file in os.listdir(os.path.join(root, d)):
                    if file == 'Results.csv':
                        continue
                    self.image_paths.append(os.path.join(root, d, file))
                    #self.labels.append(file.split('_')[0])
                    self.labels.append(label)
        
        # Post Processing
        return self._wrap_images_and_labels()

class IARCDataset(IDataset):
    def __init__(self):
        if os.path.exists(cfg.IARC_DATASET_PATH):
            dataset_path = cfg.IARC_DATASET_PATH
        else: 
            print('not found')
            dataset_path = None


        super().__init__(
            name = "IARC",
            dataset_path=dataset_path,
            nc_label = cfg.IARC_NC_LABEL,
            description = "IARC"
        )
        
    def get_filepaths_labels(self):
        for root, dirs, files in os.walk(self.path, topdown=True):
            for d in dirs:
                label = d
                for file in os.listdir(os.path.join(root, d)):
                    if file == 'Results.csv':
                        continue
                    self.image_paths.append(os.path.join(root, d, file))
                    #self.labels.append(file.split('_')[0])
                    self.labels.append(label)
        
        # Post Processing
        return self._wrap_images_and_labels()

        



if __name__ == '__main__':
    cd = CDetector()
    cd.get_filepaths_labels()
