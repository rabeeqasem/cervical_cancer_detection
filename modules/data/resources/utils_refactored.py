import os
from pathlib import Path

from collections import Counter
import uuid

import shutil
import numpy as np
import tensorflow as tf
import albumentations as A
import random

import cv2


from . import config as cfg
from transformers import transformers as tr

class CervObject():
    def __init__(self, name: str):
        self.name = name


class CervPair(CervObject):
    def __init__(self, image: str, label: str):
        super().__init__("Cerv Pair")
        self.__source = image
        self.__image = image
        self.__label = label
        dirname = os.path.dirname(self.__image).split('/')[-2].upper()
        if dirname.startswith('TEST') or dirname.startswith('TRAIN'):
            self.__source_ds = 'CDET'
        else:
            self.__source_ds = dirname[:4]
        
        self.__std_name = None
    
    def get_image_path(self):
        return self.__image

    def get_source(self):
        return self.__source
    
    def get_source_ds(self):
        return self.__source_ds

    def set_path(self, path: str):
        self.__image = path

    def get_image_name(self):
        return os.path.basename(self.__image)
        
    def get_image_name_stripped_old(self):
        stripped = os.path.basename(self.__image).split('_')[1:]
        stripped = stripped.split('.')[0]
        return stripped
        
    def get_image_name_stripped(self):
        split_name = os.path.basename(self.__image).split('_')  
        if len(split_name) > 2:
            stripped = '_'.join(split_name[1:])
        else:
            stripped = split_name[-1]
        stripped = stripped.split('.')[0]
        return stripped
        

    def get_label(self):
        return self.__label

    def set_label(self, label: str):
        self.__label = label
    
    def set_std_name(self, std_name: str):
        self.__std_name = std_name

    def get_std_name(self):
        return self.__std_name


class CervAIDataset(CervObject):
    '''
    Represents the Dataset to be created by the CervProcessingPipeline
    '''
    def __init__(self, path: str = cfg.CERVAI_PATH):
        super().__init__("Cerv AI Dataset")
        self.__pairs = []
        self.__path = path

    def as_generator(self, partition=None, include_standardized_names=False):
        iterator = None
        if partition is None:
            iterator = zip(self.__image_paths, self.__labels)
        elif partition == 'train':
            iterator = zip(self.__train_images, self.__train_labels)
        elif partition == 'val':
            iterator = zip(self.__val_images, self.__val_labels)
        elif partition == 'test':
            iterator = zip(self.__test_images, self.__test_labels)
        else:
            raise Exception("Partition not found")

        iterator = list(iterator)

        if include_standardized_names:
            return list(zip(iterator, self.__standardized_names))
        else:
            return iterator

    def add_image_pair(self, pair: CervPair):
        self.__pairs.append(pair)

    def get_image_paths(self):
        return [pair.get_image_path() for pair in self.__pairs]

    def get_labels(self):
        return [pair.get_label() for pair in self.__pairs]

    def get_pairs(self):
        return self.__pairs

    def set_pairs(self, pairs):
        self.__pairs = pairs

    def morph_to_train_test(self, train_pairs: list, test_pairs: list, val_pairs: list):
        # Create Train partitions
        self.__train_pairs = train_pairs

        # Create Test partitions
        self.__test_pairs = test_pairs

        # Create Validation partitions
        self.__val_pairs = val_pairs

    def get_pairs_with_label(self, label: str):
        return [pair for pair in self.__pairs if pair.get_label() == label]
        
    def get_pairs_with_label_and_source(self, label: str, source_ds: str):
        return [pair for pair in self.__pairs if pair.get_label() == label and pair.get_source_ds() == source_ds]
        
    def get_partition(self, partition: str):
        if partition == 'train':
            return self.__train_pairs
        elif partition == 'test':
            return self.__test_pairs
        elif partition == 'val':
            return self.__val_pairs
        else:
            raise Exception("Invalid Partition Name")


class CervEnvironmentChecker(CervObject):
    '''
    A Component of a Pipeline that is reponsible for Setting the Environment
    '''
    def __init__(self, datasets: list):
        super().__init__("Cerv Environment Checker")
        self.datasets = datasets

    def setup_environment(self, dataset):
        self.check_datasets_existance(dataset)
        self.handle_cervai_generation()
        self.create_subfolders(False)

    def check_datasets_existance(self, datasets, force_run=True):
        '''
        Check if the data sources exist within local machine
        '''
        missing_ds = []
        for dataset in self.datasets:
            if not dataset.check_exists():
                missing_ds.append(str(dataset))
        
        if force_run and missing_ds:
            print(f"Pipeline Running without: {missing_ds}")
        elif missing_ds:
            raise Exception(f"Missing Datasets: {missing_ds}")
        else:
            print("All datasets present")

    def handle_cervai_generation(self):
        '''
        Handle CervAI Folder Generation and checks
        '''
        # Handle Existence of CERVAI Folder
        if os.path.exists(cfg.CERVAI_PATH):
            print('CervAI folder already present, confirm delete existing folder? (y/n) - ', end='')
            user_answer = input().lower()
            if user_answer == 'y':
                shutil.rmtree(cfg.CERVAI_PATH)
                print('Existing Folder deleted')
            else:
                print('Add images on top of existing images (y/n) - ', end='')
                user_answer2 = input()
                if user_answer2 == 'n':
                    raise Exception("Pipeline Procedure Terminated...")
        else:
            print('CervAI folder not present, creating folder...')

        # Create CervAI Folder
        #os.mkdir(cfg.CERVAI_PATH)
        Path(cfg.CERVAI_PATH).mkdir()
        print('Made new cervAi folder...')

    def create_subfolders(self, binary_labels: bool):
        # Creation of Folders if they don't exist
        if not os.path.exists(cfg.TRAIN_PATH):
            os.mkdir(cfg.TRAIN_PATH)

        if not os.path.exists(cfg.TEST_PATH):
            os.mkdir(cfg.TEST_PATH)

        if not os.path.exists(cfg.VAL_PATH):
            os.mkdir(cfg.VAL_PATH)

        # Creation of Subfolders for each class
        for folder in [cfg.TRAIN_PATH, cfg.TEST_PATH, cfg.VAL_PATH]:
            if binary_labels:
                os.mkdir(os.path.join(folder, 'NC'))
                os.mkdir(os.path.join(folder, 'CANC'))
                continue

            for class_label in cfg.CLASS_LABELS:
                os.mkdir(os.path.join(folder,class_label))


class CervPhase1Pipeline(CervObject):
    '''
    Perform ETL from various data sources and create a CervAIDataset. 
    '''
    def __init__(self, datasets):
        super().__init__("Cerv Processing Pipeline")
        self.datasets = datasets
        self.envchkr = CervEnvironmentChecker(datasets)
        
        dataset_names = []
        for ds in self.datasets:
            dataset_names.append(ds.name)
        self.__dataset_names = dataset_names
        
    def get_dataset_names(self):
        return self.__dataset_names

    # High Level Run    
    def run(self, upsample=False, perform_train_test_split=True):
        '''
        A high level overview of the pipeline
        '''
        # Setup Environment
        self.envchkr.setup_environment(self.datasets)

        # Create new Instance of CervAIDataset
        cervai = CervAIDataset()

        ## EXTRACT
        self.extract_paths_labels_from_source(cervai)

        ## TRANSFORMATION
        # Transform Labels
        self.transform_labels(cervai)
        
        # Standardize Image Format
        self.standardize_image_format(cervai)
        

        # Perform Train_test_split
        if perform_train_test_split:
            self.partition_dataset(cervai)
            
            # Upsample Train Set
            if upsample:
                #not working
                self.upsample_dataset(cervai)

        # LOADING
        self.move_images_to_cervai(cervai)

        print('\nCervAI Folder Created!')

    # Extract    
    def extract_paths_labels_from_source(self, dataset: CervAIDataset):
        '''
        Returns the Images as Image File Paths and Labels as Labels.
        Performs inplace Extraction to Dataset
        '''
        for ds in filter(lambda x: x.check_exists(), self.datasets):
            img, lbl = ds.get_filepaths_labels()
            for i, l in zip(img, lbl):
                if l.upper() in cfg.CLASS_LABELS_OMIT:
                    pass
                else:
                    dataset.add_image_pair(CervPair(i, l))
            print(f'Extracted {ds.name} paths and labels')
    
    # Transform
    def partition_dataset(self, dataset: CervAIDataset):
        '''
        Perform Random Train Test Images
        '''

        train = []
        test = []
        val = []
        for source_ds in self.get_dataset_names():
            for label in cfg.CLASS_LABELS:
                #corpus = dataset.get_pairs_with_label(label)
                corpus = dataset.get_pairs_with_label_and_source(label, source_ds)
        
                train_size = int(cfg.SPLIT[0] * len(corpus))
                test_size = int(cfg.SPLIT[1] * len(corpus))
                val_size = int(cfg.SPLIT[2] * len(corpus))
        
                train_indexes = np.random.choice(np.arange(len(corpus)), train_size, replace=False)
                test_indexes = np.setdiff1d(np.arange(len(corpus)), train_indexes)
        
                for i in train_indexes:
                    train.append(corpus[i])
        
                temp_test = []
                for i in test_indexes:
                    temp_test.append(corpus[i])
        
                test_indexes = np.random.choice(np.arange(len(temp_test)), test_size, replace=False)
                val_indexes = np.setdiff1d(np.arange(len(temp_test)), test_indexes)
        
                for i in test_indexes:
                    test.append(temp_test[i])
        
                for i in val_indexes:
                    val.append(temp_test[i])

        # Edit Attributes of the CervAIDataset
        dataset.morph_to_train_test(train, test, val)

    def upsample_dataset(self, dataset: CervAIDataset):
        #currently not working
        new_filepaths = []
        new_labels = []
        max_count = max(Counter(dataset.get_labels()).values())
        for current_label in cfg.CLASS_LABELS: #loop through class labels
            # Extract Indexes in array with current label
            indices = np.argwhere(dataset.train_labels==current_label).reshape(-1)
            
            # Select n Indexes that match the maximum label
            upsampled_indices = np.random.choice(indices, size=max_count, replace=True) #choose n indices that match label
            
            # Set filepaths and labels based on the upsampled indices
            upsampled_filepaths = dataset.image_paths[upsampled_indices]
            upsampled_labels = dataset.labels[upsampled_indices]

            # Append New Filepaths and Labels
            new_filepaths.extend(upsampled_filepaths)
            new_labels.extend(upsampled_labels)

        # Replace Filepaths and Labels
        dataset.replace_train_images(new_filepaths)
        dataset.replace_train_labels(new_labels)
    
    def transform_labels(self, dataset: CervAIDataset):
        '''
        Standards
        ---------
        - Image Paths must follow a certain format (.jpg)
        - Labels must be following the config Class Labels 
        '''
        # Transform Labels
        t_pairs = []
        for pair in dataset.get_pairs():
            label = pair.get_label().upper()
            if label.upper() in cfg.CLASS_LABELS_NILM:
                new_label = cfg.CLASS_LABELS[0] # Set to NILM
            else:
                new_label = label
            pair.set_label(new_label)
            t_pairs.append(pair)
        dataset.set_pairs(t_pairs)
           
    def standardize_image_format(self, dataset: CervAIDataset):
        '''
        Renames the Images to a standard format
        '''
        t_pairs = []
        for pair in dataset.get_pairs():
            #img_uuid = str(uuid.uuid4())
            #new_name = f'{pair.get_label()}_{img_uuid}.jpg'
            new_name = f'{pair.get_label()}_{pair.get_image_name_stripped()}_{pair.get_source_ds()}.jpg'
            new_path = os.path.join(cfg.CERVAI_PATH, new_name)
            pair.set_path(new_path)
            pair.set_std_name(new_name)
            t_pairs.append(pair)
        dataset.set_pairs(t_pairs)
        

    # Load
    def move_images_to_cervai(self, dataset: CervAIDataset):
        '''
        Moves the Images to the CervAI Folder
        '''
        
        transform_all = A.Compose([
            A.SmallestMaxSize(300, p=1)
            ])
        
        #transform_train_only = A.Compose([])

        
        for partition in ['train', 'test', 'val']:
            p_data = dataset.get_partition(partition)
            for pair in p_data:
                source = pair.get_source()
                name = pair.get_image_name()
                std_name = pair.get_std_name()
                label = pair.get_label()
                dest = os.path.join(cfg.CERVAI_PATH, partition, label, std_name)
                
                # Copy w/out transformations
                # shutil.copy(source, dest)
                
                # Do some transformation
                image = cv2.imread(source)
                random.seed(42) 
                augmented_image = transform_all(image=image)['image']
                cv2.imwrite(dest, augmented_image)

                #print(f'Copied {name} to {partition}/{label}/{std_name}', end='\r')
    
    # Not used
    def save_transformed_train_imgs(self, dataset: CervAIDataset):
        #below function has been made redundant


        # we have dataset of numpy arrays var(transformed_images) from fun(apply_transforms_to_train_imgs)
        # we have list of labels associated with each array: self.train_image_labels
        # TO DO: write transformed images to disk as efficiently as possible. 
        # @Paolo - possible to implement batching via tensorflow? (transformed_images.batch())

        #define list of output paths so they don't need to be generated on the fly
        
        # Move Image to CervAI Folder
        
        output_paths = []
        for x in range(len(self.train_image_paths)):
            temp_label = self.train_image_labels[x]
            temp_path = self.train_image_paths[x]
            temp_name = temp_path.split('/')[-1]
            temp_name = temp_name.replace(" ","")
            temp_name = temp_name.replace("(","")
            temp_name = temp_name.replace(")","")
            temp_name = temp_name.split('.')[-2]
            temp_output_path = os.path.abspath(os.path.join(self.train_path,temp_label,temp_name+'.png'))
            output_paths.append(temp_output_path)

        #BELOW NOT WORKING: save images in parralel
        def save_image_fn(np_array,output_path):
            tf.keras.utils.save_img(output_path,
                np_array)
            
        transformed_images_array.map(lambda x: save_image_fn(x,ouput_paths[x]),num_parallel_calls=tf.data.experimental.AUTOTUNE)
        print('images saved')
    
    def get_class_counts(self):
        for d in os.listdir(cfg.PROCESSED_PATH):
            if d == "MetaData.md":
                continue
            print(f'folder {d} contains {len(os.listdir(os.path.join(cfg.PROCESSED_PATH, d)))} images')

class CervPhase2Pipeline(CervObject):
    '''
    Apply preprocessing to images. 

    To do:
    Apply resizing retaining aspect ratio
    Apply transforms 
    '''
    def __init__(self, datasets):
       pass

    def run(self):
        self.transform_pipeline 

    def define_transform_pipeline(self):
        #to do test below:
        self.transform_pipeline = tr.create_transform_pipeline()

    def apply_transforms_to_train_imgs(self):
        #to do test below
        tr.transform_images_in_parallel(train_paths, train_labels, transform_pipeline)
