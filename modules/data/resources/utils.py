'''
Author(s): Leon, Paolo 

Contains Utils that aims to provide functions to combine and process the dataset.
'''
import os
from abc import abstractmethod
import typing
from pathlib import Path 
import shutil
import time

import numpy as np
from PIL import Image
import cv2 as cv

import matplotlib.pyplot as plt

from . import config as cfg

  
'''
CervAI Objects
'''
class CervObject():
    def __init__(self, name: str):
        self.name = name


'''
Transformer Wrapper 
'''
class ImageProcessor(CervObject):
    def __init__(self, name: str, process_func: callable, probability: int, priority: int):
        super().__init__(name)
        self.process_func = process_func
        self.enabledOverride = None

        if (probability > 100 or probability < 0):
            raise Exception("Probability must be between 0 and 100")

        self.probability = probability # out of 100
        self.priority = priority

    def is_enabled(self):
        if self.enabledOverride is not None:
            return self.enabledOverride
        return False

    def process_image(self, image: np.array):
        '''
        Processes the image and returns the transformed image
        '''
        return self.process_func(image)

    def process_batch(self, images: np.array):
        batch_images = []
        for img in images:
            batch_images = batch_images + [self.process_image(img)]
        return batch_images

    def test(self, grayscale=False):
        '''
        Only used for Testing. Will not be used in Production
        '''
        oimg = cv.imread(os.path.join(cfg.CRIC_DATASET_PATH, 'ASC_H', 'ASC-H_3.png'))
        timg = self.process_image(oimg)

        plt.subplot(1, 2, 1)
        plt.imshow(oimg)
        plt.title("Original Image")
        plt.subplot(1, 2, 2)
        if (grayscale):
            plt.imshow(timg, cmap='gray')
        else:
            plt.imshow(timg)
        plt.title("Transformed Image")
        plt.show()

    def to_markdown(self):
        return f"{self.name} | {self.priority} | {self.probability}"


class CervTransformer(ImageProcessor):
    def __init__(self, name: str, transfunc: callable, probability: int):
        super().__init__(name + " Transformer", process_func=transfunc, probability=probability, priority=2)

    def __str__(self):
        return f"{self.name}\t\t|\t{self.is_enabled()}\t|\t{self.priority}\t|\t{self.probability}"


class CervAugmentor(ImageProcessor):
    def __init__(self, name: str, augment_func: callable, probability: int):
        super().__init__(name + " Augmentor", process_func=augment_func, probability=probability, priority=1)
    
    def __str__(self):
        return f"{self.name}\t\t|\t{self.is_enabled()}\t|\t{self.priority}\t|\t{self.probability}"


'''
Processing and Combination Pipeline (Performs ETL)
'''
class CervProcessingConfig(CervObject):
    def __init__(self, 
            shuffle: bool, label_map: typing.Dict[str, int],
            save_path: str, batch_size: int, reverse_label_map = None
        ):
        super().__init__("Cerv Processing Config Object")
        self.shuffle = True
        self.label_map = label_map
        self.reverse_label_map = reverse_label_map if reverse_label_map else {v: k for k, v in label_map.items()}
        self.ulabels = list(map(lambda x: x.upper(), reverse_label_map.values()))
        self.save_path = save_path
        self.batch_size = batch_size

    @staticmethod
    def get_new():
        '''
        Return the Default Cerv Processing Config
        '''
        return CervProcessingConfig(
            shuffle=True,
            label_map={"NL": 0, 'NILM': 0, "ASC-H": 1, "ASC-US": 2, "LSIL": 3, "HSIL": 4, "SCC": 5},
            reverse_label_map={0: "NL", 1: "ASC-H", 2: "ASC-US", 3: "LSIL", 4: "HSIL", 5: "SCC"},
            save_path=cfg.PROCESSED_PATH,
            batch_size = 32
        )

    def to_markdown(self):
        md = ""
        md += f"Labels| {self.ulabels}\n"
        md += f"Batch Size| {self.batch_size}\n"
        return md


class CervDataLoader(CervObject):
    def __init__(self, images, labels):
        super().__init__("Cerv Data Loader")
        if (len(images) != len(labels)):
            raise Exception("Images and Label Sizes do not Match!")
        self.image_paths = images
        self.labels = labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, key):
        try:
            return (self.image_paths[key], self.labels[key])
        except IndexError:
            return None 

    def get_batch(self, batch_size: int):
        for i in range(0, len(self), batch_size):
            if i+1 >= len(self):
                break

            # Construct Batch
            batch = []
            for b in range(batch_size):
                val = self.__getitem__(i + b)
                if val is None:
                    break
                batch.append(val)
            yield (i//batch_size, batch)


class CervProcessingPipeline(CervObject):
    def __init__(self, datasets: typing.List[IDataset], transformers: typing.List[ImageProcessor] = None, config: CervProcessingConfig = None):
        super().__init__("Cerv Processing Pipeline")
        self.transformers = sorted(transformers, key=lambda x: x.priority)
        self.datasets = datasets
        self.processing_config = CervProcessingConfig.get_new() if not config else config

    # For Manipulating the Transformers
    def override_transformer(self, name: str):
        for t in self.transformers:
            if t.name == name:
                if t.enabledOverride is None:
                    t.enabledOverride = True
                else:
                    t.enabledOverride = not t.enabledOverride
                print(f"Toggled {name} Successfully!")
                return
        print(f"{name} Not Found!")

    def show_transformers_and_augmentors(self):
        print("Transformers and Augmentors")
        print("Name\t\t\t\t|\tEnabled\t|\tPriority\t|\tProbability")
        for t in self.transformers:
            print(t)

    # Check Database Existence
    def check_datasets_exist(self):
        missing_ds = []
        for dataset in self.datasets:
            if not dataset.check_exists():
                missing_ds.append(str(dataset))
        
        if missing_ds:
            raise OSError(f"Pipeline is missing Datasets!: {missing_ds}")

        print(f"Datasets Found! {self.datasets}")

    # Manipiulate Data Source
    def add_datasource(self, dataset: IDataset):
        self.datasets.append(dataset)

    def remove_dataset_by_name(self, name: str):
        to_delete = list(filter(lambda x: x.name == name, self.datasets))
        if not to_delete:
            raise Exception(f"Dataset with name {name} not found")
        print(f"Deleted Dataset {to_delete[0].name} Successfully!")

    # Manipulate Config
    def set_processing_config(self, config: CervProcessingConfig):
        self.processing_config = config
    
    def get_processing_config(self):
        return self.processing_config

    # ETL Functions
    def perform_etl(self):
        start_time = time.time()

        # Extract Image_paths and labels
        images, labels = self.extract_from_datasets()
        # Perform Processing per Batch
        dl = CervDataLoader(images, labels)
        for i, batch in dl.get_batch(self.processing_config.batch_size):
            print()
            print('-'*20)
            print(f"Processing Batch {i}...")
            images, labels = self.transform_images(batch)
            labels = list(map(lambda x: self.processing_config.label_map[x.upper()], labels))
            self.load_into_new_dataset(images, labels, i)
            print(f"Processed Batch {i} Successfully!")
            print('-'*20)

        # Create Metadata
        self.create_metadata(start_time)

    def extract_from_datasets(self):
        '''
        Returns the Images as Image File Paths and Labels as Labels
        '''
        images = []
        labels = []
        for dataset in self.datasets:
            img, lbl = dataset.get_filepaths_labels()
            images.append(img)
            labels.append(lbl)

        # Post Processing 
        images = np.concatenate(images)
        labels = np.concatenate(labels)
        return images, labels

    def transform_images(self, batch):
        # Apply Transformers (In Sequence)
        processed_imgs = [cv.imread(img) for img, _ in batch]
        processed_labels = [label for _, label in batch]
        for transformer in filter(lambda x: x.is_enabled(),self.transformers):
            if np.random.randint(0, 100) <= transformer.probability:
                print('Performing Transformation: ' + transformer.name)
                if type(transformer) == CervAugmentor:
                    processed_imgs += transformer.process_batch(processed_imgs)
                    processed_labels += [label for _, label in batch]
                elif type(transformer) == CervTransformer:
                    processed_imgs = transformer.process_batch(processed_imgs)
        return processed_imgs, processed_labels

    def load_into_new_dataset(self, images: typing.List[np.array], labels: typing.List, batch: int):
        # Check if save folder exists
        if not os.path.exists(self.processing_config.save_path):
            os.mkdir(self.processing_config.save_path)
            for ulabel in self.processing_config.ulabels:
                os.mkdir(os.path.join(self.processing_config.save_path, ulabel))
        
        # Save Images
        for i, (img, label) in enumerate(zip(images, labels)):
            im = Image.fromarray(img)
            im.show
            lb = self.processing_config.reverse_label_map[label]
            im.save(os.path.join(self.processing_config.save_path, lb, f"{lb}_{batch}_{i}.png"))

    def create_metadata(self, start_time):
        # Create text file
        with open(cfg.METADATA_PATH, 'w') as md:
            with open(os.path.join(os.path.dirname(__file__), 'MetaDatatemplate.md'), 'r') as template:
                content = template.read()

                for dataset in self.datasets:
                    content = content.replace("{Datasets}", dataset.to_markdown() + "\n{Datasets}")
                
                content = content.replace("{Datasets}", "")

                for transformer in filter(lambda x: x.is_enabled(), self.transformers):
                    content = content.replace("{Transformers}", transformer.to_markdown() + "\n{Transformers}")

                content = content.replace("{Transformers}", "")

                content = content.replace("{Config}", self.processing_config.to_markdown())

                content = content.replace("{Execute Time}", str(start_time))
            md.write(content)

    def check_cervAI_folder_exists(self):
        if os.path.exists(cfg.PROCESSED_PATH):
            print('CervAI folder already present, confirm delete existing folder? (y/n)')
            user_answer = input()
            if user_answer == 'y':
                shutil.rmtree(cfg.PROCESSED_PATH)
            else:
                print('Add images on top of existing images (y/n)')
                user_answer2 = input()
                if user_answer2 == 'n':
                    print('nothing to do')
                    quit()
        else:
            print('will make new CervAI folder')

    def get_class_counts(self):
        for d in os.listdir(cfg.PROCESSED_PATH):
            if d == "MetaData.md":
                continue
            print(f'folder {d} contains {len(os.listdir(os.path.join(cfg.PROCESSED_PATH, d)))} images')


