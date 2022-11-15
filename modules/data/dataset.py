import os
import typing

import numpy as np
import pandas as pd
from PIL import Image

import torch


class CervData:
    def __init__(
        self, 
        data_dir: str,
        df: pd.DataFrame,
        label2ind: typing.Dict[str, int] = None,
        preprocessor: callable = None,
        shuffle: bool = False,
        random_seed: int = None,
        device: str = 'cuda',
        multi_label: bool = False,
        multi_label_sep: str = ';',
        convert_to_grayscale: bool = False
    ) -> None:
        """
        A Custom Dataset Object that holds the Images to be used for Analysis and Modelling.

        Parameters:
        -----------
        data_dir: str
            The path to the directory containing the images.
        df: pandas.DataFrame
            The dataframe containing the image paths and labels.
        preprocessor: callable
            A function that preprocesses the images.
            Function must take in a PIL.Image object and return a torch.Tensor.
        label2ind: dict (optional)
            A dictionary mapping the labels to their corresponding number.
            Is responsible for Label Encoding the labels.
            If not provided then the dictionary is created
        shuffle: bool (default: False)
            Whether to shuffle the dataset.
        random_seed: int (default: None)
            The random seed to use for shuffling.
        device: str (default: 'cuda')
            The device where the data will be stored.
        multi_label: bool (default: False)
            Whether the labels are multi-label
        multi_label_sep: str (default: ';')
            How labels are separated in the multi-label setting.
            Ignored if multi_label = False
        convert_to_grayscale: bool (default: False)
            Whether to convert images to grayscale
        """
        # Resources
        self.df = df
        self.data_dir = data_dir
        self.num_examples = len(self.df)
        self.preprocessor = preprocessor
        self.device = device
        self.random_seed = random_seed
        self.shuffle = shuffle
        self.convert_to_grayscale = convert_to_grayscale

        # Process Resources
        if shuffle:
            self._shuffle(random_seed=random_seed)
        
        # Label Metadata Map
        self.multi_label = multi_label
        self.multi_label_sep = multi_label_sep
        if label2ind is not None:
            self.label2ind = label2ind
        else:
            self.label2ind = self._enumerate_labels()
        self.ind2label = {v: k for k, v in self.label2ind.items()}

        # Misc
        self.num_labels = len(self.label2ind)
    
    def __str__(self):
        return f'''
        CervData Object
        --------
        
        Properties:
        -----------
        data_dir:             {self.data_dir}
        preprocessor:         {'Set' if self.preprocessor is not None else 'Not Set'}
        shuffle:              {self.shuffle}
        random_seed:          {self.random_seed}
        convert_to_grayscale  {self.convert_to_grayscale}

        Internal Properties:
        ---------------------
        num_labels:           {self.num_labels}
        num_examples:         {self.num_examples}
        multi_label:          {self.multi_label}
        '''
        
    def get_batch(
        self,
        s_idx: int,
        e_idx: int, 
        return_labels: bool = True
    ) -> typing.Dict[str, torch.Tensor]:
        """
        Returns a batch of images (and labels) depending on the index.
        Is a wrapper for the _get_batch_images and _get_batch_labels methods.

        Parameters:
        -----------
        s_idx: int
            The start index of the batch in the dataset
        e_idx: int
            The end index of the batch in the dataset
        return_labels: bool (default: True)
            A flag to indicate whether to return the labels or not

        Returns:
        --------
        input_dict: dict
            A dictionary containing 'x' and 'y' as keys.
            x: torch.Tensor
                The Image Tensor from s_idx to e_idx
            y: torch.Tensor
                The Label Tensor from s_idx to e_idx
        """
        input_dict = self._get_batch_images(s_idx, e_idx)
        if return_labels:
            input_dict['targets'] = self._get_batch_labels(s_idx, e_idx)
        return input_dict

    def sample(self, amount: int) -> typing.Dict[str, typing.List]:
        """
        Returns a random sample of the dataset of size amount.

        Parameters:
        -----------
        amount: int
            The amount of images to return

        Returns:
        --------
        sample: dict
            A dictionary containing the images and labels.
        """
        batch_image_idx = [np.random.randin(0, self.num_examples) for _ in range(amount)]
        batch_images = [
            Image.open(os.path.join(self.data_dir, fn)) 
            for fn in self.df.iloc[batch_image_idx, 'relative_path']
        ]
        batch_labels = self.df.iloc[batch_image_idx, 'label'].values.tolist()
        return {
            'images': batch_images,
            'labels': batch_labels
        }

    def _get_batch_images(self, s_idx: int, e_idx: int) -> typing.Dict[str, torch.Tensor]:
        if self.preprocessor is None:
            raise Exception("No Preprocessor Specified! Please Specify a Preprocessor.")
        '''
        Returns a batch of images depending on the index.

        Parameters:
        -----------
        s_idx: int
            The start index of the batch in the dataset.
        e_idx: int
            The end index of the batch in the dataset.

        Returns:
        --------
        images: Dict[str, torch.Tensor]
            A dictionary containing the preprocessed image tensors as a unified tensor 
            placed on the device.
        '''
        batch_image_paths = [
            os.path.join(self.data_dir, fn) for fn in self.df['relative_path'][s_idx:e_idx]
        ]
        batch_images = []
        for fn in batch_image_paths:
            image = Image.open(fn)
            if self.convert_to_grayscale:
                image = image.convert('L')
            image_arr = np.array(image)
            # Expand dimensions for grayscale images
            if len(image_arr.shape) == 2:
                image_arr = np.repeat(image_arr[..., np.newaxis], 3, -1)
                image = Image.fromarray(image_arr)
            batch_images.append(image)
        batch_images = [self.preprocessor(img).unsqueeze(0) for img in batch_images]
        batch_images = torch.cat(batch_images)
        batch_images = batch_images.to(self.device)
        return {'x': batch_images}
    
    def _get_batch_labels(self, s_idx: int, e_idx: int) -> torch.Tensor:
        """
        Returns a batch of labels depending on the index.

        Parameters:
        -----------
        s_idx: int
            The start index of the batch in the dataset.
        e_idx: int
            The end index of the batch in the dataset.

        Returns:
        --------
        labels: torch.Tensor
            A tensor containing the labels of the batch.
        """
        batch_labels = self.df['label'][s_idx:e_idx]
        batch_labels_arr = torch.zeros(
            (len(batch_labels), self.num_labels),
            device=self.device
        )
        for i, label in enumerate(batch_labels):
            if self.multi_label:
                for label_ in label.split(self.multi_label_sep):
                    batch_labels_arr[i, self.label2ind[label_]] = 1
            else:
                batch_labels_arr[i, self.label2ind[label]] = 1
        return batch_labels_arr

    def _shuffle(self, random_seed: int = None) -> None:
        """Shuffles the dataset."""
        self.df = self.df.sample(frac=1, replace=False, random_state=random_seed).copy()

    def _enumerate_labels(self):
        if self.multi_label:
            labels = set()
            for label_str in self.df['label'].drop_duplicates():
                for label in label_str.split(self.multi_label_sep):
                    labels.add(label)
            labels = sorted(list(labels))
        else:
            labels = self.df['label'].drop_duplicates()
        label2ind = {
            label: i for i, label in enumerate(labels)
        }
        return label2ind

    def set_preprocessor(self, preprocessor: callable) -> None:
        """Sets the preprocessor for the dataset."""
        self.preprocessor = preprocessor
