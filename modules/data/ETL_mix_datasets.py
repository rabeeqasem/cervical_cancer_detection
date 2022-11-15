'''
Author(s): Leon, Paolo, Gaylyn

Contains code for getting image filepaths and labels, transforming the images, and then 
saving the transformed images into a new folder
'''

import os

from resources import utils_refactored as utils
import resources.dataset_objects as do

def run_pipeline():
    # Extract - Transform - Load the datasets into a CervAI Dataset
    #datasets = [do.CRICDataset(), do.MENDDataset(), do.CDetectorDataset(), do.ASCYDataset(), do.IARCDataset()]
    #datasets = [do.CRICDataset(), do.MENDDataset(), do.CDetectorDataset(), do.ASCYDataset()] # Skip IARC for now, the images are too small
    #datasets = [do.IARCDataset()]

    datasets = [do.CRICDataset(), do.MENDDataset(), do.CDetectorDataset()]  # Skip ASCY and IARC so that one or both can be used for testing
    pipe = utils.CervPhase1Pipeline(datasets)
    pipe.run()


if __name__ == '__main__':
    run_pipeline()

