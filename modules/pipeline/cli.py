import os

import resources.utils as utils

from augmentation import rotate
from transformers import grayscale
from transformers import resize

def prepare_pipeline():
    datasets = [
        utils.CRICDataset(),
        utils.MENDDataset()
    ]
    
    resizer = utils.CervTransformer("Resize", resize.transform_function, probability=100)
    resizer.priority = -1
    resizer.enabledOverride = True

    transformers = [
        utils.CervAugmentor("Rotate", rotate.transform_function, probability=80), 
        utils.CervTransformer("GrayScale", grayscale.transform_function, probability=100),
        resizer
    ]

    return utils.CervProcessingPipeline(datasets, transformers)


def main():
    pipe = prepare_pipeline()

    header = """
========================================
CERVAI DATASET CREATION TOOL

Functional Commands:
etl - Perform ETL
toggle - Toggle a transformer

Display Commands:
stran - Show Transformers and Augmentors
cnt - Show Class Counts in CervAI Dataset

CLI Commands:
quit - Quit CLI
clear - Clear CLI

========================================
    """

    print(header)

    while True:
        cmd = input("> ")

        if cmd == "etl":
            pipe.check_cervAI_folder_exists()
            pipe.perform_etl()
            print('CervAI dataset folder created successfully')
        elif cmd == "toggle":
            name = input("What is the name of the transformer to enable/disable? ")
            pipe.override_transformer(name)
        elif cmd == "stran":
            pipe.show_transformers_and_augmentors()
        elif cmd == "cnt":
            pipe.get_class_counts()
        elif cmd == "quit":
            quit()
        elif cmd == 'clear':
            os.system('cls' if os.name == 'nt' else 'clear')
            print(header)
        else:
            print("Invalid Command")

if __name__ == '__main__':
    main()
