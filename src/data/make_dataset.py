# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import torch
import numpy as np
PATH = 'data\\raw\\'
PROCESSED_PATH = 'data\\processed\\'


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    images_train = []
    labels_train = []
    for i in range(5):
        data = np.load(PATH + f'train_{i}.npz')
        images_train.append(data['images'])
        labels_train.append(data['labels'])
    images_train = torch.Tensor(np.concatenate(images_train)).unsqueeze(1)
    labels_train = torch.Tensor(np.concatenate(labels_train)).long()
    
    data = np.load(PATH + f'test.npz')
    images_test = torch.Tensor(data['images']).unsqueeze(1)
    labels_test = torch.Tensor(data['labels']).long()
    
    torch.save(images_train, PROCESSED_PATH + 'images_train.pt')
    torch.save(images_test, PROCESSED_PATH +'images_test.pt')
    torch.save(labels_train, PROCESSED_PATH + 'labels_train.pt')
    torch.save(labels_test, PROCESSED_PATH + 'labels_test.pt')
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
