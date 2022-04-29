import random
import logging
import csv

import numpy as np

import torch
from torch.utils.data.dataset import Dataset

class MoleculeDatapoint:
    """A MoleculeDatapoint contains a single molecule and its associated features and targets."""

    def __init__(self,line):
        """
        Initializes a MoleculeDatapoint, which contains a single molecule.
        :param line: A list of strings generated by separating a line in a data CSV file by comma.
        """

        self.smiles = line[0]
        self.targets = [float(x) if x != '' else None for x in line[1:]]


class MoleculeDataset(Dataset):
    """A MoleculeDataset contains a list of molecules and their associated features and targets."""

    def __init__(self, data):
        """
        Initializes a MoleculeDataset, which contains a list of MoleculeDatapoints (i.e. a list of molecules).

        :param data: A list of MoleculeDatapoints.
        """
        self.data = data

    def smiles(self):
        """
        Returns the smiles strings associated with the molecules.

        :return: A list of smiles strings.
        """
        return [d.smiles for d in self.data]

    def targets(self):
        """
        Returns the targets associated with each molecule.

        :return: A list of lists of floats containing the targets.
        """
        return [d.targets for d in self.data]

    def num_tasks(self):
        """
        Returns the number of prediction tasks.

        :return: The number of tasks.
        """
        return self.data[0].num_tasks() if len(self.data) > 0 else None


    def shuffle(self, seed):
        """
        Shuffles the dataset.

        :param seed: Optional random seed.
        """
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.data)


    def __len__(self):
        """
        Returns the length of the dataset (i.e. the number of molecules).

        :return: The length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Gets one or more MoleculeDatapoints via an index or slice.

        :param item: An index (int) or a slice object.
        :return: A MoleculeDatapoint if an int is provided or a list of MoleculeDatapoints if a slice is provided.
        """
        return self.data[idx]


def load_data(args, logger):
    """
    load the training data.
    :param args:
    :param logger:
    :return:
    """
    # Get data
    logger.debug('Loading data')
    logger.debug(f'Splitting data with seed {args.seed}')

    data = get_data(args.train_path)
    if args.test_path:
        train_data = get_data(args.train_path)
        val_data = get_data(args.val_path)
        test_data = get_data(args.test_path)
    else:
        train_data, val_data, test_data = split_data(data=data, sizes=(0.7, 0.2, 0.1), seed=args.seed)

    logger.debug(f'train size = {len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}')

    return train_data, val_data, test_data


def get_data(path):
    """
    Gets smiles string and target values (and optionally compound names if provided) from a CSV file.

    :param path: Path to a CSV file.
    :return: A MoleculeDataset containing smiles strings and target values along
    with other info such as additional features and compound names when desired.
    """
    # Load data
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header

        lines = []
        for line in reader:
            lines.append(line)

        data = MoleculeDataset([
            MoleculeDatapoint(
                line=line
            ) for line in lines
        ])

    return data


def split_data(data, sizes,seed):
    """
    Splits data into training, validation, and test splits.
    :param data: A MoleculeDataset.
    :param sizes: A length-3 tuple with the proportions of data in the
    train, validation, and test sets.
    :param seed: The random seed to use before shuffling data.
    :return: A tuple containing the train, validation, and test splits of the data.
    """
    assert len(sizes) == 3 and sum(sizes) == 1

    data.shuffle(seed=seed)

    train_size = int(sizes[0] * len(data))
    train_val_size = int((sizes[0] + sizes[1]) * len(data))

    train = data[:train_size]
    val = data[train_size:train_val_size]
    test = data[train_val_size:]

    return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)


class MolCollator(object):
    """
    Collator for pytorch dataloader
    :param args: Arguments.
    """
    def __init__(self, args):
        self.args = args

    def __call__(self, batch):
        smiles_batch = [d.smiles for d in batch]
        target_batch = [d.targets for d in batch]
        mask = torch.Tensor([[x is not None for x in tb] for tb in target_batch])
        targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in target_batch])

        return smiles_batch, mask, targets