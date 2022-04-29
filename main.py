import random
import argparse
import logging
import os
import csv
import time

import numpy as np
import torch
from rdkit import RDLogger
from train import run_training

def setup(seed):
    # frozen random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_path', type=str,help='Path to train CSV file.')
    parser.add_argument('--val_path', type=str,help='Path to train CSV file.')
    parser.add_argument('--test_path', type=str,help='Path to test CSV file.')
    
    parser.add_argument('--checkpoints', type=str,help='checkpoints')
    parser.add_argument('--alpha', type=float, default=0.5, help='focal loss alpha')
    parser.add_argument('--gamma', type=int, default=2, help='focal loss gamma')

    parser.add_argument('--loss', type=str,help='bce or focalloss')
    parser.add_argument('--save_dir', type=str, help='Directory where model checkpoints will be saved')
    parser.add_argument('--class_weights', type=float, default=0.5, help='3cl class weight')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs to task')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
    parser.add_argument('--early_stop_epoch', type=int, default=30, help='If val loss did not drop in '
                                                                           'this epochs, stop running')
    parser.add_argument('--gpu', type=int, default=0,
                        choices=list(range(torch.cuda.device_count())),
                        help='Which GPU to use')

    args = parser.parse_args()

    return args

def create_logger(name, save_dir, quiet=False):
    """
    Creates a logger with a stream handler and two file handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Set logger depending on desired verbosity
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fh_v = logging.FileHandler(os.path.join(save_dir, 'verbose.log'))
        fh_v.setLevel(logging.DEBUG)

        logger.addHandler(fh_v)

    return logger


if __name__ == '__main__':
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)

    args = parse_args()
    setup(seed=args.seed)
    logger = create_logger(name='train', save_dir=args.save_dir, quiet=False)

    with open(args.train_path) as f:
        header = next(csv.reader(f))
    args.task_names = header[1:]
    args.num_tasks = len(args.task_names)

    run_training(args, logger)
