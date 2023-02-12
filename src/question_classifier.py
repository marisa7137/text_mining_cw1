import torch
import numpy as np
import argparse
from torch import nn


def test(config):
    pass


def train(config):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argument parser for loading config, training, testing')
    parser.add_argument('--config', type=str, required=True, help='Configuration file')
    parser.add_argument('--train', action='store_true', help='Training mode - model is saved')
    parser.add_argument('--test', action='store_true', help='Testing mode - needs a model to load')