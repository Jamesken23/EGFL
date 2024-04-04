import argparse


def parameter_parser():
    # Experiment parameters
    parser = argparse.ArgumentParser(description='A-Multimodal-Fusion-Learning-Framework-for-Smart-Contract-Vulnerability-Detection')
    
    # ----------------------------------------------------------------------------------------------------
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--EMBEDDING_LAYER_DIM', type=int, default=256, help='dimensions of vector')
    parser.add_argument('--CNN_LAYER_DIM', type=int, default=200, help='dimensions of vector')
    
    parser.add_argument('--lr', type=float, default=0.002, help='learning rate')
    parser.add_argument('-d', '--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--test_split_size', type=float, default=0.2, help='the split size of test dataset')
    # Found 14663 unique tokens.
    # Found 34132 unique tokens.
    parser.add_argument('--MAX_WORDS', type=int, default=1000, help='Maximum number of vocabulary lists')
    parser.add_argument('--SEQ_LEN', type=int, default=8000, help='Maximum length of a sentence')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('-th', '--threshold', type=float, default=0.5, help='threshold')
    parser.add_argument('--heads', type=int, default=10, help='numbers of attention head')

    return parser.parse_args()