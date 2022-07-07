import argparse


def get_args():
    argp = argparse.ArgumentParser(description='Top 3 Industry Prediction',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Path
    argp.add_argument('-i', '--input', type=str, default='./data/industry_input_onehot.pkl',
                      help='path to input data file')
    argp.add_argument('-o', '--output', type=str, default='./tmp/industry_output_onehot.pkl',
                      help='path to output data file')
    argp.add_argument('-ti', '--test_input', type=str, default='./data/test_100/industry_input_test.pkl',
                      help='path to test input data file')
    argp.add_argument('-to', '--test_output', type=str, default='./tmp/industry_output_test.pkl',
                      help='path to test output data file')

    # Data
    argp.add_argument('-b', '--batch_size', type=int, default=3,
                      help='how many samples per batch to load.')
    argp.add_argument('-s', '--seq_len', type=int, default=100,
                      help='length of time series in each sample.')
    argp.add_argument('-m', '--sgm', type=int, default=10,
                      help='number of segments divided in the whole dataset.')
    argp.add_argument('--train_ratio', type=float, default=0.9,
                      help='ratio to split train set and validation set.')

    # Model

    argp.add_argument('-hs', '--hidden_size', type=int, default=128,
                      help='The number of features in the hidden state h.')

    # Training
    argp.add_argument('-n', '--num_epochs', type=int, default=1,
                      help='number of epochs.')
    argp.add_argument('--learning_rate', type=float, default=1e-3,
                      help='learning rate.')
    argp.add_argument('-cuda', '--cuda', type=int, default=0,
                      help='CUDA device.')

    args = argp.parse_args()
    return args
