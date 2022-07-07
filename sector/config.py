import argparse

MODEL_TYPES = ['vanilla', 'la', 'mha', 'ffd']


def get_args():
    parser = argparse.ArgumentParser(description='Key-Memory Stock Prediction',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Path
    gp = parser.add_argument_group('Path', 'Data Paths')
    gp.add_argument('-i', '--input', type=str, default='./data/sector_input_rearranged.pkl',
                    help='path to input data file.')
    gp.add_argument('-o', '--output', type=str, default='./tmp/sector_output',
                    help='path to output data file.')
    gp.add_argument('-m', '--model', type=str, default='./tmp/sector_model',
                    help='path to save model parameters.')

    # Data
    gd = parser.add_argument_group('Data', 'How to Split Data')
    gd.add_argument('-tc', '--tgt_class', type=int, choices=[0,1,2],
                    help='target class, should be 0, 1 or 2')
    gd.add_argument('--mem_ratio', type=float, default=0.5,
                    help='ratio to split train set and validation set.')
    gd.add_argument('--val_ratio', type=float, default=0.1,
                    help='ratio to split train set and validation set.')
    gd.add_argument('--smooth_factor', type=float, default=0.1,
                    help='factor to smooth one hot vector of value.')
    gd.add_argument('--independent_mem', action='store_true',
                    help='whether memory data is separated from training data')

    # Model
    gm = parser.add_argument_group('Model', 'Model Parameters')
    gm.add_argument('-mod', '--mod', choices=MODEL_TYPES, default='vanilla',
                    help='model type')
    gm.add_argument('-hs', '--hidden_size', type=int, default=256,
                    help='The number of features in the hidden state h.')
    gm.add_argument('--n_layers', type=int, default=1,
                    help='Number of encoder layers.')

    # Training
    gt = parser.add_argument_group('Training', 'Training Parameters')
    gt.add_argument('-b', '--batch_size', type=int, default=200,
                    help='how many samples per batch to train.')
    gt.add_argument('-lr', '--learning_rate', type=float, default=1e-4,
                    help='learning rate.')
    gt.add_argument('-cuda', '--cuda', type=int, default=1,
                    help='cuda device number.')

    args = parser.parse_args()
    print('model parameters'.upper())
    for k, v in vars(args).items():
        print(k.ljust(20), '->', v)
    print('-' * 51)
    return args
