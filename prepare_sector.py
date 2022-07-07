import numpy as np
import pickle
import argparse
import ipdb
from collections import OrderedDict


def rearrange(content, seq_len):
    sectors = dict()
    dates = OrderedDict()
    result = []
    for obs, tgt, sec, dat in content:
        # build sector index
        if sec not in sectors:
            sectors[sec] = ([], OrderedDict())
        # build date index
        if dat not in dates:
            dates[dat] = []
        # append data with sector index
        sectors[sec][0].append((obs, tgt))
        sectors[sec][1][dat] = len(sectors[sec][0]) - 1
        if len(sectors[sec][0]) >= seq_len:
            # append sector name with date index
            dates[dat].append(sec)
    while len(dates) > 0:
        dat, sec_names = dates.popitem()
        for sec in sec_names:
            content = sectors[sec][0]
            seq_end = sectors[sec][1][dat]
            seq_start = seq_end - seq_len + 1
            seq = np.vstack([x[0] for x in content[seq_start: seq_end]])
            tgt = content[seq_end][1][0].tolist()
            result.append((seq, tgt, sec, dat))
    return result


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument('-i', '--input', type=str)
    argp.add_argument('-o', '--output', type=str)
    argp.add_argument('-s', '--seq_len', type=int)
    args = argp.parse_args()

    with open(args.input, 'rb') as f:
        content = pickle.load(f)

    result = rearrange(content, args.seq_len)
    result.reverse()

    print('saving data...')
    with open(args.output, 'wb') as f:
        pickle.dump(result, f)


if __name__ == '__main__':
    main()
