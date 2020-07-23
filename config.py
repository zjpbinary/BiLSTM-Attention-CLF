import argparse

parser = argparse.ArgumentParser(description="神经网络参数配置")
parser.add_argument("--gpu", default='cpu')
args = parser.parse_args()

class TextCNNConfig(object):
    trainfile = 'data/train'
    testfile = 'data/train'
    ninp = 512
    nhid = 512
    nlayers = 2
    dropout = 0.5
    lr = 0.00002
    bsz = 100
    epoch = 1000
    device = args.gpu
    max_len = 12
