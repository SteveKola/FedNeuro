# spectral bias code
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from argparse import Namespace
from functools import reduce
from copy import deepcopy

#CRDT
from py3crdt.gset import GSet
# spectral bias
def make_phased_waves(opt):
    t = np.arange(0, 1, 1./opt.N)
    if opt.A is None:
        yt = reduce(lambda a, b: a + b, 
                    [np.sin(2 * np.pi * ki * t + 2 * np.pi * phi) for ki, phi in zip(opt.K, opt.PHI)])
    else:
        yt = reduce(lambda a, b: a + b, 
                    [Ai * np.sin(2 * np.pi * ki * t + 2 * np.pi * phi) for ki, Ai, phi in zip(opt.K, opt.A, opt.PHI)])
    return t, yt

def fft(opt, yt):
    n = len(yt) # length of the signal
    k = np.arange(n)
    T = n/opt.N
    frq = k/T # two sides frequency range
    frq = frq[range(n//2)] # one side frequency range
    # -------------
    FFTYT = np.fft.fft(yt)/n # fft computing and normalization
    FFTYT = FFTYT[range(n//2)]
    fftyt = abs(FFTYT)
    return frq, fftyt

def to_torch_dataset_1d(opt, t, yt):
    t = torch.from_numpy(t).view(-1, 1).float()
    yt = torch.from_numpy(yt).view(-1, 1).float()
    # if opt.CUDA:
    #     t = t.cuda()
    #     yt = yt.cuda()
    return t, yt

# the dataset should 1. yield the model 2. narrow the search space

# searching for models
# dissect https://github.com/ianwhale/nsga-net for bitstring encoding, model deduplication
# look at https://github.com/nightstorm0909/NEvoNAS for how to structure actual search space / cells + implemenation of novelty search

# CRDTs could help with encoding, deduplication, and novelty search
gset1 = GSet(id=1)
gset2 = GSet(id=2)
gset1.add('a')
gset1.add('b')
gset1.display()
# ['a', 'b']   ----- Output
gset2.add('b')
gset2.add('c')
gset2.display()
# ['b', 'c']   ----- Output
gset1.merge(gset2)   
gset1.display()
# ['a', 'b', 'c']   ----- Output