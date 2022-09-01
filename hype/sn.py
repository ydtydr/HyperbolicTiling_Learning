#!/usr/bin/env python3
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch as th
import numpy as np
from torch import nn
from numpy.random import randint
from . import graph
from .graph_dataset import BatchedDataset

model_name = '%s_dim%d'


class Embedding(graph.Embedding):
    def __init__(self, size, dim, manifold, device, faraway, polytope, optimalisation='rsgd', sparse=True):
        super(Embedding, self).__init__(size, dim, manifold, device, faraway, polytope, optimalisation, sparse)
        self.lossfn = nn.functional.cross_entropy
        self.manifold = manifold
        self.optimalisation = optimalisation

    def _forward(self, e, int_matrix=None, int_norm=None):
        o = e.narrow(1, 1, e.size(1) - 1)
        s = e.narrow(1, 0, 1).expand_as(o)###source
        if 'group' in str(self.manifold):
            o_int_matrix = int_matrix.narrow(1, 1, e.size(1) - 1)
            s_int_matrix = int_matrix.narrow(1, 0, 1).expand_as(o_int_matrix)###source
            dists = self.dist(s, s_int_matrix, o, o_int_matrix).squeeze(-1)
        elif 'hyperbolicspace' in str(self.manifold) or 'bugaenko6' in str(self.manifold):
            o_int_matrix = int_matrix.narrow(1, 1, e.size(1) - 1)
            s_int_matrix = int_matrix.narrow(1, 0, 1).expand_as(o_int_matrix)###source
            dists = self.dist(s, s_int_matrix, o, o_int_matrix, self.g).squeeze(-1)
        else:
            dists = self.dist(s, o).squeeze(-1)
        return -dists
    
    def loss(self, preds, targets, weight=None, size_average=True):
        
        case = 1
        
        if case == 1:
            return self.lossfn(preds, targets)
        if case == 2:
            t = 10 #temperature
            return self.lossfn(-1*t*preds*preds, targets)
        if case == 3:
            r = 2*np.log(self.nobjects)-1
            t = 0.1
            t0 = (-preds-r)/t + 1
            t1 = th.exp(1/t0)
            t2 = th.sum(t1,-1,keepdim=False)
            t3 = th.div(t1.narrow(-1,0,1).squeeze(),t2)
            return -th.sum(th.log(t3))/preds.size(0)


# This class is now deprecated in favor of BatchedDataset (graph_dataset.pyx)
class Dataset(graph.Dataset):
    def __getitem__(self, i):
        t, h = self.idx[i]
        negs = set()
        ntries = 0
        nnegs = int(self.nnegatives())
        if t not in self._weights:
            negs.add(t)
#             print(negs)
        else:
            while ntries < self.max_tries and len(negs) < nnegs:
                if self.burnin:
                    n = randint(0, len(self.unigram_table))
                    n = int(self.unigram_table[n])
                else:
                    n = randint(0, len(self.objects))
                if (n not in self._weights[t]) or \
                        (self._weights[t][n] < self._weights[t][h]):
                    negs.add(n)
                ntries += 1
        if len(negs) == 0:
            negs.add(t)
        ix = [t, h] + list(negs)
        while len(ix) < nnegs + 2:
            ix.append(ix[randint(2, len(ix))])
#         print(ix)
#         assert 1==2
        return th.LongTensor(ix).view(1, len(ix)), th.zeros(1).long()


def initialize(manifold, opt, idx, objects, weights, device, sparse=True):
    conf = []
    mname = model_name % (opt.manifold, opt.dim)
    data = BatchedDataset(idx, objects, weights, opt.negs, opt.batchsize,
        opt.ndproc, opt.burnin > 0, opt.dampening)
    model = Embedding(
        len(data.objects),
        opt.dim,
        manifold,
        device,
        opt.faraway,
        opt.polytope,
        opt.optimalisation,
        sparse=sparse
    )
    data.objects = objects
    return model, data, mname, conf
