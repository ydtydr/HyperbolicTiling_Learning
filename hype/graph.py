#!/usr/bin/env python3
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict as ddict
import pandas
import numpy as np
from numpy.random import choice
import torch as th
from torch import nn
from torch.utils.data import Dataset as DS
from sklearn.metrics import average_precision_score
from multiprocessing.pool import ThreadPool
from functools import partial
import h5py
from tqdm import tqdm

from hype.lorentz import LorentzManifold

def load_adjacency_matrix(path, format='hdf5', symmetrize=False):
    if format == 'hdf5':
        with h5py.File(path, 'r') as hf:
            return {
                'ids': hf['ids'].value.astype('int'),
                'neighbors': hf['neighbors'].value.astype('int'),
                'offsets': hf['offsets'].value.astype('int'),
                'weights': hf['weights'].value.astype('float'),
                'objects': hf['objects'].value
            }
    elif format == 'csv':
        df = pandas.read_csv(path, usecols=['id1', 'id2', 'weight'], engine='c')

        if symmetrize:
            rev = df.copy().rename(columns={'id1' : 'id2', 'id2' : 'id1'})
            df = pandas.concat([df, rev])

        idmap = {}
        idlist = []

        def convert(id):
            if id not in idmap:
                idmap[id] = len(idlist)
                idlist.append(id)
            return idmap[id]
        df.loc[:, 'id1'] = df['id1'].apply(convert)
        df.loc[:, 'id2'] = df['id2'].apply(convert)

        groups = df.groupby('id1').apply(lambda x: x.sort_values(by='id2'))
        counts = df.groupby('id1').id2.size()

        ids = groups.index.levels[0].values
        offsets = counts.loc[ids].values
        offsets[1:] = np.cumsum(offsets)[:-1]
        offsets[0] = 0
        neighbors = groups['id2'].values
        weights = groups['weight'].values
        return {
            'ids' : ids.astype('int'),
            'offsets' : offsets.astype('int'),
            'neighbors': neighbors.astype('int'),
            'weights': weights.astype('float'),
            'objects': np.array(idlist)
        }
    else:
        raise RuntimeError(f'Unsupported file format {format}')


def load_edge_list(path, symmetrize=False):
    '''
    Used in embed to read data from a file (e.g. .csv)
    A dataset is a graph (edges and vertices)

    Args:
    idx (ndarray[number of edges,2]): storing pointers to objects
    objects.tolist() (list[number of vertices]): storing names of vertices
    weights (ndarra[number of edges]): weights on edges
    '''
    df = pandas.read_csv(path, usecols=['id1', 'id2', 'weight'], engine='c')
    df.dropna(inplace=True)
    if symmetrize:
        rev = df.copy().rename(columns={'id1' : 'id2', 'id2' : 'id1'})
        df = pandas.concat([df, rev])
    idx, objects = pandas.factorize(df[['id1', 'id2']].values.reshape(-1))
    idx = idx.reshape(-1, 2).astype('int')
    weights = df.weight.values.astype('float')
    return idx, objects.tolist(), weights


class Embedding(nn.Module):
    def __init__(self, size, dim, manifold, device, faraway, polytope, optimalisation = 'rsgd', sparse=True):
        super(Embedding, self).__init__()
        self.dim = dim
        self.nobjects = size
        self.manifold = manifold
        self.device = device
        self.optimalisation = optimalisation
        self.faraway = faraway
        self.polytope = polytope
        self.lt = nn.Embedding(size, dim, sparse=sparse)
        ############ add this line to store integer matrix
        if 'group' in str(manifold) and 'high' not in str(manifold):
            self.int_matrix = th.zeros(size, 3, 3, device=self.device)
        elif 'hyperbolicspace' in str(manifold) or 'discrete' in optimalisation:
            self.int_matrix = th.zeros(size, self.dim, self.dim, device=self.device) 
            self.init_scalarproduct_hyperbolicspace()
        elif 'group' in str(manifold) and 'high' in str(manifold):
            self.int_matrix = th.Tensor(size, dim//3, 3, 3)
        elif 'bugaenko6' in str(manifold):
            self.int_matrix = th.zeros(size, 2, 7, 7, device=self.device) 
            self.init_scalarproduct_bugaenko6()
        '''
        elif 'vinberg17' in str(manifold):
            self.int_matrix = th.zeros(size, 18, 18, device=self.device) 
            self.init_scalarproduct_vinberg17()
        elif 'vinberg3' in str(manifold):
            self.int_matrix = th.zeros(size, 4, 4, device=self.device) 
            self.init_scalarproduct_vinberg3()
        '''
        ############        
        #if 'discrete' in optimalisation:
        #    self.dist = LorentzManifold().distance
        #else:
        self.dist = manifold.distance

        self.pre_hook = None
        self.post_hook = None
        self.init_weights(manifold)

    def init_scalarproduct_bugaenko6(self):
        self.g = th.zeros(2,7,7, device=self.device)
        self.g[0] = th.eye(7,7, device=self.device)
        self.g[0,0,0] = -1
        self.g[1,0,0] = -1

    '''def init_scalarproduct_vinberg17(self):
        self.g = th.eye(18,18, device=self.device)
        self.g[0,0] = -1

    def init_scalarproduct_vinberg3(self):
        self.g = th.eye(4,4, device=self.device)
        self.g[0,0] = -1
    '''

    def init_scalarproduct_hyperbolicspace(self):
        self.g = th.eye(self.dim,self.dim, device=self.device)
        self.g[0,0] = -1

    def init_weights(self, manifold, scale=1e-4): #scale=1e-4
        if 'halfspace' in str(self.manifold):
            manifold.init_weights(self.lt.weight, scale, self.faraway)
        else:
            manifold.init_weights(self.lt.weight, scale)
        #if 'group' in str(self.manifold) or 'bugaenko6' in str(self.manifold) or 'vinberg17' in str(self.manifold) or 'vinberg3' in str(self.manifold) or 'hyperbolicspace' in str(self.manifold):
        if 'group' in str(self.manifold) or 'hyperbolicspace' in str(self.manifold) or 'discrete' in self.optimalisation or 'bugaenko6' in str(self.manifold):
            self.int_matrix.zero_()
            manifold.init_weights_int_matrix(self.int_matrix,self.faraway,self.dim,self.polytope)

    def forward(self, inputs):
        e = self.lt(inputs)
        with th.no_grad():
            e = self.manifold.normalize(e)
        if self.pre_hook is not None:
            e = self.pre_hook(e)
        #if 'group' in str(self.manifold) or 'bugaenko6' in str(self.manifold) or 'vinberg17' in str(self.manifold) or 'vinberg3' in str(self.manifold) or 'hyperbolicspace' in str(self.manifold):
        if 'group' in str(self.manifold) or 'hyperbolicspace' in str(self.manifold) or 'bugaenko6' in str(self.manifold):
            int_matrix = self.int_matrix[inputs]
            fval = self._forward(e, int_matrix)
        else:
            fval = self._forward(e)
        return fval

    def embedding(self):
        return list(self.lt.parameters())[0].data.cpu().numpy()

    def optim_params(self, manifold):
        return [{
            'params': self.lt.parameters(),
            'rgrad': manifold.rgrad,
            'expm': manifold.expm,
            'logm': manifold.logm,
            'ptransp': manifold.ptransp,
        }, ]


# This class is now deprecated in favor of BatchedDataset (graph_dataset.pyx)
class Dataset(DS):
    _neg_multiplier = 1
    _ntries = 10
    _sample_dampening = 0.75

    def __init__(self, idx, objects, weights, nnegs, unigram_size=1e8):
        assert idx.ndim == 2 and idx.shape[1] == 2
        assert weights.ndim == 1
        assert len(idx) == len(weights)
        assert nnegs >= 0
        assert unigram_size >= 0

        print('Indexing data')
        self.idx = idx
        self.nnegs = nnegs
        self.burnin = False
        self.objects = objects

        self._weights = ddict(lambda: ddict(int))
        self._counts = np.ones(len(objects), dtype=np.float)
        self.max_tries = self.nnegs * self._ntries
        for i in range(idx.shape[0]):
            t, h = self.idx[i]
            self._counts[h] += weights[i]
            self._weights[t][h] += weights[i]
        self._weights = dict(self._weights)
        nents = int(np.array(list(self._weights.keys())).max())
        assert len(objects) > nents, f'Number of objects do no match'

        if unigram_size > 0:
            c = self._counts ** self._sample_dampening
            self.unigram_table = choice(
                len(objects),
                size=int(unigram_size),
                p=(c / c.sum())
            )

    def __len__(self):
        return self.idx.shape[0]

    def weights(self, inputs, targets):
        return self.fweights(self, inputs, targets)

    def nnegatives(self):
        if self.burnin:
            return self._neg_multiplier * self.nnegs
        else:
            return self.nnegs

    @classmethod
    def collate(cls, batch):
        inputs, targets = zip(*batch)
        return th.cat(inputs, 0), th.cat(targets, 0)


# This function is now deprecated in favor of eval_reconstruction
def eval_reconstruction_slow(adj, lt, lt_int_matrix, distfn):
    ranks = []
    ap_scores = []

    for s, s_types in adj.items():
        s_e = lt[s].expand_as(lt)
        s_e_int_matrix = lt_int_matrix[s].expand_as(lt_int_matrix)
        _dists = distfn(s_e, s_e_int_matrix, lt, lt_int_matrix).data.cpu().numpy().flatten()
        _dists[s] = 1e+12
        _labels = np.zeros(lt.size(0))
        _dists_masked = _dists.copy()
        _ranks = []
        for o in s_types:
            _dists_masked[o] = np.Inf
            _labels[o] = 1
        for o in s_types:
            d = _dists_masked.copy()
            d[o] = _dists[o]
            r = np.argsort(d)
            _ranks.append(np.where(r == o)[0][0] + 1)
        ranks += _ranks
        ap_scores.append(
            average_precision_score(_labels, -_dists)
        )
    return np.mean(ranks), np.mean(ap_scores)


def reconstruction_worker(adj, lt, distfn, objects, progress=False, lt_int_matrix=None, g=None):
    ranksum = nranks = ap_scores = iters = 0
    labels = np.empty(lt.size(0))
    for object in tqdm(objects) if progress else objects:
        labels.fill(0)
        neighbors = np.array(list(adj[object]))
        if 'group' in str(distfn):
            dists = distfn(lt[None, object], lt_int_matrix[None, object], lt, lt_int_matrix)
        #elif 'bugaenko6' in str(distfn) or 'vinberg17' in str(distfn) or 'vinberg3' in str(distfn) or 'hyperbolicspace' in str(distfn):
        elif 'hyperbolicspace' in str(distfn) or 'bugaenko6' in str(distfn):
            dists = distfn(lt[None, object], lt_int_matrix[None, object], lt, lt_int_matrix, g)
        else:
            dists = distfn(lt[None, object], lt)
        dists[object] = 1e12
        sorted_dists, sorted_idx = dists.sort()
        ranks, = np.where(np.in1d(sorted_idx.detach().cpu().numpy(), neighbors))
        # The above gives us the position of the neighbors in sorted order.  We
        # want to count the number of non-neighbors that occur before each neighbor
        ranks += 1
        N = ranks.shape[0]

        # To account for other positive nearer neighbors, we subtract (N*(N+1)/2)
        # As an example, assume the ranks of the neighbors are:
        # 0, 1, 4, 5, 6, 8
        # For each neighbor, we'd like to return the number of non-neighbors
        # that ranked higher than it.  In this case, we'd return 0+0+2+2+2+3=14
        # Another way of thinking about it is to return
        # 0 + 1 + 4 + 5 + 6 + 8 - (0 + 1 + 2 + 3 + 4 + 5)
        # (0 + 1 + 2 + ... + N) == (N * (N + 1) / 2)
        # Note that we include `N` to account for the source embedding itself
        # always being the nearest neighbor
        ranksum += ranks.sum() - (N * (N - 1) / 2)
        nranks += ranks.shape[0]
        labels[neighbors] = 1
        # print(dists.detach().cpu().numpy().max())
        # assert 1==2
        # distss = th.clamp(dists,max=1e12)
        # print(object,dists)
        # print(dists !=dists)
        # print(lt[object])
        # print(lt[0])
        # assert 1 == 2
        ap_scores += average_precision_score(labels, -dists.detach().cpu().numpy())
        iters += 1
    return float(ranksum), nranks, ap_scores, iters


def eval_reconstruction(adj, lt, distfn, g=None, workers=1, progress=False, lt_int_matrix=None):
    '''
    Reconstruction evaluation.  For each object, rank its neighbors by distance
    Args:
        adj (dict[int, set[int]]): Adjacency list mapping objects to its neighbors
        lt (torch.Tensor[N, dim]): Embedding table with `N` embeddings and `dim`
            dimensionality
        distfn ((torch.Tensor, torch.Tensor) -> torch.Tensor): distance function.
        workers (int): number of workers to use
    '''
    objects = np.array(list(adj.keys()))
    if workers > 1:
        with ThreadPool(workers) as pool:
            #if 'group' in str(distfn) or 'bugaenko6' in str(distfn) or 'vinberg17' in str(distfn) or 'vinberg3' in str(distfn) or 'hyperbolicspace' in str(distfn):
            if 'group' in str(distfn) or 'hyperbolicspace' in str(distfn) or 'bugaenko6' in str(distfn):
                f = partial(reconstruction_worker, adj, lt, distfn, lt_int_matrix=lt_int_matrix)
            else:
                f = partial(reconstruction_worker, adj, lt, distfn)
            results = pool.map(f, np.array_split(objects, workers))
            results = np.array(results).sum(axis=0).astype(float)
    else:
        if 'group' in str(distfn):
            results = reconstruction_worker(adj, lt, distfn, objects, progress, lt_int_matrix=lt_int_matrix)
        #elif 'bugaenko6' in str(distfn) or 'vinberg17' in str(distfn) or 'vinberg3' in str(distfn) or 'hyperbolicspace' in str(distfn):
        elif 'hyperbolicspace' in str(distfn) or 'bugaenko6' in str(distfn):
            results = reconstruction_worker(adj, lt, distfn, objects, progress, lt_int_matrix=lt_int_matrix, g = g)
        else:
            results = reconstruction_worker(adj, lt, distfn, objects, progress)
    return float(results[0]) / results[1], float(results[2]) / results[3]   