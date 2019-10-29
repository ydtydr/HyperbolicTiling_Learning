#!/usr/bin/env python3
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from hype.graph import eval_reconstruction, load_adjacency_matrix
import argparse
import numpy as np
import torch
import os
from hype.lorentz import LorentzManifold
from hype.lorentz2 import Lorentz2Manifold
from hype.lorentz_high import LorentzHighManifold
from hype.group_rie import GroupRieManifold
from hype.group_high_rie import GroupHighRieManifold
from hype.group_euc import GroupEucManifold
from hype.halfspace_euc import HalfspaceEucManifold
from hype.halfspace_rie import HalfspaceRieManifold
from hype.euclidean import EuclideanManifold, TranseManifold
from hype.poincare import PoincareManifold
import timeit


MANIFOLDS = {
    'lorentz': LorentzManifold,
    'lorentz2': Lorentz2Manifold,
    'lorentz_high': LorentzHighManifold,
    'group_rie': GroupRieManifold,
    'group_high_rie': GroupHighRieManifold,
    'group_euc': GroupEucManifold,
    'halfspace_euc': HalfspaceEucManifold,
    'halfspace_rie': HalfspaceRieManifold,
    'euclidean': EuclideanManifold,
    'transe': TranseManifold,
    'poincare': PoincareManifold
}

np.random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('file', help='Path to checkpoint')
parser.add_argument('-workers', default=1, type=int, help='Number of workers')
parser.add_argument('-sample', type=int, help='Sample size')
parser.add_argument('-quiet', action='store_true', default=False)
args = parser.parse_args()

# set default tensor type
torch.set_default_tensor_type('torch.DoubleTensor')


chkpnt = torch.load(args.file)
dset = chkpnt['conf']['dset']
if not os.path.exists(dset):
    raise ValueError("Can't find dset!")

format = 'hdf5' if dset.endswith('.h5') else 'csv'
dset = load_adjacency_matrix(dset, 'hdf5')

sample_size = args.sample or len(dset['ids'])
sample = np.random.choice(len(dset['ids']), size=sample_size, replace=False)

adj = {}

for i in sample:
    end = dset['offsets'][i + 1] if i + 1 < len(dset['offsets']) \
        else len(dset['neighbors'])
    adj[i] = set(dset['neighbors'][dset['offsets'][i]:end])

manifold = MANIFOLDS[chkpnt['conf']['manifold']]()

lt = chkpnt['embeddings']
if not isinstance(lt, torch.Tensor):
    lt = torch.from_numpy(lt).cuda()

tstart = timeit.default_timer()
meanrank, maprank = eval_reconstruction(adj, lt, manifold.distance,
    workers=args.workers, progress=not args.quiet)
etime = timeit.default_timer() - tstart

print(f'Mean rank: {meanrank}, mAP rank: {maprank}, time: {etime}')
