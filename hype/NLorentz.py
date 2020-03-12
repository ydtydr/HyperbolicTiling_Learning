#!/usr/bin/env python3
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch as th
from torch.autograd import Function
from .common import acosh
from .manifold import Manifold


class NLorentzManifold(Manifold):
    __slots__ = ["eps", "_eps", "norm_clip", "max_norm", "debug"]

    @staticmethod
    def dim(dim):
        return dim*3

    def __init__(self, eps=1e-12, _eps=1e-5, norm_clip=1, max_norm=1e6,
            debug=False, **kwargs):
        self.eps = eps
        self._eps = _eps
        self.norm_clip = norm_clip
        self.max_norm = max_norm
        self.debug = debug

    @staticmethod
    def ldot(u, v, keepdim=False):
        """Lorentzian Scalar Product"""
        uv = u * v
        uv.narrow(-1, 0, 1).mul_(-1)
        return th.sum(uv, dim=-1, keepdim=keepdim)

    def to_poincare_ball(self, u):
        x = u.clone()
        d = x.size(-1) - 1
        return x.narrow(-1, 1, d) / (x.narrow(-1, 0, 1) + 1)

    def distance(self, u, v):
        dimension = u.size(-1)//3
        for i in range(dimension):
            d = -LorentzDot.apply(u[...,3*i:3*(i+1)], v[...,3*i:3*(i+1)])
            d.data.clamp_(min=1)
            if i==0:
                d_all = acosh(d, self._eps)
            else:
                d_all += acosh(d, self._eps)
        return d_all

    def pnorm(self, u):
        dimension = u.size(-1)//3
        for i in range(dimension):
            if i==0:
                all_norm = th.sqrt(th.sum(th.pow(self.to_poincare_ball(u[...,3*i:3*(i+1)]), 2), dim=-1))
            else:
                all_norm += th.sqrt(th.sum(th.pow(self.to_poincare_ball(u[...,3*i:3*(i+1)]), 2), dim=-1))
        return all_norm/dimension

    def normalize(self, ww, gra=True):
        """Normalize vector such that it is located on the hyperboloid"""
        if gra:
            dimension = ww.size(-1)//3
            for i in range(dimension):
                w = ww[...,3*i:3*(i+1)]
                d = w.size(-1) - 1
                narrowed = w.narrow(-1, 1, d)
                if self.max_norm:
                    narrowed.view(-1, d).renorm_(p=2, dim=0, maxnorm=self.max_norm)
                tmp = 1 + th.sum(th.pow(narrowed, 2), dim=-1, keepdim=True)
                tmp.sqrt_()
                w.narrow(-1, 0, 1).copy_(tmp)
        else:
            w = ww
            d = w.size(-1) - 1
            narrowed = w.narrow(-1, 1, d)
            if self.max_norm:
                narrowed.view(-1, d).renorm_(p=2, dim=0, maxnorm=self.max_norm)
            tmp = 1 + th.sum(th.pow(narrowed, 2), dim=-1, keepdim=True)
            tmp.sqrt_()
            w.narrow(-1, 0, 1).copy_(tmp)
        return ww

    def normalize_tan(self, x_all, v_all):
        d = v_all.size(1) - 1
        x = x_all.narrow(1, 1, d)
        xv = th.sum(x * v_all.narrow(1, 1, d), dim=1, keepdim=True)
        tmp = 1 + th.sum(th.pow(x_all.narrow(1, 1, d), 2), dim=1, keepdim=True)
        tmp.sqrt_().clamp_(min=self._eps)
        v_all.narrow(1, 0, 1).copy_(xv / tmp)
        return v_all

    def init_weights(self, w, irange=1e-5):
        w.data.uniform_(-irange, irange)
        w.data.copy_(self.normalize(w.data))

    def rgrad(self, p, d_p):
        """Riemannian gradient for hyperboloid"""
        if d_p.is_sparse:
            uu = d_p._values()
            xx = p.index_select(0, d_p._indices().squeeze())
        else:
            uu = d_p
            xx = p
        dimension = p.size(-1)//3
        for i in range(dimension):
            u = uu[...,3*i:3*(i+1)]
            x = xx[...,3*i:3*(i+1)]
            u.narrow(-1, 0, 1).mul_(-1)
            u.addcmul_(self.ldot(x, u, keepdim=True).expand_as(x), x)
        return d_p

    def expm(self, pp, d_pp, lr=None, out=None, normalize=False):
        dimension = pp.size(-1)//3
        ix, d_val_p = d_pp._indices().squeeze(), d_pp._values()
        p_val_p = self.normalize(pp.index_select(0, ix))
        for i in range(dimension):
            p = pp[...,3*i:3*(i+1)]
            d_val = d_val_p[...,3*i:3*(i+1)]
            p_val = p_val_p[...,3*i:3*(i+1)]
            ldv = self.ldot(d_val, d_val, keepdim=True)
            if self.debug:
                assert all(ldv > 0), "Tangent norm must be greater 0"
                assert all(ldv == ldv), "Tangent norm includes NaNs"
            nd_p = ldv.clamp_(min=0).sqrt_()
            t = th.clamp(nd_p, max=self.norm_clip)
            nd_p.clamp_(min=self.eps)
            newp = (th.cosh(t) * p_val).addcdiv_(th.sinh(t) * d_val, nd_p)
            if normalize:
                newp = self.normalize(newp,gra=False)
            p.index_copy_(0, ix, newp)

    def logm(self, x, y):
        """Logarithmic map on the Lorenz Manifold"""
        xy = th.clamp(self.ldot(x, y).unsqueeze(-1), max=-1)
        v = acosh(-xy, self.eps).div_(
            th.clamp(th.sqrt(xy * xy - 1), min=self._eps)
        ) * th.addcmul(y, xy, x)
        return self.normalize_tan(x, v)

    def ptransp(self, x, y, v, ix=None, out=None):
        """Parallel transport for hyperboloid"""
        if ix is not None:
            v_ = v
            x_ = x.index_select(0, ix)
            y_ = y.index_select(0, ix)
        elif v.is_sparse:
            ix, v_ = v._indices().squeeze(), v._values()
            x_ = x.index_select(0, ix)
            y_ = y.index_select(0, ix)
        else:
            raise NotImplementedError
        xy = self.ldot(x_, y_, keepdim=True).expand_as(x_)
        vy = self.ldot(v_, y_, keepdim=True).expand_as(x_)
        vnew = v_ + vy / (1 - xy) * (x_ + y_)
        if out is None:
            return vnew
        else:
            out.index_copy_(0, ix, vnew)


class LorentzDot(Function):
    @staticmethod
    def forward(ctx, u, v):
        ctx.save_for_backward(u, v)
        return NLorentzManifold.ldot(u, v)

    @staticmethod
    def backward(ctx, g):
        u, v = ctx.saved_tensors
        g = g.unsqueeze(-1).expand_as(u).clone()
        g.narrow(-1, 0, 1).mul_(-1)
        return g * v, g * u
