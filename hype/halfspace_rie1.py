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

def recon_halfplane(g):
    d = (g.size(-1)-1)//2
    j = g[...,-1]#n
    k = g[...,d:-1]#n*d
    y = 2**j.unsqueeze(-1).expand_as(k) *(g[...,:d]+k)
    return y

def to_lorentz(u):
    d = u.size(-1)
    if len(u.size())==2:
        uu = th.zeros(u.size(0),d+1)
    else:
        uu = th.zeros(u.size(0), u.size(1), d + 1)
    squnom = th.sum(th.pow(u, 2), dim=-1)#n
    uu[...,0] = th.div(th.ones_like(u[...,-1]),u[...,-1]) +th.div(squnom,4*u[...,-1])#n
    uu[...,1] = th.div(th.ones_like(u[...,-1]),u[...,-1]) -th.div(squnom,4*u[...,-1])#n
    uu[...,2:] = th.div(u[...,:d-1],u[...,-1].unsqueeze(-1).expand_as(u[...,:d-1]))
    return uu

def to_halfspace(u):
    d = u.size(-1)-1
    if len(u.size())==2:
        uu = th.zeros(u.size(0),2*d+1)
    else:
        uu = th.zeros(u.size(0), u.size(1), 2*d+1)
    uu[...,d-1] = 2 * th.div(th.ones_like(u[...,0]),u[...,0]+u[...,1])#n,n*m
    uu[...,:d-1] = 2 * th.div(u[...,2:],u[...,0].unsqueeze(-1).expand_as(u[...,2:])+u[...,1].unsqueeze(-1).expand_as(u[...,2:]))#n
    return uu

def normalize_halfspace_matrix(g):
    y = th.zeros(g.size())
    d = (g.size(-1)-1)//2
    a = th.floor(th.log2(g[...,d-1]))#n
    y[...,-1] = g[...,-1] + a#n
    y[...,d:-2] = th.floor(2**(-1*a).unsqueeze(-1).expand_as(g[...,:d-1]) * (g[...,:d-1] + g[...,d:-2]))#n*(d-1)
    y[...,:d] = 2**(-1*a).unsqueeze(-1).expand_as(g[...,:d]) * (g[...,:d]+g[...,d:-1]) - y[...,d:-1]#n*d
    assert y[...,-2].max()==0
    return y

def normalize_halfspace_matrix_dir(u):
    d = u.size(-1)-1
    y = th.zeros(u.size(0),2*d+1)
    a = th.floor(1-th.log2(u[...,0]+u[...,1]))#n
    y[...,-1] = a#n
    y[...,d:-2] = th.floor(th.div(2*u[...,2:], (2**a*(u[...,0]+u[...,1])).unsqueeze(-1)))
    y[...,:d-1] = th.div(2*u[...,2:], (2**a*(u[...,0]+u[...,1])).unsqueeze(-1)) - y[...,d:-2]
    y[...,d-1] = th.div(2, 2**a*(u[...,0]+u[...,1]))
    assert y[...,-2].max()==0
    return y

class HalfspaceRie1Manifold(Manifold):
    __slots__ = ["eps", "_eps", "norm_clip", "max_norm", "debug"]

    @staticmethod
    def dim(dim):
        return 2*dim+1

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

    def to_poincare_ball(self, uu):
        u = recon_halfplane(uu)
        u = to_lorentz(u)
        x = u.clone()
        d = x.size(-1) - 1
        return x.narrow(-1, 1, d) / (x.narrow(-1, 0, 1) + 1)

    def distance(self, uu, vv):
        dis = HalfspaceRie1Distance.apply(uu, vv)
        return dis

    def pnorm(self, u):
        return th.sqrt(th.sum(th.pow(self.to_poincare_ball(u), 2), dim=-1))

    def normalize(self, w):
        """Normalize vector such that it is located on the hyperboloid"""
        d = (w.size(-1) - 1)//2
        narrowed = w.narrow(-1, d-1, 1)
        narrowed.clamp_(min=1e-8)
        return w

    def normalize1(self, w):
        """Normalize vector such that it is located on the hyperboloid"""
        d = w.size(-1) - 1
        narrowed = w.narrow(-1, 1, d)
        if self.max_norm:
            narrowed.view(-1, d).renorm_(p=2, dim=0, maxnorm=self.max_norm)
        tmp = 1 + th.sum(th.pow(narrowed, 2), dim=-1, keepdim=True)
        tmp.sqrt_()
        w.narrow(-1, 0, 1).copy_(tmp)
        return w


    def normalize_tan(self, x_all, v_all):
        d = v_all.size(1) - 1
        x = x_all.narrow(1, 1, d)
        xv = th.sum(x * v_all.narrow(1, 1, d), dim=1, keepdim=True)
        tmp = 1 + th.sum(th.pow(x_all.narrow(1, 1, d), 2), dim=1, keepdim=True)
        tmp.sqrt_().clamp_(min=self._eps)
        v_all.narrow(1, 0, 1).copy_(xv / tmp)
        return v_all

    def init_weights(self, w, irange=1e-5):
        d = (w.size(-1)-1)//2
        w.data[...,:d-1].uniform_(-irange, irange)
        w.data[...,d-1] = irange * th.rand_like(w[...,d-1])
        w.data[...,d-1].add_(1)
        # ID
        w.data[...,d:].zero_()

    def rgrad(self, p, d_p):
        """Riemannian gradient for hyperboloid"""
        if d_p.is_sparse:
            d = (p.size(-1) - 1) // 2
            u = d_p._values()[...,:d+1]
            x = p.index_select(0, d_p._indices().squeeze())
            x = to_lorentz(recon_halfplane(x))
        else:
            d = (p.size(-1) - 1) // 2
            u = d_p[...,:d+1]
            x = to_lorentz(recon_halfplane(p))
        u.addcmul_(self.ldot(x, u, keepdim=True).expand_as(x), x)
        return d_p

    def expm(self, p, d_p, lr=None, out=None, normalize=False):
        """Exponential map for hyperboloid"""
        d = (p.size(-1) - 1) // 2
        if out is None:
            out = p
        if d_p.is_sparse:
            ix, d_val = d_p._indices().squeeze(), d_p._values()
            d_val = d_val[...,:d+1]
            # This pulls `ix` out of the original embedding table, which could
            # be in a corrupted state.  normalize it to fix it back to the
            # surface of the hyperboloid...
            # TODO: we should only do the normalize if we know that we are
            # training with multiple threads, otherwise this is a bit wasteful
            p_val = self.normalize1(to_lorentz(recon_halfplane(p.index_select(0, ix))))
            ldv = self.ldot(d_val, d_val, keepdim=True)
            if self.debug:
                assert all(ldv > 0), "Tangent norm must be greater 0"
                assert all(ldv == ldv), "Tangent norm includes NaNs"
            nd_p = ldv.clamp_(min=0).sqrt_()
            t = th.clamp(nd_p, max=self.norm_clip)
            nd_p.clamp_(min=self.eps)
            newp = (th.cosh(t) * p_val).addcdiv_(th.sinh(t) * d_val, nd_p)
            if normalize:
                newp = self.normalize1(newp)
            newp = normalize_halfspace_matrix_dir(newp)
#             newp = to_halfspace(newp)
#             newp = normalize_halfspace_matrix(newp)
            p.index_copy_(0, ix, newp)
        else:
            raise "Not implemented"


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

myeps1 = 1e-16
myeps2 = 1e-16

class HalfspaceRie1Distance(Function):
    @staticmethod
    def forward(self, preu, prev):
        d = (preu.size(-1)-1)//2
        assert th.isnan(preu).max()==0, "u includes NaNs"
        assert th.isnan(prev).max()==0, "v includes NaNs"
        assert preu.max() != float('inf') and preu.min() != float('inf')
        assert prev.max() != float('inf') and prev.min() != float('inf')
        if len(preu)<len(prev):
            preu = preu.expand_as(prev)
        elif len(preu)>len(prev):
            prev = prev.expand_as(preu)
        preu[..., d - 1].clamp_(min=myeps1)
        prev[..., d - 1].clamp_(min=myeps1)
        self.ones = (preu[...,-1]>prev[...,-1]).double().unsqueeze(-1).expand_as(preu)
        u = preu*self.ones+prev*(1-self.ones)
        v = prev*self.ones+preu*(1-self.ones)

        self.save_for_backward(u, v)

        j1 = u[...,-1]#m*n
        j2 = v[...,-1]#m*n
        k1 = u[...,d:-1]#m*n*d
        k2 = v[...,d:-1]#m*n*d
        twosR = k1 - (2**(j2-j1)).unsqueeze(-1).expand_as(k2)*k2#m*n*d

        # up1 = twosR + u[...,:d] - 2**(j2-j1).unsqueeze(-1).expand_as(twosR) * v[...,:d] #m*n*d
        # s = th.floor(th.log2(1+th.sqrt(th.sum(th.pow(up1, 2), dim=-1))))  # m*n
        # s = th.zeros_like(s)
        # upp = up1 * 2**(-1*s.unsqueeze(-1).expand_as(twosR))#m*n*d

        s = th.ceil(th.log2(1 + th.sqrt(th.sum(th.pow(twosR, 2), dim=-1))))  # m*n
        R = 2**(-1*s.unsqueeze(-1).expand_as(twosR)) * twosR#m*n*d
        upp = R + 2**(-1*s.unsqueeze(-1).expand_as(R)) * u[...,:d] - 2**(j2-j1-s).unsqueeze(-1).expand_as(R) * v[...,:d]#m*n*d
        lowe = th.clamp(2 * u[...,d-1] * v[...,d-1],min=myeps1)#m*n
        X = th.div(th.sum(th.pow(upp, 2), dim=-1),lowe)#m*n
        nomdis = th.sqrt(th.clamp(X * X + 2 * X * 2**(-2*s-j1+j2),min=0))#m*n sqrt
#         log1t = (2*s+j1-j2)*th.log(th.Tensor([2]))  # m*n
#         log2t = th.clamp(2**(-2*s-j1+j2)+X+nomdis,min=myeps2)#m*n
        out_p = (2*s+j1-j2)*th.log(th.Tensor([2])) + th.log(th.clamp(2**(-2*s-j1+j2)+X+nomdis,min=myeps2))#m*n
        self.ins = out_p.clone()
        return out_p

    @staticmethod
    def backward(self, g):
        u, v = self.saved_tensors
        d = (u.size(-1)-1)//2
        gu = th.zeros(u.size())  # m*n*(2d+1)
        gv = th.zeros(v.size())  # m*n*(2d+1)
        uu = to_lorentz(recon_halfplane(u))
        vv = to_lorentz(recon_halfplane(v))#n*m*(d+1)
        g = g.unsqueeze(-1)
        # squnorm = th.clamp(th.sum(uu[..., 1:] * uu[..., 1:], dim=-1), min=0)
        # sqvnorm = th.clamp(th.sum(vv[..., 1:] * vv[..., 1:], dim=-1), min=0)
        # sqdist = th.sum(uu[..., 1:] * vv[..., 1:], dim=-1)
        # gu[...,:d+1] = grad(uu, vv, squnorm, sqvnorm, sqdist)
        # gv[...,:d+1] = grad(vv, uu, sqvnorm, squnorm, sqdist)
        gu[..., :d + 1] = grad(uu, vv, self.ins)
        gv[..., :d + 1] = grad(vv, uu, self.ins)
        ###########################
        guu = gu*self.ones+gv*(1-self.ones)
        gvv = gv*self.ones+gu*(1-self.ones)
        return g.expand_as(gu) * guu, g.expand_as(gv) * gvv

def grad(x, v, t):
    z = th.clamp(th.sinh(t), min=myeps1).unsqueeze(-1)
    out = -v / z.expand_as(x)
    assert out.max() != float('inf') and out.min() != float('inf')
    return out