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

class HalfspaceRieManifold(Manifold):
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
    
    def sinhc(self, u):
        return th.div(th.sinh(u),u)

    def to_poincare_ball(self, uu):
        u = recon_halfplane(uu)
        u = to_lorentz(u)
        x = u.clone()
        d = x.size(-1) - 1
        return x.narrow(-1, 1, d) / (x.narrow(-1, 0, 1) + 1)

    def distance(self, uu, vv):
        dis = HalfspaceRieDistance.apply(uu, vv)
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
#         w.data[...,:d-1].uniform_(-1, 1)
        w.data[...,d-1] = irange * th.rand_like(w[...,d-1])
        w.data[...,d-1].add_(1)
#         w.data[...,:d-1].add_(1)
        # ID
        w.data[...,d:].zero_()

    def rgrad(self, p, d_p):
        d = (p.size(-1)-1)//2
        """Euclidean gradient for hyperboloid"""
        if d_p.is_sparse:
            u = d_p._values()
            x = p.index_select(0, d_p._indices().squeeze())
        else:
            u = d_p
            x = p
        u.mul_((x[...,d-1]).unsqueeze(-1))###rgrad
#         u.mul_((x[...,d-1]*x[...,d-1]).unsqueeze(-1))###rgrad
        return d_p

    def expm(self, p, d_p, lr=None, out=None, normalize=False):
        """Exponential map for hyperboloid"""
        d = (p.size(-1)-1)//2
        if out is None:
            out = p
        if d_p.is_sparse:
            ix, d_val = d_p._indices().squeeze(), d_p._values()
            # This pulls `ix` out of the original embedding table, which could
            # be in a corrupted state.  normalize it to fix it back to the
            # surface of the hyperboloid...
            # TODO: we should only do the normalize if we know that we are
            # training with multiple threads, otherwise this is a bit wasteful
            p_val = self.normalize(p.index_select(0, ix))
            newp_val = p_val.clone()
            s = th.norm(d_val[...,:d],dim=-1)#n
            newp_val[...,:d-1] = p_val[...,:d-1] + th.div(p_val[...,d-1], th.div(th.cosh(s), self.sinhc(s))-d_val[...,d-1]).unsqueeze(-1).expand_as(d_val[...,:d-1]) * d_val[...,:d-1]#n*(d-1)
            newp_val[...,d-1] = th.div(p_val[...,d-1], th.cosh(s)-d_val[...,d-1]*self.sinhc(s))#n
            newp_val = self.normalize(newp_val)
            p.index_copy_(0, ix, newp_val)
        else:
            if lr is not None:
                d_p.mul_(-lr)
            newp = p + d_p
            newp = self.normalize(newp)
            p.copy_(newp)


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

class HalfspaceRieDistance(Function):
    @staticmethod
    def forward(self, preu, prev, optt = 2):
        self.optt = optt
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
        if preu.dtype == th.float64:
            self.ones = (preu[...,-1]>prev[...,-1]).double().unsqueeze(-1).expand_as(preu)
        elif preu.dtype == th.float32:
            self.ones = (preu[...,-1]>prev[...,-1]).float().unsqueeze(-1).expand_as(preu)
        u = preu*self.ones+prev*(1-self.ones)
        v = prev*self.ones+preu*(1-self.ones)

        self.save_for_backward(u, v)

        self.j1 = u[...,-1]#m*n
        self.j2 = v[...,-1]#m*n
        k1 = u[...,d:-1]#m*n*d
        k2 = v[...,d:-1]#m*n*d

        if optt == 1:
            ###################
            self.upp = k1 - (2**(self.j2-self.j1)).unsqueeze(-1).expand_as(k2)*k2 + u[...,:d] - (2**(self.j2-self.j1)).unsqueeze(-1).expand_as(k2)*v[...,:d]#m*n*d
            self.Xprime = th.div(th.sum(th.pow(self.upp, 2), dim=-1), u[...,d-1] * v[...,d-1])#m*n
            self.inside_log_1 = 1+2**(self.j1-self.j2-1)*self.Xprime#m*n
            self.inside_log_2 = th.sqrt(2**(2*(self.j1-self.j2-1))*self.Xprime*self.Xprime+2**(self.j1-self.j2)*self.Xprime)#m*n
            return th.log(self.inside_log_1+self.inside_log_2)
        elif optt == 2:
            ###################
            twos_upp = k1 - (2**(self.j2-self.j1)).unsqueeze(-1).expand_as(k2)*k2 + u[...,:d] - (2**(self.j2-self.j1)).unsqueeze(-1).expand_as(k2)*v[...,:d]##m*n*d
            norm_twos_upp = th.sqrt(th.sum(th.pow(twos_upp, 2), dim=-1))
            self.zero_mask = (norm_twos_upp !=0.0)##m*n
            self.s = th.zeros_like(norm_twos_upp)###m*n
            self.s[self.zero_mask] = th.ceil(th.log2(norm_twos_upp))[self.zero_mask]  # m*n
            self.upp = 2**(-1*self.s.unsqueeze(-1).expand_as(twos_upp)) * twos_upp#m*n*d
            self.X = th.div(th.sum(th.pow(self.upp, 2), dim=-1),2 * u[...,d-1] * v[...,d-1])#m*n
            self.nomdis = th.sqrt(th.clamp(self.X * self.X + 2 * self.X * 2**(-2*self.s-self.j1+self.j2),min=0))#m*n sqrt
            log1t = (2*self.s+self.j1-self.j2)*th.log(th.Tensor([2]))  # m*n
            self.log2t = 2**(-2*self.s-self.j1+self.j2)+self.X+self.nomdis#m*n
            return log1t + th.log(self.log2t)#m*n
        elif optt == 3:
            ###################
            twosR = k1 - (2**(self.j2-self.j1)).unsqueeze(-1).expand_as(k2)*k2#m*n*d
            self.s = th.ceil(th.log2(1 + th.sqrt(th.sum(th.pow(twosR, 2), dim=-1))))  # m*n
            R = 2**(-1*self.s.unsqueeze(-1).expand_as(twosR)) * twosR#m*n*d
            self.upp = R + 2**(-1*self.s.unsqueeze(-1).expand_as(R)) * u[...,:d] - 2**(self.j2-self.j1-self.s).unsqueeze(-1).expand_as(R) * v[...,:d]#m*n*d
            print(self.s.max())
            print(th.log(th.sum(th.pow(self.upp, 2), dim=-1).min()))
            lowe = th.clamp(2 * u[...,d-1] * v[...,d-1],min=myeps1)#m*n
            self.X = th.div(th.sum(th.pow(self.upp, 2), dim=-1),lowe)#m*n
            self.nomdis = th.sqrt(th.clamp(self.X * self.X + 2 * self.X * 2**(-2*self.s-self.j1+self.j2),min=0))#m*n sqrt
            log1t = (2*self.s+self.j1-self.j2)*th.log(th.Tensor([2]))  # m*n
    #         self.log2t = th.clamp(2**(-2*self.s-self.j1+self.j2)+self.X+self.nomdis,min=myeps2)#m*n
            self.log2t = 2**(-2*self.s-self.j1+self.j2)+self.X+self.nomdis#m*n
    #         if self.nomdis.min()==0:
    #         print(th.log(self.nomdis.min()))
            return log1t + th.log(self.log2t)#m*n

    @staticmethod
    def backward(self, g):
        if self.optt == 1:
            u, v = self.saved_tensors
            d = (u.size(-1) - 1) // 2
            u[..., d - 1].clamp_(min=myeps1)
            v[..., d - 1].clamp_(min=myeps1)
            g = g.unsqueeze(-1).expand_as(u).clone()
            gu = th.zeros_like(u)  # m*n*(2d+1)
            gv = th.zeros_like(v)  # m*n*(2d+1)
            auxli_term1 = th.div(th.ones_like(self.inside_log_1), self.inside_log_1+self.inside_log_2)*(2**(self.j1-self.j2-1)+th.div(2**(2*(self.j1-self.j2-1))*self.Xprime + 2**(self.j1-self.j2-1), self.inside_log_2)) #m*n
            auxli_term2 = th.div(2*self.upp, (u[...,d-1] * v[...,d-1]).unsqueeze(-1).expand_as(self.upp))#m*n*d
            gu[..., :d - 1] = auxli_term1.unsqueeze(-1).expand_as(u[...,:d-1]) * auxli_term2[...,:d-1]#m*n*(d-1)
            gu[..., d - 1] = auxli_term1 * (auxli_term2[...,d-1]-th.div(self.Xprime,u[...,d-1]))#m*n
            gv[..., :d - 1] = -1 * 2 ** (self.j2 - self.j1).unsqueeze(-1).expand_as(u[...,:d-1]) * gu[..., :d - 1]  # m*n*(d-1)
            gv[..., d - 1] = auxli_term1 * (-1 * 2 ** (self.j2 - self.j1) * auxli_term2[...,d-1]-th.div(self.Xprime,v[...,d-1]))#m*n
            guu = gu*self.ones+gv*(1-self.ones)
            gvv = gv*self.ones+gu*(1-self.ones)
            assert gu.max() != float("Inf"), " gu max includes inf"
            assert gv.max() != float("Inf"), " gv max includes inf"
            assert gu.min() != float("Inf"), " gu min includes inf"
            assert gv.min() != float("Inf"), " gv min includes inf"
            assert th.isnan(gu).max() == 0, "gu includes NaNs"
            assert th.isnan(gv).max() == 0, "gv includes NaNs"
            return g * guu, g * gvv
        else:
            u, v = self.saved_tensors
            d = (u.size(-1) - 1) // 2
            u[..., d - 1].clamp_(min=myeps1)
            v[..., d - 1].clamp_(min=myeps1)
            g = g.unsqueeze(-1).expand_as(u).clone()
            gu = th.zeros_like(u)  # m*n*(2d+1)
            gv = th.zeros_like(v)  # m*n*(2d+1)
            auxli_term1 = th.div(th.ones_like(self.log2t), self.log2t)*(1+th.div(self.X + 2**(-2*self.s-self.j1+self.j2), self.nomdis)) #m*n
            auxli_term2 = th.div(self.upp, (2**(self.s) * u[...,d-1] * v[...,d-1]).unsqueeze(-1).expand_as(self.upp))#m*n*d
            gu[..., :d - 1] = auxli_term1.unsqueeze(-1).expand_as(u[...,:d-1]) * auxli_term2[...,:d-1]#m*n*(d-1)
            gu[..., d - 1] = auxli_term1 * (auxli_term2[...,d-1]-th.div(self.X,u[...,d-1]))#m*n
            gv[..., :d - 1] = -1 * 2 ** (self.j2 - self.j1).unsqueeze(-1).expand_as(u[...,:d-1]) * gu[..., :d - 1]  # m*n*(d-1)
            gv[..., d - 1] = auxli_term1 * (-1 * 2 ** (self.j2 - self.j1) * auxli_term2[...,d-1]-th.div(self.X,v[...,d-1]))#m*n
            if g.dtype == th.float64:
                zero_mask = self.zero_mask.unsqueeze(-1).expand_as(gu)
            elif g.dtype == th.float32:
                zero_mask = self.zero_mask.unsqueeze(-1).expand_as(gu)
            guu = th.zeros_like(u)
            gvv = th.zeros_like(v)
            guu[zero_mask] = (gu*self.ones+gv*(1-self.ones))[zero_mask]
            gvv[zero_mask] = (gv*self.ones+gu*(1-self.ones))[zero_mask]
            assert guu.max() != float("Inf"), " gu max includes inf"
            assert gvv.max() != float("Inf"), " gv max includes inf"
            assert guu.min() != float("Inf"), " gu min includes inf"
            assert gvv.min() != float("Inf"), " gv min includes inf"
            assert th.isnan(guu).max() == 0, "gu includes NaNs"
            assert th.isnan(gvv).max() == 0, "gv includes NaNs"
            return g * guu, g * gvv