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

def to_lorentz(u, u_int_matrix):
    L = th.sqrt(th.Tensor([[3, 0, 0], [0, 1, 0], [0, 0, 1]]))
    R = th.sqrt(th.Tensor([[1.0 / 3.0, 0, 0], [0, 1, 0], [0, 0, 1]]))
#     uu = th.matmul(L.expand_as(u_int_matrix), th.matmul(u_int_matrix.float(), th.matmul(R.expand_as(u_int_matrix), u[..., :3].unsqueeze(-1)))).squeeze(-1)
    uu = th.matmul(L.expand_as(u_int_matrix), th.matmul(u_int_matrix, th.matmul(R.expand_as(u_int_matrix), u[..., :3].unsqueeze(-1)))).squeeze(-1)
    return uu

class GroupRieManifold(Manifold):
    __slots__ = ["eps", "_eps", "norm_clip", "max_norm", "debug"]

    @staticmethod
    def dim(dim):
        return 3

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

    def to_poincare_ball(self, uu, uu_int_matrix):
        u = to_lorentz(uu, uu_int_matrix)
        x = u.clone()
        d = x.size(-1) - 1
        return x.narrow(-1, 1, d) / (x.narrow(-1, 0, 1) + 1)

    def distance(self, uu, uu_int_matrix, vv, vv_int_matrix):
        dis = GroupRieDistance.apply(uu, uu_int_matrix, vv, vv_int_matrix)
        return dis

    def pnorm(self, u, u_int_matrix):
        return th.sqrt(th.sum(th.pow(self.to_poincare_ball(u, u_int_matrix), 2), dim=-1))

    def normalize(self, ww, gra=False):
        """Normalize vector such that it is located on the hyperboloid"""
        if not gra:
            w = ww[...,:3]
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
        w.data[...,1:3].uniform_(-irange, irange)
        w.data[...,0] = th.sqrt(th.clamp(th.sum(w[...,1:3] * w[...,1:3], dim=-1),min=0) + 1)

    def init_weights_int_matrix(self, w):
        ID = th.eye(3,3)
        w.data.zero_()
        w.data.add_(ID)
    
    def rgrad(self, p, d_p):
        """Riemannian gradient for hyperboloid"""
        if d_p.is_sparse:
            u = d_p._values()
            x = p.index_select(0, d_p._indices().squeeze())
        else:
            u = d_p
            x = p
        u.narrow(-1, 0, 1).mul_(-1)
        u[...,:3].addcmul_(self.ldot(x[...,:3], u[...,:3], keepdim=True).expand_as(x[...,:3]), x[...,:3])
        return d_p


    def expm(self, pp, d_p, lr=None, out=None, normalize=False):
        p = pp[...,:3]
        """Exponential map for hyperboloid"""
        if out is None:
            out = p
        if d_p.is_sparse:
            ix, d_val1 = d_p._indices().squeeze(), d_p._values()
            d_val = d_val1[..., :3]
            # This pulls `ix` out of the original embedding table, which could
            # be in a corrupted state.  normalize it to fix it back to the
            # surface of the hyperboloid...
            # TODO: we should only do the normalize if we know that we are
            # training with multiple threads, otherwise this is a bit wasteful
            p_val = self.normalize(p.index_select(0, ix),gra=True)
            ldv = self.ldot(d_val, d_val, keepdim=True)
            if self.debug:
                assert all(ldv > 0), "Tangent norm must be greater 0"
                assert all(ldv == ldv), "Tangent norm includes NaNs"
            nd_p = ldv.clamp_(min=0).sqrt_()
            t = th.clamp(nd_p, max=self.norm_clip)
            nd_p.clamp_(min=self.eps)
            newp = (th.cosh(t) * p_val).addcdiv_(th.sinh(t) * d_val, nd_p)
            if normalize:
                newp = self.normalize(newp,gra=True)
            p.index_copy_(0, ix, newp)
        else:
            d_p1 = d_p[...,:3].clone()
            if lr is not None:
                d_p1.narrow(-1, 0, 1).mul_(-1)
                d_p1.addcmul_((self.ldot(p, d_p1, keepdim=True)).expand_as(p), p)
                d_p1.mul_(-lr)
            ldv = self.ldot(d_p1, d_p1, keepdim=True)
            if self.debug:
                assert all(ldv > 0), "Tangent norm must be greater 0"
                assert all(ldv == ldv), "Tangent norm includes NaNs"
            nd_p = ldv.clamp_(min=0).sqrt_()
            t = th.clamp(nd_p, max=self.norm_clip)
            nd_p.clamp_(min=self.eps)
            newp = (th.cosh(t) * p).addcdiv_(th.sinh(t) * d_p1, nd_p)
            if normalize:
                newp = self.normalize(newp,gra=True)
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

myeps1 = 1e-8
myeps2 = 1e-16
myn = 25##for float64
# myn = 11##for float32

class GroupRieDistance(Function):
    @staticmethod
    def forward(self, u, u_int_matrix, v, v_int_matrix, optt =1):
        assert th.isnan(u_int_matrix).max()==0, "u includes NaNs"
        assert th.isnan(v_int_matrix).max()==0, "v includes NaNs"
        assert th.abs(u_int_matrix).max() < 2**25, "u_int_matrix may include Inf integers"
        assert th.abs(v_int_matrix).max() < 2**25, "v_int_matrix may include Inf integers"
        assert th.abs(u).max() < 100, "u are out of F"
        assert th.abs(v).max() < 100, "v are out of F"
        assert u.max() != float('inf') and u.min() != float('inf')
        assert v.max() != float('inf') and v.min() != float('inf')
        if len(u)<len(v):
            u = u.expand_as(v)
            u_int_matrix = u_int_matrix.expand_as(v_int_matrix)
        elif len(u)>len(v):
            v = v.expand_as(u)
            v_int_matrix = v_int_matrix.expand_as(u_int_matrix)
        self.save_for_backward(u, v)
        M3 = th.Tensor([[3, 0, 0], [0, -1, 0], [0, 0, -1]])
        R = th.sqrt(th.Tensor([[1.0 / 3.0, 0, 0], [0, 1, 0], [0, 0, 1]]))
        if optt == 1:
            #####################
            u_int_matrix2 = th.fmod(u_int_matrix, 2**myn)
            u_int_matrix1 = u_int_matrix - u_int_matrix2
            v_int_matrix2 = th.fmod(v_int_matrix, 2**myn)
            v_int_matrix1 = v_int_matrix - v_int_matrix2
            self.hatQ = th.matmul(u_int_matrix1.transpose(-2,-1), th.matmul(M3.expand_as(u_int_matrix), v_int_matrix1))+(th.matmul(u_int_matrix1.transpose(-2,-1), th.matmul(M3.expand_as(u_int_matrix), v_int_matrix2))+th.matmul(u_int_matrix2.transpose(-2,-1), th.matmul(M3.expand_as(u_int_matrix), v_int_matrix1)))+th.matmul(u_int_matrix2.transpose(-2,-1), th.matmul(M3.expand_as(u_int_matrix), v_int_matrix2))#cpu, this may overflow, need to decompose them in some way
            RThatQR = th.matmul(R.expand_as(self.hatQ),th.matmul(self.hatQ, R.expand_as(self.hatQ)))#cpu float
            d_c = th.matmul(u[..., :3].unsqueeze(-1).transpose(-2,-1), th.matmul(RThatQR, v[..., :3].unsqueeze(-1))).squeeze(-1).squeeze(-1)#cpu float
            self.nomdis = th.sqrt(th.clamp(d_c*d_c-th.ones_like(d_c),min=myeps2))#cpu float
            outp = th.log(th.clamp(d_c + self.nomdis,min=myeps1))#cpu float
        elif optt == 2:
            #####################
            #####################
            Q = th.matmul(u_int_matrix.transpose(-2,-1), th.matmul(M3.expand_as(u_int_matrix), v_int_matrix))#cpu, this may overflow, need to decompose them in some way
            Q11 = th.clamp(Q.narrow(-2,0,1).narrow(-1,0,1),min=myeps1)#Long tensor, cpu
    #         self.hatQ = th.div(Q.float(), Q11.float().expand_as(Q))#cpu float
            self.hatQ = th.div(Q, Q11.expand_as(Q))
            RThatQR = th.matmul(R.expand_as(self.hatQ),th.matmul(self.hatQ, R.expand_as(self.hatQ)))#cpu float
            d_c = th.matmul(u[..., :3].unsqueeze(-1).transpose(-2,-1), th.matmul(RThatQR, v[..., :3].unsqueeze(-1))).squeeze(-1).squeeze(-1)#cpu float
    #         invQ11 = th.div(th.ones_like(Q11.squeeze(-1).squeeze(-1)).float(),Q11.float().squeeze(-1).squeeze(-1))#cpu float
            invQ11 = th.div(th.ones_like(Q11.squeeze(-1).squeeze(-1)),Q11.squeeze(-1).squeeze(-1))#cpu float
            self.nomdis = th.sqrt(th.clamp(d_c*d_c-invQ11*invQ11,min=myeps2))#cpu float
    #         outp = th.log(Q11.float().squeeze(-1).squeeze(-1)) + th.log(th.clamp(d_c + self.nomdis,min=myeps1))#cpu float
            outp = th.log(Q11.squeeze(-1).squeeze(-1)) + th.log(th.clamp(d_c + self.nomdis,min=myeps1))#cpu float
        elif optt == 3:
            ####################
            self.hatQ = th.matmul(u_int_matrix.transpose(-2,-1), th.matmul(M3.expand_as(u_int_matrix), v_int_matrix)).double()#cpu
            RThatQR = th.matmul(R.expand_as(self.hatQ),th.matmul(self.hatQ, R.expand_as(self.hatQ)))#cpu float
            d_c = th.matmul(u[..., :3].unsqueeze(-1).transpose(-2,-1), th.matmul(RThatQR, v[..., :3].unsqueeze(-1))).squeeze(-1).squeeze(-1)#cpu float
            invQ11 = th.ones_like(d_c)#cpu float
            self.nomdis = th.sqrt(th.clamp(d_c*d_c-invQ11*invQ11,min=myeps2))#cpu float
            outp = th.log(th.clamp(d_c + self.nomdis,min=myeps1))#cpu float
        ####################
        return outp

    @staticmethod
    def backward(self, g):
        R = th.sqrt(th.Tensor([[1.0 / 3.0, 0, 0], [0, 1, 0], [0, 0, 1]])).unsqueeze(0).unsqueeze(0).expand_as(self.hatQ)
        u, v = self.saved_tensors
        g = g.unsqueeze(-1).expand_as(u)
        uupfrac = th.matmul(R,th.matmul(self.hatQ, th.matmul(R,v[..., :3].unsqueeze(-1)))).squeeze(-1)
        vupfrac = th.matmul(R,th.matmul(self.hatQ.transpose(-2,-1), th.matmul(R,u[..., :3].unsqueeze(-1)))).squeeze(-1)
        gu = th.div(uupfrac, self.nomdis.unsqueeze(-1).expand_as(uupfrac))
        gv = th.div(vupfrac, self.nomdis.unsqueeze(-1).expand_as(vupfrac))
        assert gu.max() != float("Inf"), " gu max includes inf"
        assert gv.max() != float("Inf"), " gv max includes inf"
        assert gu.min() != float("Inf"), " gu min includes inf"
        assert gv.min() != float("Inf"), " gv min includes inf"
        assert th.isnan(gu).max() == 0, "gu includes NaNs"
        assert th.isnan(gv).max() == 0, "gv includes NaNs"
        return g * gu, None, g * gv, None