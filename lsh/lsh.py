# -*- coding: utf-8 -*-

import math
import random
from abc import ABCMeta, abstractmethod

import numpy as np

# matrix inner product
def dot(u, v):    
    # return sum(ux * vx for ux, vx in zip(u,v))
    return np.dot(u, v)

def g_ext_norm(vec, m):      #L2-ALSH 
    l2norm_square = dot(vec, vec)
    return [l2norm_square**(i+1) for i in xrange(m)]

def g_ext_half(m):			#L2-ALSH 
    return [0.5 for i in xrange(m)]

def g_ext_zero(m):   #sign-ALSH 
    return [0 for i in xrange(m)]

# [x] => [x;    ||x||**2; ||x||**4; ...; ||x||**(2*m);    1/2; ...; 1/2(m)]
def g_index_extend(datas, m):
    return [(dv + g_ext_norm(dv, m) + g_ext_half(m)) for dv in datas]


# [x] => [x;    1/2; ...; 1/2(m);    ||x||**2; ||x||**4; ...; ||x||**(2*m)]
def g_query_extend(queries, m):
    return [(qv + g_ext_half(m) + g_ext_norm(qv, m)) for qv in queries]


# get max norm for two-dimension list
def g_max_norm(datas):
    norm_list = [math.sqrt(dot(dd, dd)) for dd in datas]
    return max(norm_list)


# datas transformation. S(xi) = (U / M) * xi
def g_transformation(datas):
    # U < 1  ||xi||2 <= U <= 1. recommend for 0.83
    U = 0.83
    #U = 0.75
    max_norm = g_max_norm(datas)
    ratio = float(U / max_norm)
    return ratio, max_norm, [[ratio * dx for dx in dd] for dd in datas]


# normalization for each query  
def g_normalization(queries):
    U = 0.83
    #U = 0.75
    norm_queries = []
    for qv in queries:
        norm = math.sqrt(dot(qv, qv))
        ratio = float(U / norm)
        norm_queries.append([ratio * qx for qx in qv])
    return norm_queries

class Hash:
    'hash base'

    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def hash(self, vec):
        print "Hash.hash()"
        pass

    @staticmethod
    def distance(u, v):
        print "Hash.distance()"
        pass

    @staticmethod
    def combine(hashes):
        # print "Hash.combine()"
        return str(hashes)

class L2Lsh(Hash):
    'L2 LSH'

    # r: fixed size
    # d: data length
    # RandomData: random data vector
    def __init__(self, r, d):     
        self.r, self.d = r, d
        self.b = random.uniform(0, self.r)      # 0 < b < r     
        self.Data = [random.gauss(0, 1) for i in xrange(self.d)]

    def hash(self, vec):	# hash family 
        # use str() as a naive way of forming a single value
        return int((dot(vec, self.Data) + self.b) / self.r)

    # Euclidean Distance
    @staticmethod
    def distance(u, v):
        # print "L2Lsh.distance()"
        return sum((ux - vx)**2 for ux, vx in zip(u, v))**0.5

