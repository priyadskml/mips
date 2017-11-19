# -*- coding: utf-8 -*-

import math
import random
import numpy as np
from operator import itemgetter
from collections import defaultdict
from abc import ABCMeta, abstractmethod

def g_ext_norm(vec, m):      #L2-ALSH
    l2norm_square = np.dot(vec, vec)
    return [l2norm_square**(i+1) for i in xrange(m)]

# get max norm for two-dimension list
def g_max_norm(datas):
    norm_list = [math.sqrt(np.dot(dd, dd)) for dd in datas]
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
        norm = math.sqrt(np.dot(qv, qv))
        ratio = float(U / norm)
        norm_queries.append([ratio * qx for qx in qv])
    return norm_queries

class Hash:
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
        return str(hashes)

class L2Lsh(Hash):
    def __init__(self, r, d):
        self.r, self.d = r, d
        self.b = random.uniform(0, self.r)      # 0 < b < r
        self.Data = [random.gauss(0, 1) for i in xrange(self.d)]

    def hash(self, vec):	# hash family
        # use str() as a naive way of forming a single value
        return int((np.dot(vec, self.Data) + self.b) / self.r)

    # Euclidean Distance
    @staticmethod
    def distance(u, v):
        # print "L2Lsh.distance()"
        return sum((ux - vx)**2 for ux, vx in zip(u, v))**0.5

class LshTester():
    # datas: datas for build hash index
    # queries: query datas
    # rand_range: random range for norm
    # num_neighbours: query top num_neighbours
    def __init__(self, datas, queries, rand_range = 1.0, num_neighbours = 1):
        kdata, qdata = len(datas[0]), len(queries[0])

        self.d = kdata
        self.datas = datas
        self.queries = queries
        self.rand_range = rand_range
        self.q_num = len(self.queries)
        self.num_neighbours = num_neighbours

    def linear(self, q, metric, max_results):
        candidates = [(ix, metric(q, p)) for ix, p in enumerate(self.datas)]
        temp = sorted(candidates, key=itemgetter(1))[:max_results]
        return temp

    def run(self, type, k_vec = [2], l_vec = [2]):
        # set distance func object
        if 'l2' == type:
            metric = L2Lsh.distance
        elif 'cosine' == type:
            metric = CosineLsh.distance
        start = time.time()
        exact_hits = [[ix for ix, dist in self.linear(q, metric, self.num_neighbours)] for q in self.queries]   
        linear_time = time.time() - start
        print '=============================='
        print type + ' TEST:'
        # print 'L\tk\tacc\tData_Points_seen\tPrecision\tRecall\tTime'

        for k in k_vec:
            lsh = LshWrapper(type, self.d, self.rand_range, k, 0)

            for L in l_vec:
                lsh.resize(L)
                lsh.index(self.datas)

                correct, precision, recall , start = 0 , 0 , 0 , time.time()
                for q, hits in zip(self.queries, exact_hits):
                    lsh_hits = [ix for ix, dist in lsh.query(q, metric, self.num_neighbours)]
                    correct += int(lsh_hits == hits)
                precision = correct
                print "{0}\t{1}\t{2}\t{3}\t\t{4}\t\t{5}\t{6}".format(L, k, float(correct) / self.q_num, float(lsh.get_avg_touched()) / len(self.datas), precision, recall, time.time()-start)

class L2AlshTester(LshTester):
    'L2-ALSH for MIPS Test parameters'

    # datas: datas for build hash index
    # queries: query datas
    # rand_range: random range for norm
    # num_neighbours: query top num_neighbours
    # m: ALSH extend metrix length. default 3
    def __init__(self, datas, queries, rand_range = 1.0, num_neighbours = 1, m = 3):
        kdata = len(datas[0])
        qdata = len(queries[0])
        self.m = m
        self.half_extend = g_ext_half(self.m)
        # storage original datas & queries. used for validation
        self.origin_datas = datas
        self.origin_queries = queries
        self.q_num = len(self.origin_queries)
        # datas & queries transformation
        dratio, dmax_norm, self.norm_datas = g_transformation(self.origin_datas)
        self.norm_queries = g_normalization(self.origin_queries)
        # expand k dimension into k+2m dimension
        self.ext_datas = g_index_extend(self.norm_datas, self.m)
        self.ext_queries = g_query_extend(self.norm_queries, self.m)
        new_len = kdata + 2 * m
        LshTester.__init__(self, self.ext_datas, self.ext_queries, rand_range, num_neighbours)

    # MIPS
    def linear(self, q, metric, max_results):
        """ brute force search by linear scan """
        # print 'MipsLshTester linear:'
        candidates = [(ix, metric(q, p)) for ix, p in enumerate(self.origin_datas)]
        temp = sorted(candidates, key=itemgetter(1), reverse=True)[:max_results]
        # print "I am here"
        return temp

    def run(self, type, k_vec = [2], l_vec = [2]):
        if 'l2' == type:
            pass
        else:
            raise ValueError

        validate_metric, compute_metric = dot, L2Lsh.distance
        start = time.time()
        exact_hits = [[ix for ix, dist in self.linear(q, compute_metric, self.num_neighbours)] for q in self.origin_queries]
        exact_hits = [[ix for ix, dist in self.linear(q, validate_metric, self.num_neighbours)] for q in self.origin_queries]
        linear_time = time.time() - start

        print '=============================='
        print 'L2AlshTester ' + type + ' TEST:'
        print 'L\tk\tacc\tData_ratio\tPrecision\tRecall\ttime'

        # concatenating more hash functions increases selectivity
        for k in k_vec:
            lsh = LshWrapper(type, self.d, self.rand_range, k, 0)

            # using more hash tables increases recall
            for L in l_vec:
                lsh.resize(L)
                lsh.index(self.ext_datas)

                correct, precision, recall , start = 0 , 0 , 0 , time.time()
                for q, hits in zip(self.ext_queries, exact_hits):
                    lsh_hits = [ix for ix, dist in lsh.query(q, compute_metric, self.num_neighbours)]
                    correct += int(lsh_hits == hits)
                lash_time = time.time() - start
                print "{0}\t{1}\t{2}\t{3}\t\t{4}\t\t{5}\t{6}\t{7}".format(L, k, float(correct) / self.q_num, float(lsh.get_avg_touched()) / len(self.datas), precision, recall , lash_time, linear_time)
    
    @staticmethod
    # type: l2 & cosine
    # mips: True for ALSH
    def createTester(type, datas, queries, rand_num, num_neighbours):
        return L2AlshTester(datas, queries, rand_num, num_neighbours)
