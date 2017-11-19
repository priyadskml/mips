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

class LshWrapper:
    'LSH Wrapper'

    # lsh_type: lsh hash func type: in 'l2
    # r: random float data
    # d: data vector size
    # k: number of hash func for each hashtable. default 2
    # L: number of hash tables for each hash type: default 2
    def __init__(self, lsh_type, d, r = 1.0, k = 2, L = 2):
        self.type = lsh_type
        self.d, self.r, self.k, self.L, self.hash_tables = d, r, k, 0, []
        self.resize(L)

    def __get_hash_class__(self):
        return L2Lsh
        # return CosineLsh

    def create_hash_table(self):
        return L2Lsh(self.r, self.d)
        # return CosineLsh(self.d)

    def resize(self, L):
        # shrink the number of hash tables to be used
        if L < self.L:
            self.hash_tables = self.hash_tables[:L]
        else:
            # initialise a new hash table for each hash function
            hash_funcs = [[self.create_hash_table() for h in xrange(self.k)] for l in xrange(self.L, L)]
            self.hash_tables.extend([(g, defaultdict(lambda:[])) for g in hash_funcs])
        self.L = L

    def hash(self, ht, data):
        #  used for combine
        return self.__get_hash_class__().combine([h.hash(data) for h in ht])

    def index(self, datas):
        # index the supplied datas
        self.datas = datas
        for ht, ct in self.hash_tables:
            for ix, p in enumerate(self.datas):
                ct[self.hash(ht, p)].append(ix)
        # reset stats
        self.tot_touched = 0
        self.num_queries = 0

    def query(self, q, metric, max_results = 1):
        """
        triple_l = 3 * self.L
        if max_results > triple_l:
            max_results = triple_l
        elif
        """
        if 0 == max_results:
            max_results = 1
        # find the max_results closest indexed datas to q according to the supplied metric
        candidates = set()
        for ht, ct in self.hash_tables:
            matches = ct.get(self.hash(ht, q), [])
            candidates.update(matches)
        # update stats
        self.tot_touched += len(candidates)
        self.num_queries += 1
        # rerank candidates
        candidates = [(ix, metric(q, self.datas[ix])) for ix in candidates]
        candidates.sort(key=itemgetter(1))
        return candidates[:max_results]

    def get_avg_touched(self):
        # mean number of candidates inspected per query
        return self.tot_touched / self.num_queries

    def distance(self, u, v):
        return __get_hash_class__.distance(u, v)

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

        exact_hits = [[ix for ix, dist in self.linear(q, metric, self.num_neighbours)] for q in self.queries]   #exact_hits是距离q最近的那些点（这是真的，精准的）

        print '=============================='
        print type + ' TEST:'
        print 'L\tk\tacc\ttouch'

        for k in k_vec:
            lsh = LshWrapper(type, self.d, self.rand_range, k, 0)

            for L in l_vec:
                lsh.resize(L)
                lsh.index(self.datas)

                correct = 0
                for q, hits in zip(self.queries, exact_hits):
                    lsh_hits = [ix for ix, dist in lsh.query(q, metric, self.num_neighbours)]
                    correct += int(lsh_hits == hits)

                print "{0}\t{1}\t{2}\t{3}".format(L, k, float(correct) / self.q_num, float(lsh.get_avg_touched()) / len(self.datas))

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
        self.half_extend = [0.5 for i in xrange(self.m)]
        # storage original datas & queries. used for validation
        self.origin_datas = datas
        self.origin_queries = queries
        self.q_num = len(self.origin_queries)
        # datas & queries transformation
        dratio, dmax_norm, self.norm_datas = g_transformation(self.origin_datas)
        self.norm_queries = g_normalization(self.origin_queries)
        # expand k dimension into k+2m dimension
        self.ext_datas = [(dv + g_ext_norm(dv, self.m) + [0.5 for i in xrange(self.m)]) for dv in self.norm_datas]
        self.ext_queries = [(qv + [0.5 for i in xrange(self.m)] + g_ext_norm(qv, self.m)) for qv in self.norm_queries]
        new_len = kdata + 2 * m
        LshTester.__init__(self, self.ext_datas, self.ext_queries, rand_range, num_neighbours)

    # MIPS
    def linear(self, q, metric, max_results):
        """ brute force search by linear scan """
        # print 'MipsLshTester linear:'
        candidates = [(ix, np.dot(q, p)) for ix, p in enumerate(self.origin_datas)]
        temp = sorted(candidates, key=itemgetter(1), reverse=True)[:max_results]
        return temp

    def run(self, type, k_vec = [2], l_vec = [2]):
        validate_metric, compute_metric = np.dot, L2Lsh.distance
        exact_hits = [[ix for ix, dist in self.linear(q, validate_metric, self.num_neighbours)] for q in self.origin_queries]

        print '=============================='
        print 'L2AlshTester ' + type + ' TEST:'
        print 'L\tk\tacc\ttouch'

        # concatenating more hash functions increases selectivity
        for k in k_vec:
            lsh = LshWrapper(type, self.d, self.rand_range, k, 0)

            # using more hash tables increases recall
            for L in l_vec:
                lsh.resize(L)
                lsh.index(self.ext_datas)

                correct = 0
                for q, hits in zip(self.ext_queries, exact_hits):
                    lsh_hits = [ix for ix, dist in lsh.query(q, compute_metric, self.num_neighbours)]
                    correct += int(lsh_hits == hits)

                print "{0}\t{1}\t{2}\t{3}".format(L, k, float(correct) / self.q_num, float(lsh.get_avg_touched()) / len(self.datas))

    @staticmethod
    # type: l2 & cosine
    # mips: True for ALSH
    def createTester(type, datas, queries, rand_num, num_neighbours):
        return L2AlshTester(datas, queries, rand_num, num_neighbours)