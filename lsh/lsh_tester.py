# -*- coding: utf-8 -*-
import random
import numpy as np
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from operator import itemgetter
import time
from lsh import *
from lsh_wrapper import *
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

class LshTesterFactory():
    @staticmethod
    # type: l2 & cosine
    # mips: True for ALSH
    def createTester(type, mips, datas, queries, rand_num, num_neighbours):
        # if False == mips:
        #     return LshTester(datas, queries, rand_num, num_neighbours)
        if 'l2' == type:
            return L2AlshTester(datas, queries, rand_num, num_neighbours)