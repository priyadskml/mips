from lsh import *
import random
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from operator import itemgetter


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
