import numpy as np
from scipy import special as spc
import scipy.spatial.distance as dist

def ALSHDataTransform(data, queries, m):
    data_trans = np.array(map(lambda x: np.append(x, \
        [np.linalg.norm(x)**(2*(k+1)) for k in range(m)]), data))
    queries_trans = np.array(map(lambda x: np.append(x, \
        [.5 for k in range(m)]), queries))
    return (data_trans, queries_trans)

def hashedFunctions(N_Hashs, dim):
    return np.random.normal(size=(N_Hashs, dim))

def dataToHashFunctions(datapts, hash_raw, rad):
    hashed = np.dot(datapts, hash_raw.T)
    val = rad * np.random.random(1)
    return np.floor((hashed + val) / rad)

def hammDist(hash_pt1, hash_pt2):
    hDist = dist.cdist(hash_pt1, hash_pt2, metric='hamming')
    hDist = 1.0 - hDist
    return (hDist, np.argsort(hDist, axis=1))

def convertto1D(Mat2D):
    n = len(Mat2D)
    m = len(Mat2D[0])
    Mat1D = np.zeros(n * m)
    k = 0
    for i in range(n):
        for j in range(m):
            Mat1D[k] = Mat2D[i, j]
            k += 1
    return Mat1D

def distances(datapts, querypts, metric='euclidean'):
    dists = dist.cdist(datapts, querypts, metric='euclidean')
    args_ = np.argsort(dists, axis=1)
    return (dists, args_, convertto1D(dists))

def dot_prod(datapts, queriespts):
    n = len(datapts)
    dotpdt_pts = np.dot(datapts, queriespts.T)
    dotpdt_arg = np.argsort(dotpdt_pts, axis=1)
    return (dotpdt_pts, dotpdt_arg, convertto1D(dotpdt_pts))

def topTindices(hashedvals, groundtruth, t):
    indexs = np.zeros(t, dtype=int)
    for k in range(t):
        indexs[k] = np.where(hashedvals == groundtruth[-1-k])[0][0]
    indexs = np.sort(indexs)
    indexs = len(hashedvals) - indexs[::-1] - 1
    indexs = indexs.astype(int)
    return indexs

def precisionRecall(rankedpts, groundtruth, t):
    t = int(t)
    n = int(len(rankedpts))
    inverse_range = np.array(range(1, n+1))
    inverse_range = 1./inverse_range
    precisionmean = np.zeros(t)
    for q in range(n):
        precision = np.zeros(t)
        indexs = topTindices(rankedpts[q], groundtruth[q], t)
        for k in range(t-1):
            precision[k] = (k+1) * np.mean(inverse_range[int(indexs[k]):int(indexs[k+1])])
        precision[t-1] = t * np.mean(inverse_range[indexs[-1]:])
        precisionmean += precision
    precisionmean /= n
    return precisionmean
