from collections import defaultdict

import numpy as np
from sklearn.cluster import DBSCAN

from aabb import AABB
from iou import compute_dist_mat


def cluster_aabbs(aabbs):
    """cluster aabbs using DBSCAN and the Jaccard distance between bounding boxes"""
    if len(aabbs) < 2:
        return aabbs

    dists = compute_dist_mat(aabbs)
    clustering = DBSCAN(eps=0.7, min_samples=3, metric='precomputed').fit(dists)

    clusters = defaultdict(list)
    for i, c in enumerate(clustering.labels_):
        if c == -1:
            continue
        clusters[c].append(aabbs[i])

    res_aabbs = []
    for curr_cluster in clusters.values():
        xmin = np.median([aabb.xmin for aabb in curr_cluster])
        xmax = np.median([aabb.xmax for aabb in curr_cluster])
        ymin = np.median([aabb.ymin for aabb in curr_cluster])
        ymax = np.median([aabb.ymax for aabb in curr_cluster])
        res_aabbs.append(AABB(xmin, xmax, ymin, ymax))

    res_aabbs = cluster_aabbs_into_lines(res_aabbs) #cluster lines

    return res_aabbs

def cluster_aabbs_into_lines(aabbs):
    """cluster aabbs using DBSCAN and the Jaccard distance between bounding boxes"""
    if len(aabbs) < 2:
        return aabbs

    X = np.array([[aabb.ymax, aabb.ymin] for aabb in aabbs])
    clustering = DBSCAN(eps=10, min_samples=3).fit(X)

    clusters = defaultdict(list)
    for i, c in enumerate(clustering.labels_):
        if c == -1:
            continue
        clusters[c].append(aabbs[i])

    res_aabbs = []
    for curr_cluster in clusters.values():
        xmin = np.min([aabb.xmin for aabb in curr_cluster])
        xmax = np.max([aabb.xmax for aabb in curr_cluster])
        ymin = np.min([aabb.ymin for aabb in curr_cluster])
        ymax = np.max([aabb.ymax for aabb in curr_cluster])
        res_aabbs.append(AABB(xmin, xmax, ymin, ymax))

    return res_aabbs