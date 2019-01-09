# @author : Shashank Iyer

"""Contains functions that compute similarity search precision"""

import numpy as np

# Computes the hamming distance between bitvectors
def hamming_dist(bitvec1, bitvec2):

    return np.count_nonzero(bitvec1!=bitvec2)

# Computes pairwise hamming distance between the query and database bitvectors
def compute_hamming_dist(query, database):
    
    res = np.empty(shape=(len(query), len(database)), dtype=int)
    
    it = np.nditer(res, flags=['multi_index'], op_flags=['writeonly'])
    while not it.finished:
        it[0] = hamming_dist(query[it.multi_index[0]], database[it.multi_index[1]])
        it.iternext()
    
    return res

# Computes pairwise hamming distances and keep only those images that are within a threshold
def coarse_grain_search(threshold, query_emb_bin, database_emb_bin):
    
    hamming = compute_hamming_dist(query_emb_bin, database_emb_bin)
    ind = np.array([np.where( i <= threshold) for i in hamming])

    return ind

# Compute pairwise Euclidean distances between a query and all database images
def euclidean_dist(query, database):
    return np.sqrt(np.sum(np.square(np.expand_dims(query, 0) - database), axis = -1))

# Select k images that are closest to the query
def fine_grain_search(k, euclidean_arr, list_indices):

    euclidean_arr = np.argsort(euclidean_arr)
    list_indices = list_indices[euclidean_arr]
    list_indices = list_indices[:k:1]
  
    return list_indices

# Computes ground truth relevance imagewise.
def reli_image_wise(k, each_query, list_indices, database_emb_float, each_query_lab, database_lab):

    # if no images are within the specified hamming threshold of the query, return 0
    if len(list_indices) == 0:
        return 0
    
    # Select db embeddings for fine-grain search
    db_take = np.take(database_emb_float , list_indices , axis=0)
    
    euc = euclidean_dist( each_query , db_take)
    
    fg_list_indices = fine_grain_search(k, euc, list_indices)
    
    # Select top 'k' db image's labels
    db_lab_k = np.take(database_lab, fg_list_indices)
    
    # Comparing top 'k' db labels with query label
    res = np.equal(np.expand_dims(each_query_lab,0), db_lab_k)
    
    # mAP computation
    den = np.arange(1, res.shape[0] + 1, dtype = int)
    num = np.cumsum(res)
    res = num * res / den
    res = np.mean(res)

    return res


def reli(threshold, k, query_emb_bin, query_emb_float, query_lab, database_emb_bin, database_emb_float, database_lab):
    
    ind = coarse_grain_search(threshold, query_emb_bin, database_emb_bin)
    
    database_lab = np.argmax(database_lab,1)
    query_lab = np.argmax(query_lab,1)
    
    res = np.empty(query_lab.shape[0])
    it = np.nditer(res, flags=['c_index'], op_flags=['writeonly'])
    while not it.finished:
        it[0] = reli_image_wise(k , query_emb_float[it.index], ind[it.index][0], database_emb_float, query_lab[it.index], database_lab)
        it.iternext()
    
    return np.mean(res)

if __name__ == "__main__":
    
    query_emb_bin = np.array(np.random.randint(2,size = (4,5)), dtype = np.float)
    database_emb_bin = np.array(np.random.randint(2,size = (14,5)), dtype = np.float)

    query_emb_float = np.array(np.random.randint(0 , 10  ,size = (4,20)), dtype = np.float)
    database_emb_float = np.array(np.random.randint(0 , 10 ,size = (14,20)), dtype = np.float)
    
    query_lab = np.random.randint(0, 3, size = 4)
    one_hot = np.zeros((4,3))
    one_hot[np.arange(4), query_lab] = 1
    query_lab = one_hot

    database_lab = np.random.randint(0, 3, size = 14)
    one_hot = np.zeros((14,3))
    one_hot[np.arange(14), database_lab] = 1
    database_lab = one_hot

    print(reli(3 ,3, query_emb_bin, query_emb_float, query_lab, database_emb_bin, database_emb_float, database_lab))



