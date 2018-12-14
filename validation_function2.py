import numpy as np
#import time
def hamming_dist(bitvec1, bitvec2):

    return np.count_nonzero(bitvec1!=bitvec2)

def compute_hamming_dist(query, database):
    
    res = np.empty(shape=(len(query), len(database)), dtype=int)
    
    it = np.nditer(res, flags=['multi_index'], op_flags=['writeonly'])
    while not it.finished:
        it[0] = hamming_dist(query[it.multi_index[0]], database[it.multi_index[1]])
        it.iternext()
    
    return res

def top_k(k, euclidean_arr, database_lab):

    euclidean_arr = np.argsort(euclidean_arr)
    euclidean_arr = euclidean_arr[:k:1]
    
    return np.take(database_lab, euclidean_arr)

def euclidean_dist(query, database):
    return np.sqrt(np.sum(np.square(np.expand_dims(query, 0) - database), axis = -1))

def reli_image_wise(k, each_query, list_indices, database_emb_float, each_query_lab, database_lab):

    #db_take = np.take(database_emb_float , list_indices , axis=0)

    if len(list_indices) == 0:
        return 0
    #euc = euclidean_dist( each_query , db_take)
    #if len(euc) == 0:
        #return 0
    db_lab_take = np.take(database_lab, list_indices, axis = 0)
    db_lab_take = db_lab_take[:k:1]
    #db_lab_k = top_k(k, euc, db_lab_take)
    
    res = np.equal(np.expand_dims(each_query_lab,0), db_lab_take)
    
    res = np.mean(res)

    return res
    

def reli(threshold, k, query_emb_bin, query_emb_float, query_lab, database_emb_bin, database_emb_float, database_lab):
    
    hamming = compute_hamming_dist(query_emb_bin, database_emb_bin)
    
    ind = np.array([np.where( i <= threshold) for i in hamming])
    
    database_lab = np.argmax(database_lab,1)
    query_lab = np.argmax(query_lab,1)
    
    res = np.empty(query_lab.shape[0])
    it = np.nditer(res, flags=['c_index'], op_flags=['writeonly'])
    while not it.finished:
        it[0] = reli_image_wise(k , query_emb_float[it.index], ind[it.index][0], database_emb_float, query_lab[it.index], database_lab)
        it.iternext()
    
    return np.mean(res) #, np.median(res),np.std(res), count

if __name__ == "__main__":
    weights_dict = np.load('model_data/trained_weights/data.npy', encoding='bytes').item()
    '''
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
    '''
    #start = time.time()
    print(reli(12 ,100, weights_dict['val_emb'], weights_dict['val_embf'], weights_dict['val_lab'], weights_dict['database_emb'], weights_dict['database_embf'], weights_dict['database_lab']))
    #print(time.time()-start)



