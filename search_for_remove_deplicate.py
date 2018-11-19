from __future__ import print_function
import os
import time
import numpy as np
import pdb
import sys
#sys.path.insert(0,'/workspace/mnt/group/alg-retri/guoqi/workspace/delf/faiss/python')
import faiss
import random

# set some parameters
basecount = 0
scale = 0.0
abs_scale = 0.0
final_result=[]
# get customed parameters
d = int(4096) # dimension : 4096 = feature_lenth
k = int(3)  # search until top k = search how much same images less than len(all_images)

def chose_one_for_query(base_list,base_base):
  #base_list is for idx;base_base is for feature
  #return original idx , query for search ,base for seach not include query,
  # next base list not include query idx
  lenth, feature_lenth = base_base.shape
  rand_idx = random.randint(0, lenth - 1)
  ary_idx = base_list[rand_idx]
  now_query = base_base[rand_idx]
  base_list.remove(ary_idx)
  now_base = np.concatenate((base_base[:rand_idx], base_base[rand_idx + 1:]), axis=0)
  return ary_idx, now_query.reshape(1, feature_lenth), now_base, base_list


def search_and_on(idx,base,query,now_list):
  base = base.astype(np.float32)
  query = query.astype(np.float32)
  # we need only a StandardGpuResources per GPU
  res = faiss.StandardGpuResources()

  flat_config = faiss.GpuIndexFlatConfig()
  flat_config.device = 0

  index = faiss.GpuIndexFlatIP(res, d, flat_config)
  for nnn in range(1):
    index.add(base)
  D, I = index.search(query, k)
  I = I.tolist()[0]
  #to map to ori idx,similar is the indx for similar images map to original idx,left_base
  #is the next base for search,left_list is the next list for chose query
  similar = []
  for i in range(len(I)):
    idx_this = now_list[I[i]]
    similar.append(idx_this)
  similar.append(idx)
  left_list = [x for x in now_list if x not in similar]
  left_base = np.delete(base, I, axis=0)
  return similar, left_base, left_list

def stop_or_on(list,base):
  #the recursion for search
  if (base.shape[0])>=k+1:
    idx_n,query_n,base_n,list_n = chose_one_for_query(list,base)
    similar_idx,left_base,left_list=search_and_on(idx_n,base_n,query_n,list_n)
    final_result.append(similar_idx)
    stop_or_on(left_list,left_base)
  else:
    print(final_result)
    return final_result





if __name__ == '__main__':
    base_original=np.load("base.npy")
    #npy file is the feature npy of all images
    original_list=list(range(base_original.shape[0]))

    stop_or_on(original_list,base_original)

