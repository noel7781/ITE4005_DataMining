#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from matplotlib import pyplot as plt
import sys


# In[2]:


data = []


# In[3]:

input_path = sys.argv[1]
input_number = int(input_path[5:-4])
with open(input_path, "r") as f:
    total = f.readlines()
    for line in total:
        idx, x, y = line.strip('\n').split('\t')
        data.append([idx, x, y])


# In[4]:


data = np.array(data, dtype=np.float32)
X = np.array([i[1] for i in data], dtype=np.float32)
Y = np.array([i[2] for i in data], dtype=np.float32)
total_coord = np.stack((X,Y), axis=-1)
# print(total_coord)
# plt.figure(figsize=[12, 12])
# plt.scatter(X, Y)
# plt.show()


# In[5]:


n_clusters = int(sys.argv[2])
radius = int(sys.argv[3])
minPts = int(sys.argv[4])


# In[6]:


# 모든 라벨에 unvisited 라벨을 표시
visited = np.zeros(data.shape[0], dtype=np.int32) # 0 -> 미확인 1 -> 방문 2-> 노이즈


# In[7]:


def getUnvisitedObject(visited):
    while True:
        ret = np.random.randint(0, visited.shape[0])
        if visited[ret] == 0:
            visited[ret] = 1
            return ret
        
def getDistances(k):
    size = visited.shape[0]
    ret = np.zeros(size, dtype=np.float32)
    distances = np.sum((total_coord-total_coord[k]) ** 2, axis=1)
    distances = distances ** 0.5
    return distances

def getAdjacentIndex(distances):
    return np.where(distances <= radius)[0]
            


# In[8]:


clusters = np.zeros(data.shape[0], dtype=np.int32)
noises = np.zeros(data.shape[0], dtype=bool)
end_condition = np.ones(data.shape[0], dtype=bool)
cluster_idx = 0
while not np.array_equal(end_condition, visited != 0):# and cluster_idx < n_clusters:
    next = getUnvisitedObject(visited)
    distances = getDistances(next)
    adj_idx = getAdjacentIndex(distances)
    mask = distances[adj_idx]
#     print(mask.shape[0]) # adj_idx -> distance가 radius보다 작은 집합, .shape -> 갯수
#     print("next:",next," and adj count:", mask.shape[0])
    if mask.shape[0] >= minPts:
        cluster_idx += 1
        clusters[next] = cluster_idx
        neighbors = list(adj_idx)
        while neighbors:
            idx = neighbors.pop()
            if visited[idx] == 0:
                visited[idx] = 1
                check_distances = getDistances(idx)
                check_adj_idx = getAdjacentIndex(check_distances)
                check_mask = check_distances[check_adj_idx]
                if check_mask.shape[0] >= minPts:
                    for check_idx in check_adj_idx:
                        if visited[check_idx] == 0:
                            neighbors.append(check_idx)
                if clusters[idx] == 0:
                    clusters[idx] = cluster_idx
    else:
        visited[next] = 2


# In[9]:
total_labels = set(clusters)
cluster_size = np.zeros(len(total_labels), dtype=np.int32)
for k in range(len(total_labels)):
    if k == 0:
        continue
    cluster_size[k] = np.where(clusters==k)[0].shape[0]

size = n_clusters
chosen_clusters_idx = np.where(cluster_size > np.sort(cluster_size)[::-1][size])[0]
hash_table = dict()
for index, number in enumerate(chosen_clusters_idx):
    hash_table[number] = index

for idx in range(len(clusters)):
    if clusters[idx] not in chosen_clusters_idx:
        clusters[idx] = -1

unique_labels = set(clusters[clusters >= 0])
colors = [plt.cm.gist_rainbow(each) 
        for each in np.linspace(0, 1, len(unique_labels))]

# In[10]:


plt.figure(figsize=[8, 8])
for cluster_index, col in zip(unique_labels, colors):
    if cluster_index == -1:
        col = [0, 0, 0, 1]
    class_mask = np.where(clusters == cluster_index)
    sorted_idx = sorted(class_mask[0])
    with open("input{}_cluster_{}.txt".format(input_number, hash_table[cluster_index]), "w") as f:
        for index in sorted_idx:
            f.write("{}\n".format(index))
        plt.plot(data[class_mask][:, 1], 
             data[class_mask][:, 2], 
            'o', markerfacecolor=tuple(col), markeredgecolor=tuple(col), 
            markersize=1)
#plt.show()
        
 
