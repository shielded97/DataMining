# create dataset from canvas set for additional testing
# gonna pickle some things

import numpy as np
import random
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import collections
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys

# read in dataset.txt 
# pickle the dataset into subsets including .x, .tx, .allx as scipy.sparse.csr.csr_matrix
# pickle the dataset into subsets including .y, .ty, .ally as numpy.ndarray
# also:
#   ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict object;
#   ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.
values = np.array([[0,0]])
xvalues = np.array([], dtype=int)
yvalues = np.array([], dtype=int)
zvalues = np.array([], dtype=int)
with open("data/DGAddedCats.txt", 'rb') as f:
    for line in f:
        x,y,z = line.split()
        x = int(x)
        y = int(y)
        z = int(z)
        values = np.append(values,[[x,y]], axis=0)
        xvalues = np.append(xvalues,x)
        yvalues = np.append(yvalues,y)
        zvalues = np.append(zvalues,z)

#delete first empty element
# index = [0]
# values = np.delete(values, 0,0)
# ones = np.ones([2500], dtype=int)

# choose between 150 and 300 indices per paper to make hot using the same random seed
# to allow the GNN to make correlations between papers with similar vocabularies
#   -Arbitrarily choose the vocabulary to be 2500 words
xtot = np.empty((5242,2500))
numHot = random.randint(150, 300)
x_cur = xvalues[10]
i = 0
for x, z in zip(xvalues, zvalues):
    hot = np.zeros((2500))
    random.seed(z)
    if x != x_cur:
        x_cur = x
        for j in range(0, numHot):
            index = random.randint(0, 2499)
            hot[[index]] = 1
        xtot[[i]] = hot
        i = i + 1

x_train = sp.csr_matrix(xtot[0:140])
x_test = sp.csr_matrix(xtot[4242:5242])
x_all = sp.csr_matrix(xtot[0:4242])        

# now get the y values in a numpy array
# first 100 entries are used as labeled training examples
# 4k to 5k are used as labeled testing examples
y_all = np.zeros(shape=(5242,6), dtype=int)
x_cur = xvalues[10]
i = 0
for x, z in zip(xvalues, zvalues):
    if x != x_cur:
        x_cur = x
        y_all[i,z] = 1
        i = i + 1
y_train = y_all[0:140]
y_test = y_all[4242:5242]
y_all = y_all[0:4242]

# now make .graph
# contains an entry for every unique 'document' and each of the 'papers' it cites

# lets us reference each document in the order of appearance in xvalues via indexing
conversion = collections.defaultdict(list)
x_init = xvalues[10]
i = 0
for x in xvalues:
    if x_init != x:
        x_init = x
        conversion[x].append(i)
        i = i + 1

# now create graph with 'conversion'
d = collections.defaultdict(list)
d1 = collections.defaultdict(list)
x_init = xvalues[0]
i = 0
for x, y in zip(xvalues, yvalues):
    if x_init != x:
        x_init = x
        i = i + 1
    d[i].append(conversion[y][0])
    d1[i].append(y)

print(d[0])
print(d[1])

print(d1[0])
print(d1[1])
print('\nDG:')
print('\tlen graphdg: ',len(d))
print('\tx_test size: ', x_test.toarray().shape)
print('\tx_train size: ', x_train.toarray().shape)
print('\ty_test size: ', y_test.shape)
print('\ty_train size: ', y_train.shape)

# now make .index
index = random.sample(range(4242, 5242), 1000)

# pickle some things
with open("data/ind.canvas6.x", 'wb') as f:
    pkl.dump(x_train, f)
    f.close()

with open("data/ind.canvas6.y", 'wb') as f:
    pkl.dump(y_train, f)
    f.close()

with open("data/ind.canvas6.tx", 'wb') as f:
    pkl.dump(x_test, f)
    f.close()

with open("data/ind.canvas6.ty", 'wb') as f:
    pkl.dump(y_test, f)
    f.close()

with open("data/ind.canvas6.allx", 'wb') as f:
    pkl.dump(x_all, f)
    f.close()

with open("data/ind.canvas6.ally", 'wb') as f:
    pkl.dump(y_all, f)
    f.close()

with open("data/ind.canvas6.graph", 'wb') as f:
    pkl.dump(d, f)
    f.close()

with open("data/ind.canvas6.test.index", 'w') as f:
    for num in index:
        f.write(str(num) + '\n')
    f.close()

# load/unpickle cora/citeseer to see what it's like
names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
objects = []
for i in range(len(names)):
    with open("data/ind.{}.{}".format("citeseer", names[i]), 'rb') as f:
        if sys.version_info > (3, 0):
            objects.append(pkl.load(f, encoding='latin1'))
        else:
            objects.append(pkl.load(f))

x, y, tx, ty, allx, ally, graph = tuple(objects)

print('\nCiteseer:')
print('\tsize graph: ', len(graph))
# print(graph)
print('\tsize tx: ', tx.toarray().shape)
print('\tsize x: ', x.toarray().shape)

# print('tx:\n', tx)
#print('ty:\n', ty)

print('\tsize ty: ', ty.shape)
print('\tsize y: ', y.shape)
# print(tx.toarray())

names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
objects = []
for i in range(len(names)):
    with open("data/ind.{}.{}".format("cora", names[i]), 'rb') as f:
        if sys.version_info > (3, 0):
            objects.append(pkl.load(f, encoding='latin1'))
        else:
            objects.append(pkl.load(f))

x, y, tx, ty, allx, ally, graph = tuple(objects)

print('\nCora:')
# print(graph)
print('\tsize graph: ', len(graph))
print('\tsize tx: ', tx.toarray().shape)
print('\tsize x: ', x.toarray().shape)
print('\tsize ty: ', ty.shape)
print('\tsize y: ', y.shape)

names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
objects = []
for i in range(len(names)):
    with open("data/ind.{}.{}".format("pubmed", names[i]), 'rb') as f:
        if sys.version_info > (3, 0):
            objects.append(pkl.load(f, encoding='latin1'))
        else:
            objects.append(pkl.load(f))

x, y, tx, ty, allx, ally, graph = tuple(objects)

print('\nPubMed:')
print('\tsize graph: ', len(graph))
print('\tsize tx: ', tx.toarray().shape)
print('\tsize x: ', x.toarray().shape)
print('\tsize ty: ', ty.shape)
print('\tsize y: ', y.shape)