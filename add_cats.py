# add categories to the directed graph dataset from canvas

import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys

values = np.array([[0,0]])
xvalues = np.array([], dtype=int)
yvalues = np.array([], dtype=int)
with open("data/CA-GrQc.txt", 'rb') as f:
    for line in f:
        x,y = line.split()
        x = int(x)
        y = int(y)
        values = np.append(values,[[x,y]], axis=0)
        xvalues = np.append(xvalues,x)
        yvalues = np.append(yvalues,y)

with open("data/DGAddedCats.txt", 'w') as f:
    cat = 0
    x_init = xvalues[0]
    for x, y in zip(xvalues, yvalues):
        if x != x_init:
            if cat >= 5:
                cat = 0
            else:
                cat = cat + 1
            x_init = x
        f.write(str(x) + ' ' + str(y) + ' ' + str(cat) + '\n')