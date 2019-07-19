import numpy as np
import sparsegrad.forward as ad
def f(x):
    return x-x[::-1,::-1]
x=np.zeros((3,3))
print(f(ad.seed(x)).dvalue)

# to recover n-D entries from sparsematrix, use indexing property
x_ = ad.seed(x)
f = f(x_)
print(f.dvalue[f.indexing[0,0], x_.indexing[0,0]])
print(f.dvalue[f.indexing[0,0], x_.indexing[2,2]])
