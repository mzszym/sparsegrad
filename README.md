# sparsegrad - automatic sparse gradients of numpy expressions

## Installation:

python setup.py install

## Testing after installation

import sparsegrad
sparsegrad.test()

## Usage

import sparsegrad as ad
import numpy as np
x=ad.forward.seed(0)
y=np.sin(x)
print(y.value)
print(y.gradient)

See tutorial.