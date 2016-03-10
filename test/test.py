
import numpy as np
from sklearn import datasets as d

iris = d.load_iris()
print 'iris feature: ', iris.data.shape
print 'iris target:  ', iris.target.shape
