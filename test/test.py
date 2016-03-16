
# import numpy as np
# from sklearn import datasets as d

# iris = d.load_iris()
# print 'iris feature: ', iris.data.shape
# print 'iris target:  ', iris.target.shape

import os
import ipyparallel as ipp

rc = ipp.Client()
ar = rc[:].apply_async(os.getpid)
pid_map = ar.get_dict()

print pid_map
