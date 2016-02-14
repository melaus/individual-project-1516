
from sklearn import svm
from sklearn import cross_validation
import cPickle as pickle
import numpy as np
import scipy as sp

path = '/Users/melaus/repo/uni/individual-project/data/py-data/'

# load the data in, we all need it
data = pickle.load(open(path+'features_targets.p', 'rb'))


"""
run model
"""
def model_svc(kernel_type):
    if kernel_type == 'linear':
        model = svm.SVC(kernel='linear', C=1, random_state=1)
    elif kernel_type == 'rbf':
        model = svm.SVC(kernel='rbf', C=1, gamma=0.001, random_state=1)
    model.fit(data['features'], data['targets'])

    return model

"""
get cross val score
"""
def cross_val(model):
    scores = cross_validation.cross_val_score(model, data['features'], data['targets'], cv=2)
    return scores.mean()


"""
store output to a file
"""
def store_output(data, filename, path=''):
    pickle.dump(data, open(path+filename, 'wb'))


"""
run
"""
if __name__ == '__main__':
    #store_output(model_svc('linear'), 'svc_linear_6.p', path)
    #store_output(model_svc('rbf'), 'svc_rbf_6.p', path)

    model = svm.SVC(kernel='linear', C=1, random_state=1)
    print 'linear score:', cross_val(model)
    
    #model = svm.SVC(kernel='rbf', C=1, gamma=0.001, random_state=1)
    #print 'rbf score:', cross_val(model)
