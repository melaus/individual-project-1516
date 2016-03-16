
from sklearn import svm
from sklearn import cross_validation
import cPickle as pickle
import argparse
import numpy as np
import scipy as sp
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.grid_search import GridSearchCV


"""
fit SVC models
"""
def model_svc(kernel_type, features, targets):
    if kernel_type == 'linear':
        model = svm.SVC(kernel='linear', C=1, random_state=1)
    elif kernel_type == 'rbf':
        model = svm.SVC(kernel='rbf', C=1, gamma=0.001, random_state=1)
    model.fit(features, targets)

    return model


"""
get cross val score
"""
def cross_val(model, features, targets):
    scores = cross_validation.cross_val_score(model, features, targets, cv=2)
    return scores.mean()


"""
generate model and data that is used for constructing the confusion matrix 

returns:
    - fitted model
    - test data (dict)
"""
def gen_cf_model(model, features, targets):
    print 'start split'
    train_x, test_x, train_y, test_y = cross_validation.train_test_split(features, targets, test_size = 0.3, random_state=1)
    print 'end split'
    print 'start fit'
    model.fit(train_x, train_y)
    print 'end fit'
    return model, {'features': test_x, 'targets': test_y}


""" confusion matrix """
def get_cf_matrix(actual, predicted):
    return confusion_matrix(actual, predicted)


"""
GridSearch for parameters
"""
def grid_search(params_dict):
    search = GridSearchCV()
    # search.fit()
    return score
    pass

"""
store output to a file
"""
def save_data(data, filename, path=''):
    pickle.dump(data, open(path+filename, 'wb'))


"""
command line argument parser
"""
def parser():
    parser     = argparse.ArgumentParser(description='train models and some associated tasks')
    subparsers = parser.add_subparsers(help='different arguments for different model training and associated tasks')

    # svc
    p_svc = subparsers.add_parser('svc', help='SVC')
    p_svc.add_argument('-k', '-kernel', action='store', dest='kernel', help='choose kernel - rbf, linear')
    p_svc.add_argument('-gamma', action='store', type=int, dest='gamma', help='dimension of a patch')
    p_svc.set_defaults(which='svc')

    # confusion matrix
    p_cf = subparsers.add_parser('cf', help="confusion matrix")
    p_cf.add_argument('-x', '-width', action='store', type=int, dest='x', help='width of image')
    p_cf.add_argument('-y', '-height', action='store', type=int, dest='y', help='height of image')
    p_svc.set_defaults(which='cf')

    return parser.parse_args()


"""
check if there are any none arguments
"""
def check_args(args):
    for key, val in vars(args).iteritems():
        # don't check for optional keys
        if val is None and key not in ('x', 'y'):
            return False
    return True


"""
ENTRY

main function
"""
def main():
    # path = '/Users/melaus/repo/uni/individual-project/data/py-data/'
    # path = '/beegfs/scratch/user/i/awll20/data/ip/'
    pass


"""
run
"""
if __name__ == '__main__':
    """ the data """ 
    path = '/Users/melaus/repo/uni/individual-project/data/py-data/'
    path = '/beegfs/scratch/user/i/awll20/data/ip/'
    #data = pickle.load(open(path+'features_targets.p', 'rb'))


    """ train model """
    #store_output(model_svc('linear', data['features'], data['targets']), 'svc_linear_6.p', path)
    #store_output(model_svc('rbf', data['features'], data['targets']), 'svc_rbf_6.p', path)


    """ cross validation """
    #model = svm.SVC(kernel='linear', C=1, random_state=1)
    #print 'linear score:', cross_val(model, data['features', data['targets'])
    
    #model = svm.SVC(kernel='rbf', C=1, gamma=0.001, random_state=1)
    #print 'rbf score:', cross_val(model, data['features'], data['targets'])


    """ model and test data for confusion matrix """
    # linear 
    #model = svm.SVC(kernel='linear', C=1, random_state=1)
    #trained_model, test_data = gen_cf_model(model, data['features'], data['targets']) 
    #store_output(trained_model, 'linear_cf_model.p', path)
    #store_output(test_data, 'linear_cf_test_data.p', path)
     
    # rbf
    #model = svm.SVC(kernel='rbf', C=1, gamma=0.001, random_state=1)
    #trained_model, test_data = gen_cf_model(model, data['features'], data['targets'])
    #store_output(trained_model, 'rbf_cf_model.p', path)
    #store_output(test_data, 'rbf_cf_test_data.p', path)

    
    """ confusion matrix """
    # load data if requires
    linear_cf_test_data = pickle.load(open(path+'linear_cf_test_data.p', 'rb')) 
    linear_cf_predicted = pickle.load(open(path+'linear_cf_predicted.p', 'rb'))

    rbf_cf_test_data = pickle.load(open(path+'rbf_cf_test_data.p', 'rb')) 
    rbf_cf_predicted = pickle.load(open(path+'rbf_cf_predicted.p', 'rb')) 

    # confusion matrices 
    linear_cf_matrix = get_cf_matrix(linear_cf_test_data['targets'], linear_cf_predicted)
    rbf_cf_matrix = get_cf_matrix(rbf_cf_test_data['targets'], rbf_cf_predicted)
    
    print 'linear_cf_matrix'
    print linear_cf_matrix
    print classification_report(linear_cf_test_data['targets'], linear_cf_predicted)
    print ''

    print 'rbf_cf_matrix'
    print rbf_cf_matrix
    print classification_report(rbf_cf_test_data['targets'], rbf_cf_predicted)
