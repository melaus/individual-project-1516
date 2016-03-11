
from sklearn import svm
from sklearn import cross_validation
import cPickle as pickle
import numpy as np
import scipy as sp
from sklearn.metrics import confusion_matrix, classification_report


"""
run model
"""
def model_svc(kernel_type, features, targets):
    if kernel_type == 'linear':
        model = svm.SVC(kernel='linear', C=1, random_state=1)
    elif kernel_type == 'rbf':
        model = svm.SVC(kernel='rbf', C=1, gamma=0.001, random_state=1)
    model.fit(train_x, train_y)

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
store output to a file
"""
def store_output(data, filename, path=''):
    pickle.dump(data, open(path+filename, 'wb'))


"""
run
"""
if __name__ == '__main__':
    """ the data """ 
    path = '/Users/melaus/repo/uni/individual-project/data/py-data/'
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
