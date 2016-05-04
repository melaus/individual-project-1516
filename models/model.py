
from sklearn import svm
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
import argparse, sys
import numpy as np
import cPickle as p
import re
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.datasets import load_iris
from time import time
from sklearn.externals import joblib as j


"""
make iris into dict
"""
def test_data():
    iris = load_iris()
    return {'features': iris.data, 'targets': iris.target}


"""
perform GridSearch
"""
def gridsearch(model, param_grid, data, filename, mode):
    if mode == 'gs':
        grid = GridSearchCV(model, param_grid=param_grid, cv=2, verbose=5, n_jobs=16)
    elif mode == 'rs':
        grid = RandomizedSearchCV(model, param_distributions=param_grid, cv=3, verbose=6, n_iter=1, n_jobs=50)

    grid = grid.fit(data['features'], data['targets'])

    print('\n\n\n'+filename)
    print('best params: ', grid.best_params_)
    print('best score:  ', grid.best_score_)

    return grid.best_params_


"""
fit SVC models
"""
def model_svc(filename, path, id, kernel): # , C=1, gamma=0.001):
    print '--- in model_svc ---\n'
    data = np.load(path+'lbl/'+filename).tolist()
    # data = test_data()

    print 'shape of data:', data['features'].shape

    model = svm.SVC(kernel=kernel, C=0.1) #, C=C, gamma=gamma, random_state=42)

    print 'to fit model'
    t0 = time()
    model.fit(data['features'], data['targets'])
    print 'time taken:', time() - t0

    print('')
    print(model)
    print('')

    j.dump(model, path+'model/'+id+'.jl')
    print('saved joblib')


"""
fit random forest models
"""
def model_rf(filename, path, id):
    print '--- in model_rf ---\n'
    data = np.load(path+'lbl/'+filename).tolist()
    # data = test_data()

    print 'shape of data:', data['features'].shape
    
    # model = RandomForestClassifier(n_estimators=40, max_depth=40, min_samples_split=2, class_weight='balanced', n_jobs=16, verbose=5) 
    model = RandomForestClassifier(min_samples_split=2, class_weight='balanced', n_jobs=16, verbose=5) 

    print 'to fit model'
    t0 = time()
    model = model.fit(data['features'], data['targets'])
    print 'time taken:', time() - t0

    print('')
    print(model)
    print('')

    j.dump(model, path+'model/'+id+'.jl')
    print('saved joblib')
    # save_data(model, 'rf_mx'+str(max_depth)+id+'.npy', path+'model/')


"""
fit AdaBoost models
"""
def model_ada(filename, path, id, alg):
    print '--- in model_ada ---\n'
    data = np.load(path+'lbl/'+filename).tolist()
    # data = test_data()

    print 'shape of data:', data['features'].shape
    
    if alg == 'ada':
        model = AdaBoostClassifier(algorithm='SAMME') 
    elif alg == 'ada-r':
        model = AdaBoostClassifier(algorithm='SAMME.R')

    print 'to fit model'
    t0 = time()
    model = model.fit(data['features'], data['targets'])
    print 'time taken:', time() - t0

    print('')
    print(model)
    print('')

    j.dump(model, path+'model/'+id+'.jl')
    print('saved joblib')
    # save_data(model, 'rf_mx'+str(max_depth)+id+'.npy', path+'model/')


"""
get cross val score
"""
def cross_val(model, features, targets, cv):
    scores = cross_validation.cross_val_score(model, features, targets, cv=cv)
    return scores.mean()


"""
generate model and data that is used for constructing the confusion matrix 

returns:
    - fitted model
    - test data (dict)
"""
def gen_cf_model(model, features, targets):
    print 'start split'
    train_x, test_x, train_y, test_y = cross_validation.train_test_split(features, targets, test_size = 0.3, random_state=42)
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
def save_data(data, filename, path=''):
    np.save(path+filename, data)
    print 'data saved'


"""
open file
"""
def load_data(filename, path=''):
    return np.load(path+filename)


"""
command line argument parser
"""
def parser():
    parser     = argparse.ArgumentParser(description='train models and some associated tasks')
    subparsers = parser.add_subparsers(help='different arguments for different model training and associated tasks')

    # svc
    p_svc = subparsers.add_parser('svc', help='SVC')
    p_svc.add_argument('-file', '-filename', action='store', dest='filename', help='filename')
    p_svc.add_argument('-id', action='store', dest='id', help='storage name')
    p_svc.add_argument('-k', '-kernel', action='store', dest='kernel', help='choose kernel - rbf, linear')
    p_svc.set_defaults(which='svc')

    # rf
    p_rf = subparsers.add_parser('rf', help='Random Forest')
    p_rf.add_argument('-file', '-filename', action='store', dest='filename', help='filename')
    p_rf.add_argument('-id', action='store', dest='id', help='storage name')
    p_rf.set_defaults(which='rf')

    # ada
    p_ada = subparsers.add_parser('ada', help='AdaBoost')
    p_ada.add_argument('-file', '-filename', action='store', dest='filename', help='filename')
    p_ada.add_argument('-id', action='store', dest='id', help='storage name')
    p_ada.add_argument('-alg', '-algorithm', action='store', dest='alg', help='algorithm to be used (SAMME, SAMME.R)')
    p_ada.set_defaults(which='ada')

    # confusion matrix
    p_cf = subparsers.add_parser('cf', help='confusion matrix')
    p_cf.add_argument('-x', '-width', action='store', type=int, dest='x', help='width of image')
    p_cf.add_argument('-y', '-height', action='store', type=int, dest='y', help='height of image')
    p_cf.set_defaults(which='cf')

    # TODO: gridsearch
    p_gs = subparsers.add_parser('gridsearch', help='perform GridSearch with given input')
    p_gs.add_argument('-f', '-file', action='store', dest='filename', help='file to be loaded')
    p_gs.add_argument('-mdl', '-model', action='store', dest='model', help='model used (e.g. svc-rbf, rf)')
    p_gs.add_argument('-m', '-mode', action='store', dest='mode', help='gridsaerch or randomised search')
    p_gs.set_defaults(which='gridsearch')

    return parser.parse_args()


"""
check if there are any none arguments
"""
def check_args(args):
    for key, val in vars(args).iteritems():
        # don't check for optional keys
        if val is None:
            return False
    return True


"""
ENTRY

main function
"""
def main():
    path = '/beegfs/scratch/user/i/awll20/data/ip/'

    args = parser()
    print 'args:', args

    if not check_args(args):
        print >> sys.stderr, 'invalid parameters inputted -> use -h to find out the required parameters'
        sys.exit(1)

    # find out which function to perform
    if args.which == 'svc':
        model_svc(args.filename, path, args.id, args.kernel)

    elif args.which == 'rf':
        model_rf(args.filename, path, args.id)

    elif args.which == 'ada':
        model_ada(args.filename, path, args.id, args.alg)

    elif args.which == 'cf':
        pass

    elif args.which == 'gridsearch':
        data  = load_data(args.filename, path+'lbl/').tolist()

        if 'svc' in args.model:
            if args.model == 'svc-linear':
                param_grid = {'C': np.logspace(-7, -1, 6)}
                model = OneVsRestClassifier(BaggingClassifier(svm.SVC(kernel='linear')), n_jobs=30)
            elif args.model == 'svc-rbf':
                param_grid = {'C': np.logspace(-1, 3, 3), 'gamma': np.logspace(-9, 0, 3)}
                model = svm.SVC(kernel='rbf')

        elif args.model == 'rf':
            # param_grid = {'n_estimators': [50, 100, 500, 1000], 'max_depth': [None, 10, 100, 500, 1000], 'min_samples_split': [2, 5, 10], 'class_weight': [None, 'balanced']}
            # param_grid = {'min_samples_split': [2], 'class_weight': [None], 'n_estimators':[40], 'max_depth':[40]} # model 1 
            # param_grid = {'min_samples_split': [2], 'class_weight': ['balanced'], 'n_estimators':[50], 'max_depth':[40]} # model 2 
            # param_grid = {'min_samples_split': [2], 'class_weight': ['balanced'], 'n_estimators':[45], 'max_depth':[45]} # model 3 
            # param_grid = {'min_samples_split': [2], 'class_weight': ['balanced'], 'n_estimators':[60], 'max_depth':[60]} # model 4 
            param_grid = {'min_samples_split': [2], 'class_weight': ['balanced'], 'n_estimators':[100], 'max_depth':[100]} # model 5 
            model = RandomForestClassifier(n_jobs=50, verbose=6)

        elif 'ada' in args.model:
            # base_estimator = dt_stump, learning_rate = learning_rate, n_estimators = n_estimators,
            param_grid = {'learning_rate': [1,0.5,0.2,0.1], 'n_estimators': [30, 50, 100, 500, 1000]}
            if args.model == 'ada-r':
                model = AdaBoostClassifier(algorithm='SAMME.R')
            else:
                model = AdaBoostClassifier(algorithm='SAMME')

        else:
            print('invalid model')

        print(model)
        # param_grid = dict()
        gridsearch(model, param_grid, data, args.filename, args.mode)


    else:
        print >> sys.stderr, 'possible inputs: svc, rf, gridsearch'
        sys.exit(1)


"""
run
"""
if __name__ == '__main__':
    main()
    """ the data """ 
    # path = '/Users/melaus/repo/uni/individual-project/data/py-data/'
    # path = '/beegfs/scratch/user/i/awll20/data/ip/'
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
    # linear_cf_test_data = p.load(open(path+'linear_cf_test_data.p', 'rb'))
    # linear_cf_predicted = p.load(open(path+'linear_cf_predicted.p', 'rb'))
    #
    # rbf_cf_test_data = p.load(open(path+'rbf_cf_test_data.p', 'rb'))
    # rbf_cf_predicted = p.load(open(path+'rbf_cf_predicted.p', 'rb'))
    #
    # # confusion matrices
    # linear_cf_matrix = get_cf_matrix(linear_cf_test_data['targets'], linear_cf_predicted)
    # rbf_cf_matrix = get_cf_matrix(rbf_cf_test_data['targets'], rbf_cf_predicted)
    #
    # print 'linear_cf_matrix'
    # print linear_cf_matrix
    # print classification_report(linear_cf_test_data['targets'], linear_cf_predicted)
    # print ''
    #
    # print 'rbf_cf_matrix'
    # print rbf_cf_matrix
    # print classification_report(rbf_cf_test_data['targets'], rbf_cf_predicted)
