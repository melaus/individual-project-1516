
from sklearn.grid_search import GridSearchCV
# from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.svm import SVC
import numpy as np
import cPickle as p

# iris = load_iris()
# data = iris.data
# labels = iris.target
path = '/beegfs/scratch/user/i/awll20/data/ip/top/'
path = '/Users/melaus/scratch/ip/'
data = p.load(open(path+'rand_500_all.p', 'rb'))



# scaler = StandardScaler()
# data = scaler.fit_transform(data)

#C_range = np.logspace(-2, 10, 13)
#gamma_range = np.logspace(-9, 3, 13)

C_range = np.array([1e-02, 1e+01, 1e+02, 1e+04, 1e+08])
gamma_range = np.array([1e-08, 1e-04, 1e-02, 1e+00, 1e+02])

param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(labels, n_iter=5, test_size=0.2, random_state=42)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv, verbose=4)
grid.fit(data['features'], labels['targets'])

print("The best parameters are %s with a score of %0.2f"
              % (grid.best_params_, grid.best_score_))
