
from sklearn import svm
from sklearn import cross_validation
import cPickle as pickle

path = '/Users/melaus/repo/uni/individual-project/data/'

# load the data in, we all need it
data = pickle.load(open(path+'py-data/features_targets.p', 'rb'))


def modelling():
    model = svm.SVC(kernel='linear', C=1)
    scores = cross_validation.cross_val_score(model, data.features, data.targets, cv=2)
    return scores.mean()

if __name__ == '__main__':
    data.keys()
    #print modelling()

