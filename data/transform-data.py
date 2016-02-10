#!/Users/melaus/virtualenv/env_python2/bin/python

import cPickle as pickle
import numpy as np
import sys, getopt
#import matplotlib.pyplot as plt

"""
get the data required
"""
def initialise():
    # load image data
    image = pickle.load(open('images_6.p', 'r'))
    depth = pickle.load(open('depths_6.p', 'r'))
    label = pickle.load(open('labels_6.p', 'r')) # 640 * 480

    return image, depth, label


"""
"""
def get_object_points(label_data, lbl):
    # find out chair areas
    coordinates = zip(*np.where(label_data == lbl))
    areas = []
    
    # find the area of which we use to normalise the middle point 
    for pt in coordinates: 
        # get the range of the area concerned
        x_min = pt[0]-7 if pt[0]-7 >= 0 else 0
        x_max = pt[0]+8 if pt[0]+8 <= 640 else 640

        y_min = pt[1]-7 if pt[1]-7 >= 0 else 0
        y_max = pt[1]+8 if pt[1]+8 <= 480 else 480 
        
        areas.append(zip(range(x_min, x_max)*15, sorted(range(y_min, y_max)*15)))
    
    return areas 


"""
get the actual depth values for the points concerned
"""
def get_depth_values(areas):
    # construct a list in the form of areas produced by object_area()
    return [[depth[pt[0]][pt[1]] for pt in area] for area in areas]


"""
return the mean depth for each point area
"""
def mean_depth(areas):
    return [np.mean(area) for area in areas]


"""
normalise depth
"""
def get_normalised_depth(areas, means):
    # means[areas.index(area)] - corresponding mean for that depth area
    # for each point in the area, minus the mean from each of its depth
    #out = []
    #for area in areas:
        #for val in area:
            #out.append(val-means[areas.index(area)])
    #return [[val-means[areas.index(area)] for val in area] for area in areas] 
    return [[j - b for j in a] for a, b in zip(areas,means)]


"""
store features and target dicitonary to a file for reuse
"""
def store_output_dict(features, targets):
    output = {'features': features, 'targets': targets}
    pickle.dump(output, open('py-data/features_targets.p', 'wb'))
    print 'done'


"""
testing point
"""
if __name__ == '__main__':
    image, depth, label = initialise()

    print list(set(label.flatten().tolist()))

    #plt.imshow(image[:][:][:][0])
    #plt.show()

    # areas that requires to be examined as features
    bk_pts          = get_object_points(label, 0)
    bk_dps          = get_depth_values(bk_pts)
    bk_mean         = mean_depth(bk_dps)
    bk_normalised   = get_normalised_depth(bk_dps, bk_mean)
    print 'background sorted, length', len(bk_mean)

    ceiling_pts          = get_object_points(label, 4)
    ceiling_dps          = get_depth_values(ceiling_pts)
    ceiling_mean         = mean_depth(ceiling_dps)
    ceiling_normalised   = get_normalised_depth(ceiling_dps, ceiling_mean)
    print 'ceiling sorted, length', len(ceiling_mean)
    
    chair_pts          = get_object_points(label, 5)
    chair_dps          = get_depth_values(chair_pts)
    chair_mean         = mean_depth(chair_dps)
    chair_normalised   = get_normalised_depth(chair_dps, chair_mean)
    print 'chairs sorted, length', len(chair_mean)
 
    targets = []
    targets.extend([0]*len(bk_mean))
    targets.extend([4]*len(ceiling_mean))
    targets.extend([5]*len(chair_mean))
     
    print 'targets, length', len(targets)

    features = []
    features.extend(bk_normalised)
    features.extend(ceiling_normalised)
    features.extend(chair_normalised)

    print 'features, length', len(features)
    
    store_output_dict(features,targets)
