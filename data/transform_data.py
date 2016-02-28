
import cPickle as pickle
import numpy as np
import sys, getopt
import math
#import matplotlib.pyplot as plt

path = '/Users/melaus/repo/uni/individual-project/data/py-data/'

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
generic way to find depths given boundaries and dimension
"""
def get_depths(x_min, x_max, y_min, y_max, dim):
    
    x_lower, x_upper = x_min, x_min+dim
    y_lower, y_upper = y_min, y_min+dim

    tmp = []
    output = []

    for y_groups in range(0, int(math.floor(y_max/dim))):
	for x_groups in range(0, int(math.floor(x_max/dim))):
            # get depths for these points
            tmp = [img_depths[x][y] for y in range(y_lower, y_upper) for x in range(x_lower, x_upper)]
            #tmp = [(x,y) for y in range(y_lower, y_upper) for x in range(x_lower, x_upper)]
	    
	    # increase the y bounds to process the next group
	    x_lower += dim
	    x_upper += dim
	
            output.append(tmp)

	# reset x bounds
	x_lower = x_min
	x_upper = x_min + dim
	
	# increase y bounds to process the next group
	y_lower += dim
	y_upper += dim
        
    return output


"""
find the depths of the required areas with respect to 
the size of an image and the required dimension

uses get_depths

input:
    x   : width
    y   : height
    dim : dimension

return: list of depths 
"""
def gen_area_depths(x, y, dim):
    # check what is the maximum area that the dimension can cover
    excess_x = x % dim
    excess_y = y % dim
    
    # set the min and max value
    x_min = int(math.ceil(excess_x/2-1)) if excess_x > 0 else 0 
    x_max = x - int(math.floor(excess_x/2)) if excess_x > 0 else x 

    y_min = int(math.ceil(excess_y/2-1)) if excess_y > 0 else 0 
    y_max = y - int(math.floor(excess_y/2)) if excess_y > 0 else y

    return get_depths(x_min, x_max, y_min, y_max, dim)

"""
get areas of interest
"""
def get_object_depths(label_data, lbl, dim):
    # find out chair areas
    coordinates = zip(*np.where(label_data == lbl))
    areas       = []
    
    # find the area of which we use to normalise the middle point 
    for pt in coordinates: 
        # get the range of the area concerned
        x_min = pt[0]-int(math.floor(dim/2)) if pt[0]-int(math.floor(dim/2)) >= 0 else 0
        x_max = pt[0]+int(math.ceil(dim/2))  if pt[0]+int(math.ceil(dim/2)) < 640 else 479 

        y_min = pt[1]-int(math.floor(dim/2)) if pt[1]-int(math.floor(dim/2)) >= 0 else 0
        y_max = pt[1]+int(math.ceil(dim/2))  if pt[1]+int(math.ceil(dim/2)) < 480 else 479 
        
        if (x_max-x_min == dim-(dim%2) & y_max-y_min == dim-(dim%2)):
            # append depths to output list
            areas.append([img_depths[x][y] for y in range(y_min, y_max+1) for x in range(x_min, x_max+1)])
    
    return areas 


"""
return the mean depth for each point area
"""
def mean_depth(areas):
    return [np.mean(area) for area in areas]


"""
normalise depth by deducting the mean value 
from each depth value in the area concerned
"""
def get_normalised_depths(areas, means):
    return [[j - b for j in a] for a, b in zip(areas,means)]


"""
create feature-target dicitonary
"""
def create_ft_dict(features, targets):
    return {'features': features, 'targets': targets}

def store_output(data, filename, path=''):
    pickle.dump(data, open(path+filename, 'wb'))
    print 'done'

"""
testing point
"""
if __name__ == '__main__':
    img_images, img_depths, img_labels = initialise()

    ## background 
    #bk_dps        = get_object_depths(img_labels, 0, 15)
    #bk_mean       = mean_depth(bk_dps)
    #bk_normalised = get_normalised_depths(bk_dps, bk_mean)
    #print 'background sorted'
    #print ' - mean length:', len(bk_mean)
    #print ' - dps length: ', len(bk_dps)
    #print ' - dps set len:', set([len(i) for i in bk_dps])
    #print ''

    ## ceiling
    #ceiling_dps        = get_object_depths(img_labels, 4, 15)
    #ceiling_mean       = mean_depth(ceiling_dps)
    #ceiling_normalised = get_normalised_depths(ceiling_dps, ceiling_mean)
    #print 'ceiling sorted'
    #print ' - mean length:', len(ceiling_mean)
    #print ' - dps length: ', len(ceiling_dps)
    #print ' - dps set len:', set([len(i) for i in ceiling_dps])
    #print ''
    
    ## chair
    #chair_dps          = get_object_depths(img_labels, 5, 15)
    #chair_mean         = mean_depth(chair_dps)
    #chair_normalised   = get_normalised_depths(chair_dps, chair_mean)
    #print 'chair sorted'
    #print ' - mean length:', len(ceiling_mean)
    #print ' - dps length: ', len(ceiling_dps)
    #print ' - dps set len:', set([len(i) for i in ceiling_dps])
    #print ''
 
    ## construct targets
    #targets = []
    #targets.extend([0]*len(bk_mean))
    #targets.extend([4]*len(ceiling_mean))
    #targets.extend([5]*len(chair_mean))
    #print 'targets, length', len(targets)

    ## construct features
    #features = []
    #features.extend(bk_normalised)
    #features.extend(ceiling_normalised)
    #features.extend(chair_normalised)
    #print 'features, length', len(features)
    #print ''

    ## store features and targets to file
    #store_output(create_ft_dict(features, targets), 'features_targets.p', path)
    #print 'stored dict'
    #print ''

    # find depths of an image
    #area_dps            = gen_area_depths(640,480,15)
    #area_dps_mean       = mean_depth(area_dps)
    #area_dps_normalised = get_normalised_depths(area_dps, area_dps_mean)
    #print 'area_depths'
    #print ' - length:', len(area_dps)
    #print ' - set length:', set([len(i) for i in area_dps_normalised])
    #print ''
    
    ## store this output 
    #store_output(area_dps_normalised, 'area_depths_6.p', path)
    #print 'stored area_depths'
    #print 'done'

    #print gen_area_depths(6,6,3)
