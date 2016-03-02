
import cPickle as pickle
import numpy as np
import sys, getopt
import math as m
#import matplotlib.pyplot as plt

"""
get the data required

input:
    - img - image number
"""
def initialise(img='',path=''):
    # load image data
    image = pickle.load(open(path+'images'+img+'.p', 'r'))
    depth = pickle.load(open(path+'depths'+img+'.p', 'r'))
    label = pickle.load(open(path+'labels'+img+'.p', 'r')) # 640 * 480

    return image, depth, label


"""
get a depth patch for each pixel
"""
def get_patch_depth(depths, dim):
    x,y = depths.shape
    lim = int(m.floor(dim/2))

    output = []

    print 'lim - ', lim, ' x-1 - ', x-1, ' y-1 - ', y-1

    for row in range(lim, x-lim):
        for col in range(lim, y-lim):

            #current image matrix concerned
            output.append(depths[row-lim:row+lim+1, col-lim:col+lim+1])
    
    return output


"""get coordinates of a given object type


input:
    - (np.array) label_data: labels of where objects are
    - (int) lbl: the label concerned
    - (int) dim: dimension of patch

output:
    - (array) coordinates: a list of coordinates for which their patches has to be found
"""
def get_object_coordinates(label_data, lbl, dim):
    lim = int(m.floor(dim/2))
    x,y = label_data.shape
    coordinates = zip(*np.where(label_data == lbl))

    # get all the pairs that can form a patch without padding
    coordinates = [pair for pair in coordinates if ((lim <= pair[0] <= x-1-lim) and (lim <= pair[1] >= lim <= y-1-lim))]

    return coordinates


"""
get depths of given coordinates
"""
def get_object_depth(depth_data, co, dim):
    # get the coordinates
    lim = int(m.floor(dim/2))

    output = []

    for pair in co:
        output.append(depth_data[pair[0]-lim:pair[0]+lim+1, pair[1]-lim:pair[1]+lim+1])

    return output


"""
return the mean value for each patch
"""
def mean_val(patches):
    return [np.mean(patch) for patch in patches]


"""
normalise depth by deducting the mean value 
from each depth value in the area concerned
"""
def get_norm_depth(areas, means):
    return [[j - b for j in a] for a, b in zip(areas,means)]


"""
create feature-target dicitonary
"""
def create_ft_dict(features, targets):
    return {'features': features, 'targets': targets}


"""
obtain the patches of normalised depth of a given list of labels 

input:
    - lbl_dict (dict): a dictionary containing the labels concerned
"""
def find_feature_patch_depth(lbl_dict):
    pass

"""
get depth patch for each pixel in image

--depths
pad_depth
get_patch_depth
"""

"""
for each object

get_object_depth
mean_depth
get_norm_depth
"""


"""
save data to file

input:
    - (any) data: data to be saved
    - (string) filename: the name of the file to be saved
    - (string) path: store it outside of 'root'
"""
def save_data(data, filename, path=''):
    pickle.dump(data, open(path+filename, 'wb'))

    print 'data saved'


"""
ENTRY

generate depth patches for every pixel of the image
"""
def entry_per_pixel(depths, dim):
    patches = get_patch_depth(depths, dim)
    return np.array(get_norm_depth(patches, mean_val(patches)))


"""
ENTRY

generate the patches of some given coordinates
"""
def entry_given_co(depths, labels, lbl, dim):
    co = get_object_coordinates(labels, lbl, dim)
    object_depth = get_object_depth(depths, co, dim)
    return object_depth


"""
ENTRY

main function
"""
def main():
    path = '/Users/melaus/repo/uni/individual-project/data/py-data/'
    img_images, img_depths, img_labels = initialise('6')

    per_pixel = entry_per_pixel(img_depths, 15)

    print 'shape', per_pixel.shape
    print per_pixel[0]

    # depth patches for each pixel
    save_data(per_pixel, 'img_6_per_pixel.p', path)


"""
testing point
"""
if __name__ == '__main__':
    main()

    ## background 
    #bk_dps        = get_object_depth(img_labels, 0, 15)
    #bk_mean       = mean_depth(bk_dps)
    #bk_normalised = get_norm_depth(bk_dps, bk_mean)
    #print 'background sorted'
    #print ' - mean length:', len(bk_mean)
    #print ' - dps length: ', len(bk_dps)
    #print ' - dps set len:', set([len(i) for i in bk_dps])
    #print ''

    ## ceiling
    #ceiling_dps        = get_object_depth(img_labels, 4, 15)
    #ceiling_mean       = mean_depth(ceiling_dps)
    #ceiling_normalised = get_norm_depth(ceiling_dps, ceiling_mean)
    #print 'ceiling sorted'
    #print ' - mean length:', len(ceiling_mean)
    #print ' - dps length: ', len(ceiling_dps)
    #print ' - dps set len:', set([len(i) for i in ceiling_dps])
    #print ''
    
    ## chair
    #chair_dps          = get_object_depth(img_labels, 5, 15)
    #chair_mean         = mean_depth(chair_dps)
    #chair_normalised   = get_norm_depth(chair_dps, chair_mean)
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
    #area_dps_normalised = get_norm_depth(area_dps, area_dps_mean)
    #print 'area_depths'
    #print ' - length:', len(area_dps)
    #print ' - set length:', set([len(i) for i in area_dps_normalised])
    #print ''
    
    ## store this output 
    #store_output(area_dps_normalised, 'area_depths_6.p', path)
    #print 'stored area_depths'
    #print 'done'

    #print gen_area_depths(6,6,3)
