
import cPickle as pickle
import numpy as np
import sys, getopt
import math as m
import argparse
#import matplotlib.pyplot as plt

"""
get the data required

input:
    - img - image number
"""
def initialise(path='',img=''):
    # load image data
    depth = pickle.load(open(path+'depths'+img+'.p', 'r'))
    label = pickle.load(open(path+'labels'+img+'.p', 'r')) # 640 * 480

    return depth, label


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
    coordinates = [pair for pair in coordinates if ((lim <= pair[0] <= x-1-lim) and (lim <= pair[1] <= y-1-lim))]
    return coordinates


"""
get depths of given coordinates
"""
def get_object_depth(depth_data, co, dim):
    # get the coordinates
    lim = int(m.floor(dim/2))

    output = []

    for pair in co:
        output.append(depth_data[pair[0]-lim:pair[0]+lim+1, pair[1]-lim:pair[1]+lim+1].tolist())

    return output


"""
return the mean value for each patch
"""
def mean_val(patches):
    return [float(np.mean(patch)) for patch in patches]


"""
normalise depth by deducting the mean value 
from each depth value in the area concerned
"""
def get_norm_depth(areas, means):
    return [[[j - mean for j in ls] for ls in area] for area, mean in zip(areas,means)]


"""
create feature-target dicitonary
"""
def create_ft_dict(features, targets):
    return {'features': features, 'targets': targets}



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
    return get_norm_depth(patches, mean_val(patches))


"""
AVAILABLE FUNCTION

obtain and store all patches of the required image
"""
def aggr_per_pixel(depths, img_start, img_end, dim, path):

    # for each image
    for img in range(img_start, img_end+1):
        # print entry_per_pixel(depths[img], dim)
        save_data(entry_per_pixel(depths[img], dim), 'px_'+str(dim)+'_'+str(img)+'.p', path)


"""
ENTRY

generate the patches of some given coordinates
"""
def entry_given_co(depths, labels, lbl, dim):
    co = get_object_coordinates(labels, lbl, dim)
    object_depth = get_object_depth(depths, co, dim)
    norm = get_norm_depth(object_depth, mean_val(object_depth))

    return norm


"""
AVAILABLE FUNCTION

obtain and store the required object patches for the required image
"""
def aggr_given_co(depths, labels, img_start, img_end, dim, path):

    output_patches = []
    output_targets = []

    # do this for each given image
    for img in range(img_start, img_end+1):
        img_depths = depths[img]
        img_labels = labels[img]
        x, y = img_labels.shape
        set_labels = list(set(img_labels.reshape(x*y,))) # the set of labels in the image

        # do this for all the labels
        for lbl in set_labels:
            patches = entry_given_co(img_depths, img_labels, lbl, dim)

            # append only if not empty
            if patches:
                output_patches.extend(patches)
                output_targets.extend([lbl for l in range(len(patches))])

        # store feature-target dictionary and reset outputs
        save_data(create_ft_dict(output_patches, output_targets),'ft_co_'+str(img)+'.p', path)
        # print output_patches
        # print output_targets
        output_patches = []
        output_targets = []


"""
command line argument parser
"""
def parser():
    parser = argparse.ArgumentParser(description='transform some given data into a desired format')
    parser.add_argument('-fn', '-function', action='store', dest='fn', help='operation to perform')
    parser.add_argument('-img_s', '-img_start', action='store', type=int, dest='img_s', help='image range start')
    parser.add_argument('-img_e', '-img_end', action='store', type=int, dest='img_e', help='image range end')
    parser.add_argument('-dim', '-dimension', action='store', type=int, dest='dim', help='dimension of a patch')
    args = parser.parse_args()
    return args


"""
check if there are any none arguments
"""
def check_args(args):
    for key, val in vars(args).iteritems():
        if val is None:
            return False
    return True


"""
ENTRY

main function
"""
def main():
    # path = '/Users/melaus/repo/uni/individual-project/data/'
    path = '/beegfs/scratch/user/i/awll20/data/ip/'

    # per_pixel = entry_per_pixel(img_depths, 15)
    #
    # print 'shape', per_pixel.shape
    # print per_pixel[0]
    #
    # # depth patches for each pixel
    # save_data(per_pixel, 'img_6_per_pixel.p', path)

    # initialisation
    depths, labels = initialise(path)
    # depths = pickle.load(open('test_depths.p', 'rb'))
    # labels = pickle.load(open('test_labels.p', 'rb'))

    args = parser()

    if not check_args(args):
        print >> sys.stderr, 'invalid parameters inputted -> use -h to find out the required parameters'
        sys.exit(1)

    # find out which function to perform
    if args.fn == 'per_pixel':
        aggr_per_pixel(depths, args.img_s, args.img_e, args.dim, path)
        print 'per_pixel'
    elif args.fn == 'co':
        aggr_given_co(depths, labels, args.img_s, args.img_e, args.dim, path)
        # print 'co'
    else:
        print >> sys.stderr, 'possible inputs: per_pixel, co'
        sys.exit(1)


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
