
import cPickle as pickle
import numpy as np
import sys
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
ONE-OFF

generate a label-image dictionary
"""
# def gen_labels_dict():
#     output = []
#     d = []
#     for i in range(894):
#         d[i] = [ img for img in range(1449) if np.where(output[img] == i)[0] ]
#
# def top_n_labels_dict():
#     d = dict()
#     d_mod = dict()
#     for i in range(894):
#         to_store = d[i] if len(d[i]) > 1 else 123456
#         if to_store != 123456:
#             d_mod[i] = to_store


"""
get shuffled top_n records

input:
    - data_s start
    - data_e end
    - (optional) path

return:
    - records top n records
"""
def top_n(target, labels_dict, n, path=''):
    records = np.array([])
    imgs = labels_dict[target]

    # load images
    for img in imgs:
        loaded = load_data(path+'ft_co_'+img+'p.', 'rb', path)
        np.append(records, [loaded], axis=0)

    print 'records shape:', records.shape

    return np.random.shuffle(records)[0:n]



"""
create feature-target dicitonary
"""
def create_ft_dict(features, targets):
    return {'features': features, 'targets': targets}



"""
load data from file
"""
def load_data(filename, mode, path=''):
    return pickle.load(open(path+filename, mode))


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
    subparsers = parser.add_subparsers(help='available commands')

    # co AND per_pixel
    p_patches = subparsers.add_parser('patches', help='obtain patches with known labels')
    p_patches.add_argument('-fn', '-function', action='store', dest='fn', help='operation to perform')
    p_patches.add_argument('-img_s', '-img_start', action='store', type=int, dest='img_s', help='image range start')
    p_patches.add_argument('-img_e', '-img_end', action='store', type=int, dest='img_e', help='image range end')
    p_patches.add_argument('-dim', '-dimension', action='store', type=int, dest='dim', help='dimension of a patch')
    p_patches.set_defaults(which='patches')

    # top_n
    p_topn = subparsers.add_parser('top_n', help='get n random samples for a given label')
    p_topn.add_argument('-l', '-label', action='store', dest='target', type=int, help='the label to be explored')
    p_topn.add_argument('-n', action='store', dest='n', type=int, help='the number of random samples required')
    p_topn.set_defaults(which='top_n')

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

    # initialisation
    depths, labels = initialise(path)
    # depths = pickle.load(open('test_depths.p', 'rb'))
    # labels = pickle.load(open('test_labels.p', 'rb'))

    args = parser()
    print args

    if not check_args(args):
        print >> sys.stderr, 'invalid parameters inputted -> use -h to find out the required parameters'
        sys.exit(1)

    # find out which function to perform
    if args.which == 'per_pixel':
        aggr_per_pixel(depths, args.img_s, args.img_e, args.dim, path)
    elif args.which == 'co':
        aggr_given_co(depths, labels, args.img_s, args.img_e, args.dim, path)
    elif args.which == 'top_n':
        labels_dict = load_data('labels_map.p', 'rb', path)
        print top_n(args.l, labels_dict, args.n, path)
    else:
        print >> sys.stderr, 'possible inputs: per_pixel, co, top_n'
        sys.exit(1)


"""
testing point
"""
if __name__ == '__main__':
    main()
