
import cPickle as pickle
import numpy as np
import sys
import math as m
import argparse
import glob
from math import ceil
from itertools import groupby
from operator import itemgetter
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
generate ignore list
"""

# for i in range(0, 1449):
#     imgs2labels[i] = np.array(list(set(labels[i].reshape(640*480))))

# for label in range(0,9):
#     labels2imgs[label] =  np.array([ img for img in range(0,3) if len(np.where(imgs2labels[img] == label)[0]) > 0 ])


# def ignore_images():
#     out = np.array([])
#     out = np.append(out, [169,57, 311,1345,349,341,342,350,351,362,363])
#     out = np.append(out, range(83,118+1))
#     out = np.append(out, range(335,337+1))
#     out = np.append(out, range(238,248+1))
#     out = np.append(out, [136,1295,270])
#     out = np.append(out, [407,408,409,440,441,442,466,467,645,646,278])

# out = dict()
# for lbl in labels2imgs:
    # out[lbl] = np.array([x for x in labels2imgs[lbl] if x not in to_ignore])

# p.dump(out, open('labels2imgs_ignore.p', 'wb'))


"""
generate shuffled top_n records from co_* files

input:
    - data_s start
    - data_e end
    - (optional) path

return:
    - records top n records
"""
def top_n(label_s, label_e, labels_dict, num_images, num_samples, path=''):
    for label in range(label_s, label_e+1):
        print '----- DEALING WITH LABEL', label, '-----'
        # change [num_images] if there are not enough images for that label
        extract_imgs = num_images if ceil(len(labels_dict[label])/2) >= num_images else ceil(len(labels_dict[label])/2)
        print 'extract_images: ', extract_imgs 

        # go no further if 0 images are available for extraction
        if extract_imgs == 0:
            continue

        smp_per_img = num_samples / extract_imgs
        print 'sample per image:', smp_per_img 
        pos_dict = dict()

        # get [extract_imgs] images to extract features from
        imgs = np.random.choice(labels_dict[label], extract_imgs, replace=False)
        print 'images to use:', imgs
        print '\n'

        collected = np.array([])

        # extract features and pick an equal amount of random samples from each image
        # to form [num_samples] samples for each label
        for img in imgs:
            print 'in', img
            # get image and randomised position
            data = load_data('co/ft_co_'+str(img)+'.p', 'rb', path)
            print 'loaded data'
            print 'size of targets with this label:', np.where(data['targets'] == label)[0].shape
            
            
            potentials = np.where(data['targets'] == label)[0]
            extract_smps = smp_per_img if smp_per_img <= len(potentials) else len(potentials)
            collected = np.append(collected, extract_smps)

            pos = np.random.choice( potentials, extract_smps, replace=False )
            print 'size of random pos chosen:', len(pos)

            # relate each location extracted to the image it is from
            pos_dict[img] = pos

            # obtain data and appended to the aggregation
            features = np.array([data['features'][po] for po in pos])
            if img == imgs[0]:
                print 'first image'
                data_aggr = features 
            else:
                print 'others'
                data_aggr = np.append(data_aggr, features, axis=0)
            print 'data_aggr shape at this point:', data_aggr.shape
            print ''

        col_min = min(collected)
        col_max = max(collected)
        col_sum = sum(collected)

        output_dict = {'features': data_aggr, 'images' : imgs, 'positions' : pos_dict}
        save_data(output_dict, 'top/top_'+str(label)+'_'+str(int(extract_imgs))+'_'+str(int(col_sum))+'.p', path)
        print 'top_n for label', label, 'saved'
        print ''
        print '----shape of data----'
        print 'size(features)      :', data_aggr.shape
        print 'len(positions_dict) :',len(pos_dict)
        print 'size(images)        :', imgs.shape 
        print 'min(collected):', col_min
        print 'max(collected):', col_max 
        print 'sum(collected):', col_sum
        print '----shape of data----'
        print '\n\n\n#=============================================================\n'
    # return {'features': data_aggr, 'images':imgs, 'positions':pos_dict}


"""
get [n] random records as a combined dict
"""
def n_random_records(max_records, label_s, label_e, path=''):
    features = np.array([])
    targets = np.array([], int)
    out_pos = []

    for lbl in range(label_s, label_e+1):

        # only run loop if there exist such file
        filename = glob.glob(path+'top_'+str(lbl)+'_*')
        if len(glob.glob(path+'top_'+str(lbl)+'_*')) > 0:
            filename = filename[0]
        else:
            continue

        print 'load data from', filename
        data = load_data(filename, 'rb')

        # number of records to get for this label
        num_records = max_records if len(data['features']) >= max_records else len(data['features'])
        print 'records in dict:', len(data['features']), 'wanted:', max_records, 'gets:', num_records

        pos = []
        # create pairs of positions
        # should use 'images' as the order, as dictionary uses its own random order
        for img in data['images']:
            pos.extend(zip( [img for i in range( len(data['positions'][img]) )], data['positions'][img]) )

        print 'size of pos:', len(pos)

        # find random choice of positions
        random_list = np.array( [i for i in range(len(pos))] )
        print 'random_list size:', len(random_list)
        random_list = sorted(np.random.choice( random_list, num_records, replace=False ))
        out_pos.extend([pt for pt in [pos[i] for i in random_list]])
        print 'random_list:', random_list

        # add randomised features to output list
        random_fts = np.array( [data['features'][loc] for loc in random_list] )
        num, x, y = random_fts.shape
        if len(features) == 0:
            features = random_fts.reshape(num, x*y)
        else:
            features = np.append(features, random_fts.reshape(num,x*y), axis=0)

        # targets for the obtained positions
        targets = np.append(targets, [lbl for i in range(num_records)])
    
    print 'len(out_pos):', len(out_pos)
    print '\n'
    out = dict(dict_creator(out_pos))

    save_data({'features': features, 'targets':targets, 'positions': out}, 'test_output.p', path)
    print 'features.shape:', features.shape
    print 'targets.shape: ', targets.shape
    print 'DONE'



"""
create dict given a list of tuples
"""
def dict_creator(tups):
    tups = sorted(tups, key=itemgetter(0))
    it = groupby(tups, itemgetter(0))
    for key, vals in it:
        out = np.array([val[1] for val in vals])
        print 'key:', key
        yield key, out 


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
        save_data(np.array(entry_per_pixel(depths[img], dim)), 'px_'+str(dim)+'_'+str(img)+'.p', path)


"""
obtain patches for each label
"""
def patches_per_label(label_s, label_e, labels, labels2imgs_i, dim, path):
    lim = int(m.floor(dim/2))
    x, y = 640, 480

    out = np.array([])

    # obtain patches for all images that has labels
    for lbl in range(label_s, label_e+1):
        for img in labels2imgs_i[lbl]:
            # get the positions of the required features
            print '[lim:x-lim]:', (lim, x-lim), 'lim:y-lim', (lim, y-lim)
            img_labels = labels[img, lim:x-lim, lim:y-lim].reshape((x-lim*2)*(y-lim*2))
            pos = sorted(np.where(img_labels == lbl)[0])
            print '\n',pos, '\n'

            # open the required patch file
            px = load_data('px_'+str(dim)+'_'+str(img)+'.p', 'rb', path+'px/')

            # add to output array
            if len(out) == 0 :
                out = px[pos,:,:]
            else:
                out = np.append(out, px[pos,:,:], axis=0)

        print 'shape:', out.shape

        save_data(out, 'per_lbl_'+str(lbl)+'.p', path+'lbl/')
        out = np.array([])





# """
# ENTRY
#
# generate the patches of some given coordinates
# """
# def entry_given_co(depths, labels, lbl, dim):
#     co = get_object_coordinates(labels, lbl, dim)
#     object_depth = get_object_depth(depths, co, dim)
#     norm = get_norm_depth(object_depth, mean_val(object_depth))
#
#     return norm
#
#
# """
# AVAILABLE FUNCTION
#
# obtain and store the required object patches for the required image
# """
# def aggr_given_co(depths, labels, img_start, img_end, dim, path):
#
#     output_patches = []
#     output_targets = []
#
#     # do this for each given image
#     for img in range(img_start, img_end+1):
#         img_depths = depths[img]
#         img_labels = labels[img]
#         x, y = img_labels.shape
#         set_labels = list(set(img_labels.reshape(x*y,))) # the set of labels in the image
#         print set_labels
#
#         # do this for all the labels
#         for lbl in set_labels:
#             patches = entry_given_co(img_depths, img_labels, lbl, dim)
#
#             # append only if not empty
#             if patches:
#                 output_patches.extend(patches)
#                 output_targets.extend([lbl for l in range(len(patches))])
#
#         # store feature-target dictionary and reset outputs
#         save_data(create_ft_dict(np.array(output_patches), np.array(output_targets)),'ft_co_'+str(img)+'.p', path)
#         output_patches = []
#         output_targets = []



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
    p_topn.add_argument('-ls', '-label_s', action='store', dest='label_s', type=int, help='the staring label to be explored')
    p_topn.add_argument('-le', '-label_e', action='store', dest='label_e', type=int, help='the ending label to be explored')
    p_topn.add_argument('-imgs', '-images', action='store', dest='images', type=int, help='the number of images to draw the sample from')
    p_topn.add_argument('-n', action='store', dest='n', type=int, help='the number of random samples required')
    p_topn.set_defaults(which='top_n')

    # n_random
    p_rand = subparsers.add_parser('rand', help='get n random samples from training set')
    p_rand.add_argument('-ls', '-label_s', action='store', dest='label_s', type=int, help='the staring label to be explored')
    p_rand.add_argument('-le', '-label_e', action='store', dest='label_e', type=int, help='the ending label to be explored')
    p_rand.add_argument('-n', action='store', dest='n', type=int, help='the number of random samples required')
    p_rand.set_defaults(which='rand')

    p_lbl = subparsers.add_parser('lbl', help='get patches from all images for a given label')
    p_lbl.add_argument('-ls', '-label_s', action='store', dest='label_s', type=int, help='the staring label to be explored')
    p_lbl.add_argument('-le', '-label_e', action='store', dest='label_e', type=int, help='the ending label to be explored')
    p_lbl.add_argument('-d', '-dim', action='store', dest='dim', type=int, help='dimension of patches')
    p_lbl.set_defaults(which='lbl')

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

    # depths = pickle.load(open('test_depths.p', 'rb'))
    # labels = pickle.load(open('test_labels.p', 'rb'))

    args = parser()
    print 'args:', args

    if not check_args(args):
        print >> sys.stderr, 'invalid parameters inputted -> use -h to find out the required parameters'
        sys.exit(1)

    # find out which function to perform
    if args.which == 'patches':
        print 'in patches'
        # load required data
        # depths = np.array([[[1,2,3,10,11,12],[4,5,6,13,14,15],[7,8,9,16,17,18]]])
        # labels = np.array([[[13,13,14,19,24,26],[15,15,19,42,44,46],[4,3,2,9,10,11]]])

        if args.fn == 'per_pixel':
            depths, labels = initialise(path)
            print 'running per_pixel'
            aggr_per_pixel(depths, args.img_s, args.img_e, args.dim, path)
            print 'done per_pixel'
        # elif args.fn== 'co':
        #     print 'running co'
        #     aggr_given_co(depths, labels, args.img_s, args.img_e, args.dim, path)
        #     print 'done co'
        elif args.fn == 'lbl':
            labels = load_data('labels.p','rb', path)
            labels2imgs_i = load_data('labels2imgs_ignore.p', 'rb', path)
            patches_per_label(args.label_s, args.label_e, labels, labels2imgs_i, args.dim, path)

    elif args.which == 'top_n':
        print 'running top_n'
        labels2imgs = load_data('labels2imgs.p', 'rb', path)
        print '\n'
        top_n(args.label_s, args.label_e, labels2imgs, args.images, args.n, path)
        print '\n'
        print 'done top_n'

    elif args.which == 'rand':
        n_random_records(args.n, args.label_s, args.label_e, path+'top/')

    elif args.fn == 'lbl':
        labels = load_data('labels.p','rb', path)
        labels2imgs_i = load_data('labels2imgs_ignore.p', 'rb', path)
        patches_per_label(args.label_s, args.label_e, labels, labels2imgs_i, args.dim, path)

    else:
        print >> sys.stderr, 'possible inputs: per_pixel, co, top_n, rand, lbl'
        sys.exit(1)


"""
testing point
"""
if __name__ == '__main__':
    main()
