
import pickle as pickle
import numpy as np
import sys
import math as m
import argparse
import glob
import re
from math import ceil
from itertools import groupby
from operator import itemgetter
from sklearn.cluster import KMeans
from sklearn.cross_validation import StratifiedShuffleSplit
import time
#import matplotlib.pyplot as plt

"""
get the data required

input:
    - img - image number
"""
def initialise(path='',img=''):
    # load image data
    # TODO: NOT GOING TO WORK ANYMORE
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
            data = load_data('co/ft_co_'+str(img)+'.p', 'rb', 'p', path)
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
        save_data(output_dict, 'top/top_'+str(label)+'_'+str(int(extract_imgs))+'_'+str(int(col_sum))+'.p', 'p', path)
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
def load_data(filename, mode, method, path=''):
    if method == 'p':
        return pickle.load(open(path+filename, mode))
    elif method == 'np':
        print 'load file path:',path+filename,'\n'
        return np.load(path+filename)


"""
save data to file

input:
    - (any) data: data to be saved
    - (string) filename: the name of the file to be saved
    - (string) path: store it outside of 'root'
"""
def save_data(data, filename, method, path=''):
    if method == 'p':
        pickle.dump(data, open(path+filename, 'wb'))
    elif method == 'np':
        np.save(path+filename, data)

    print 'data saved using', method


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
        save_data(np.array(entry_per_pixel(depths[img], dim)), 'px_'+str(dim)+'_'+str(img)+'.p', 'p', path)


"""
obtain patches for each label
"""
def patches_per_label(label_s, label_e, labels, labels2imgs_i, dim, path):
    lim = int(m.floor(dim/2))
    x, y = 640, 480

    out = np.array([])

    # obtain patches for all images that has labels
    for lbl in range(label_s, label_e+1):
        print '\n\n--- in lbl', lbl, '---\n'

        if lbl in [0,3,21]:
            imgs = sorted(np.random.choice(labels2imgs_i[lbl], 400, replace=False))
            save_data(imgs, 'per_lbl_'+str(lbl)+'_imgs', 'np', path+'lbl/')
            print 'using new length:     ', len(imgs) 
        else:
            imgs = labels2imgs_i[lbl] 
            print 'using original length:', len(imgs)

        img_ctr = 1;

        for img in imgs:
            print 'img_ctr:', img_ctr
            img_ctr += 1
            # get the positions of the required features
            print '[lim:x-lim]:', (lim, x-lim), 'lim:y-lim', (lim, y-lim)
            img_labels = labels[img, lim:x-lim, lim:y-lim].reshape((x-lim*2)*(y-lim*2))
            pos = sorted(np.where(img_labels == lbl)[0])
            print 'img', img, 'has', len(pos), 'positions with the label'

            # open the required patch file
            px = load_data('px_'+str(dim)+'_'+str(img)+'.npy', 'rb', 'np', path+'px/')

            # add to output array
            if len(out) == 0 :
                out = px[pos,:,:]
            else:
                out = np.append(out, px[pos,:,:], axis=0)

        print 'shape:', out.shape
        print ''

        # save using numpy
        save_data(out, 'per_lbl_'+str(lbl), 'np', path+'lbl/')
        out = np.array([])


"""
get [n] random records from [per_lbl] for kmeans
"""
def n_random_records(records, label_s, label_e, path=''):
    features = np.array([])

    print '===== RAND',label_s, label_e,'====='

    # labels to deal with
    all = np.load(path+'remaining_large.npy')
    labels =  [x for x in range(label_s, label_e+1) if x in all] 
    print 'number of labels to deal with:', len(labels)

    for lbl in labels:
        print '\n--- lbl', lbl, '---'

        filename = 'per_lbl_'+str(lbl)+'.npy'

        print 'load data from', filename
        data = load_data(filename, '', 'np', path)
        print 'shape of data:', data.shape

        # find random position 
        random_list = np.array(range(len(data)))
        random_list = sorted(np.random.choice( random_list, records, replace=False ))
        print 'random_list size:', len(random_list)

        # save random records
        random_records = np.array( [data[loc] for loc in random_list] )
        print 'random_records shape:', random_records.shape
        save_data(random_records, 'per_lbl_'+str(lbl)+'_'+str(records), 'np', path)

    print '===== END RAND =====\n\n'


"""
use k-means to create a smaller dataset
"""
def kmeans(init, n_clusters, n_init, label_s, label_e, path):

    for lbl in range(label_s, label_e+1):
        print '--- lbl', lbl, '---'
        data = load_data('per_lbl_'+str(lbl)+'_100000.npy', '', 'np', path+'lbl/lbl/')

        # to simplify process, we'll just specify start and end label and use this to ignore anything we don't want
        if len(data) >= 0 and len(data) <= 1000:
            continue

        print 'ori shape of data:', data.shape

        if len(data.shape) == 3:
            i, x, y = data.shape
            print 'i, x, y:', i, x, y
            data = data.reshape(i,x*y)
            print 'new shape of data:', data.shape
        
        cls = KMeans(init=init, n_clusters=n_clusters, n_init=n_init)
        
        t0 = time.time()
        cls.fit(data)
        t1 = time.time()
        print 'time taken to get', len(cls.cluster_centers_), 'features in', t1-t0, 'seconds'

        save_data(cls.cluster_centers_, 'per_lbl_'+str(lbl)+'_slim', 'np', path+'lbl/')
        print ''

def combine_data(path=''):
    print '--- combine_data ---'
    labels = range(1,3)
    all_kmeans = np.load(path+'all_kmeans.npy')
    
    features = np.array([])
    targets  = np.array([])
    lengths = dict()

    # load each label and check output
    for lbl in labels:

        # load data
        print 'load data of', lbl
        if lbl in all_kmeans:
            file = 'per_lbl_'+str(lbl)+'_'+'slim.npy'
            data = np.load(path+file)
            print 'k_meaned filename:', file, 'of shape', data.shape
        else:
            file = 'per_lbl_'+str(lbl)+'.npy'
            data = np.load(path+file)
            data = data.reshape(len(data), 225)
            print 'normal filename:  ', file, 'of shape', data.shape

        # ignore 0-length labels
        if len(data) == 0:
            print '\n\n= lbl', lbl, 'ignored\n'
            continue
        else:
            lengths[lbl] = data.shape[0]
            print 'lbl',lbl,'> 0, added to lengths'
        
        # append data to array 
        if len(features) == 0 and len(targets) == 0:
            features = data
            targets = np.array([lbl for x in range(len(data))])
            print 'len(targets) so far:', len(targets)
        else:
            features = np.append(features, data, axis=0)
            targets = np.append(targets, np.array([lbl for x in range(len(data))]))
            print 'len(targets) so far:', len(targets)

        print 'added to dict\n\n'

    save_data(create_ft_dict(features, targets), 'combined_1_2', 'np', path)
    # save_data(lengths, 'per_lbl_lengths_ked', 'np', path)


"""
create required datasets
"""
def datasets(filename, path):
    data = np.load(path+filename).tolist()

    # train/test, validation split
    print 'tt and val split'
    sss = StratifiedShuffleSplit(data['targets'], n_iter=1, test_size=0.3, random_state=42)
    for tt_index, val_index in sss:
        X_tt  = data['features'][tt_index]
        X_val = data['features'][val_index]
        y_tt  = data['targets'][tt_index]
        y_val = data['targets'][val_index]


    # train/test, validation split
    print 'val split'
    sss = StratifiedShuffleSplit(y_tt, n_iter=1, test_size=0.3, random_state=42)
    for train_index, test_index in sss:
        X_train = X_tt[train_index]
        X_test  = X_tt[test_index]
        y_train = y_tt[train_index]
        y_test  = y_tt[test_index]

    print 'shape of training:  ', X_train.shape
    print 'shape of testing:   ', X_test.shape
    print 'shape of validation:', X_val.shape

    id = re.search('_[0-9]*_[0-9]*', filename).group(0)
    save_data(create_ft_dict(X_train, y_train), 'data_train'+id, 'np', path)
    save_data(create_ft_dict(X_test, y_test), 'data_test'+id, 'np', path)
    save_data(create_ft_dict(X_val, y_val), 'data_val'+id, 'np', path)


"""
merge labels in datasets
"""
def merge_labels(filename, merge_file_info, path):
    dataset = np.load(path+'lbl/'+filename).tolist()
    to_merge = np.load(path+'lbl/per_lbl_'+merge_file_info+'.npy').tolist()

    for key in to_merge.keys():
        locs = np.where(dataset['targets'] == key)[0]

        for loc in locs:
            dataset['targets'][loc] = 900

    save_data(dataset, 'combined_'+merge_file_info, 'np', path+'lbl/')

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

    p_pts = subparsers.add_parser('pts', help='k-means clustering')
    p_pts.add_argument('-n_init', action='store', dest='n_init', type=int, help='number of seeds')
    p_pts.add_argument('-n_clusters', action='store', dest='n_clusters', type=int, help='number of cluster/ points to generate')
    p_pts.add_argument('-ls', '-label_s', action='store', dest='label_s', type=int, help='the staring label to be explored')
    p_pts.add_argument('-le', '-label_e', action='store', dest='label_e', type=int, help='the ending label to be explored')
    p_pts.set_defaults(which='pts')

    p_com = subparsers.add_parser('combine', help='combine multiple files as one feature-target dictionary')
    p_com.set_defaults(which='combine')

    p_data = subparsers.add_parser('data', help='create training dataset')
    p_data.add_argument('-f', '-filename', action='store', dest='filename', help='filename')
    p_data.set_defaults(which='data')

    p_merge = subparsers.add_parser('merge', help='create training dataset')
    p_merge.add_argument('-f', '-filename', action='store', dest='filename', help='filename')
    p_merge.add_argument('-info', '-merge_file_info', action='store', dest='info', help='the criteria for labels to be merge')
    p_merge.set_defaults(which='merge')

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

    args = parser()
    print 'args:', args

    if not check_args(args):
        print >> sys.stderr, 'invalid parameters inputted -> use -h to find out the required parameters'
        sys.exit(1)

    # find out which function to perform
    if args.which == 'patches':
        print 'in patches'

        if args.fn == 'per_pixel':
            depths, labels = initialise(path)
            print 'running per_pixel'
            aggr_per_pixel(depths, args.img_s, args.img_e, args.dim, path)
            print 'done per_pixel'
        # elif args.fn== 'co':
        #     print 'running co'
        #     aggr_given_co(depths, labels, args.img_s, args.img_e, args.dim, path)
        #     print 'done co'

    elif args.which == 'top_n':
        print 'running top_n'
        labels2imgs = load_data('labels2imgs.p', 'rb', 'p', path)
        print '\n'
        top_n(args.label_s, args.label_e, labels2imgs, args.images, args.n, path)
        print '\n'
        print 'done top_n'

    elif args.which == 'rand':
        n_random_records(args.n, args.label_s, args.label_e, path+'lbl/lbl/')

    elif args.which == 'lbl':
        labels = load_data('labels.npy','', 'np', path)
        labels2imgs_i = load_data('labels2imgs_ignore.p', 'rb', 'p', path)
        patches_per_label(args.label_s, args.label_e, labels, labels2imgs_i, args.dim, path)

    elif args.which == 'pts':
        kmeans('k-means++', args.n_clusters, args.n_init, args.label_s, args.label_e, path)

    elif args.which == 'combine':
        combine_data(path+'lbl/')

    elif args.which == 'merge':
        merge_labels(args.filename, args.info, path)

    elif args.which == 'data':
        datasets(args.filename, path+'lbl/')

    else:
        print >> sys.stderr, 'possible inputs: per_pixel, co, top_n, rand, lbl'
        sys.exit(1)


"""
testing point
"""
if __name__ == '__main__':
    main()
