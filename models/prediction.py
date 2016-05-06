#!/usr/bin/env python

import numpy as np
import sys, argparse
from random import randrange
from sklearn.externals import joblib as jl
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


"""
generate unique colours for each object
"""
def gen_colours(size=894):
    colours = {0:[0,0,0]}

    for i in range(1,size+1):
        colours.update({i:[randrange(0,255), randrange(0,255), randrange(0,255)]})

    return colours


"""
generate output image using predictions of each pixel
"""
def gen_image(predictions, colours_dict, dim=15, x_im=626, y_im=466):

    # size of x y
    # x, y = x_im-(dim-1), y_im-(dim-1)
    x,y = x_im, y_im

    # print predictions =shape
    predictions = predictions.reshape(x,y)
    print 'predictions.shape', predictions.shape
    # output = np.array(np.array([ np.array([None for i in range(y)]).astype('float_') for j in range(x) ]).astype('float_')).reshape(x,y)
    output = np.array([ [np.array([None for colour in range(3)]) for i in range(y)] for j in range(x) ]).astype('float_')
    # output = np.array([ [None for i in range(y)] for j in range(x) ]).astype('float_')
    print 'output.shape', output.shape

    for row in range(x):
        for col in range(y):
            # fill in the colours
            print 'row,col:', (row,col), ', predictions[row,col]:', predictions[row,col]
            print np.array(colours_dict[predictions[row, col]])
            # output[row, col] = np.array(colours_dict[predictions[row, col]]).astype('float_')
            output[row, col] = np.array(colours_dict[predictions[row, col]])
            # output[row, col] = predictions[row, col]

    print 'output shape:', output.shape

    return output


"""
save image
"""
def save_figure(img, filename, dpi_val, path=''):
    # fig = plt.figure(frameon=False)
    # ax  = plt.Axes(fig, [0.,0.,1.,1.])

    # ax.set_axis_off()
    # fig.add_axes(ax)

    # ax.imshow(img)
    # fig.savefig(path+filename)
    plt.imshow(img, aspect='equal')
    plt.axis('off')
    plt.savefig(path+filename, bbox_inches='tight') #, frameon=False, bbox_inches='tight')

    print 'image saved'


"""
save data to file

input:
    - (any) data: data to be saved
    - (string) filename: the name of the file to be saved
    - (string) path: store it outside of 'root'
"""
def save_data(data, filename, mode, path=''):
    if mode == 'np':
        np.save(path+filename, data)
        print 'data saved'
    elif mode == 'jl':
        jl.dump(data, path+filename+'.jl')
        print 'data saved'
    else:
        print 'data not save'



"""
load data from file
"""
def load_data(filename, mode, path=''):
    if mode == 'np':
        return np.load(path+filename+'.npy')
    elif mode == 'jl':
        return jl.load(path+filename+'.jl', mmap_mode='r+')



"""
perform prediction given a model

input:
    - (np.array) data: the pixels to be predicted

output:
    - (np.array) prediction
"""
def prediction(model, data, type):
    if type == 'data':
        print('average score using .score():', model.score(data['features'], data['targets']))
        return model.predict(data['features'])
    elif type == 'img':
        return model.predict(data)

"""
precision-recall report
"""
def precision_recall(original, predicted):
    print classification_report(original, predicted)


def merge_labels(labels, path):
    to_merge = load_data('per_lbl_less1000', 'np', path+'lbl/').tolist()

    for key in to_merge.keys():
        locs = np.where(labels == key)[0]

        for loc in locs:
            labels[loc] = 900

    return labels

"""
command line argument parser
"""
def parser():
    parser = argparse.ArgumentParser(description='obtain predictions and create predicted image')
    subparsers = parser.add_subparsers(help='different arguments for different activities')

    p_predict = subparsers.add_parser('predict', help='predict some given data on a given trained model')
    p_predict.add_argument('-df', '-datafile', action='store', dest='file', help='filename of the data file to be predicted')
    p_predict.add_argument('-mf', '-modelfile', action='store', dest='model', help='filename of the model to be used')
    p_predict.add_argument('-save', action='store', dest='save', help='filename of the saved file')
    p_predict.add_argument('-s_flag', action='store', type=int, dest='s_flag', help='whether to save the file')
    p_predict.add_argument('-t', '-type', action='store', dest='type', help='whether to predict data sets or img')
    p_predict.set_defaults(which='predict')

    p_gen = subparsers.add_parser('gen', help='generate an image based on prediction')
    p_gen.add_argument('-img', '-image', action='store', type=int, dest='img', help='the image we are dealing with')
    p_gen.add_argument('-p_file', '-pre_file', action='store', dest='pre_file', help='the prediction file we need')
    p_gen.set_defaults(which='gen')

    p_pc = subparsers.add_parser('precall-data', help='precision-recall report')
    p_pc.add_argument('-p', '-predicted', action='store', dest='predicted', help='filename of predicted values')
    p_pc.add_argument('-o', '-original', action='store', dest='original', help='filename of original dataset')
    p_pc.set_defaults(which='precall-data')

    p_pcimg = subparsers.add_parser('precall-img', help='precision-recall report')
    p_pcimg.add_argument('-p', '-predicted', action='store', dest='predicted', help='filename of predicted values')
    p_pcimg.add_argument('-img', '-image', action='store', dest='img', help='the image we are dealing with')
    p_pcimg.set_defaults(which='precall-img')

    p_save = subparsers.add_parser('save-fig', help='save figure from image matrices')
    p_save.add_argument('-img_file', action='store', dest='img_file', help='image data file')
    p_save.add_argument('-out_file', action='store', dest='out_file', help='filename of the final image')
    p_save.set_defaults(which='save-fig')

    args = parser.parse_args()

    return args


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

    # check arguments to see if all the necessary arguments are given
    if not check_args(args):
        print >> sys.stderr, 'invalid parameter(s) inputted -> use -h to find out the required parameters'
        sys.exit(1)

    # find out which function to perform
    # possible functions: predict, gen
    if args.which == 'predict':
        # TODO: predict some given data on a given trained model
        model = load_data(args.model, 'jl', path+'model/')
        if args.type == 'data':
            print 'in data'
            dataset = load_data(args.file, 'np', path+'lbl/').tolist()
        elif args.type == 'img':
            print 'in img'
            dataset = load_data(args.file, 'np', path+'px/')
            dataset = dataset.reshape(dataset.shape[0], 225)

        if args.s_flag == 1:
            print 'to save'
            save_data(prediction(model, dataset, args.type), args.save, 'np', path+'prediction/')
        elif args.s_flag == 0:
            print 'to not save'
            prediction(model, dataset, args.type)

    elif args.which == 'gen':
        # TODO: predict all patches, gen images with those labels
        colours = load_data('colours_f', 'np', path).tolist()
        predicted = load_data(args.pre_file, 'np', path+'prediction/')

        generated = gen_image(predicted, colours) #\
            # if args.x is not None and args.y is not None \
            # else gen_image(pre, colours, args.dim)

        save_data(generated, 'gen_'+str(args.img)+'.npy', 'np', path+'generated/')
        save_figure(generated, 'gen_'+str(args.img)+'.png', 150, path+'generated/')
        print 'saved generated image'

    elif 'precall' in args.which :
        predicted = load_data(args.predicted, 'np', path+'prediction/')

        if args.which == 'precall-data':
            original = load_data(args.original, 'np', path+'lbl/').tolist()['targets']

        elif args.which == 'precall-img':
            original = load_data('labels', 'np', path)[args.img, 7:633, 7:473]
            # original = np.transpose(original).reshape(466,626).reshape(626*466,)
            original = original.reshape(626*466)
            # original = merge_labels(original, path)
            
            print original

        precision_recall(original, predicted)

    elif args.which == 'save-fig':
        data = load_data(args.img_file, 'np', path+'generated/')
        save_figure(data, args.out_file, 150, path+'generated/')

    else:
        # error message
        print >> sys.stderr, 'possible inputs: predict, gen\n', \
            '    predict-data - predict and save prediction , given image patches (dataset)\n', \
            '    predict-img  - predict and save prediction , given image patches (image)\n', \
            '    gen          - predict, generate and save image , given prediction'
        sys.exit(1)


"""
run
"""
if __name__ == '__main__':
    main()
