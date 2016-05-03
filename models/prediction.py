#!/usr/bin/env python

import numpy as np
import cPickle as pickle
import sys, argparse
from random import randrange
from matplotlib import pyplot as plt
from sklearn.externals import joblib as jl


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
def gen_image(predictions, colours_dict, dim, x_im=640, y_im=480):

    # size of x y
    x, y = x_im-(dim-1), y_im-(dim-1)

    predictions = predictions.reshape(x,y)
    # output = np.array(np.array([ np.array([None for i in range(y)]).astype('float_') for j in range(x) ]).astype('float_')).reshape(x,y)
    output = np.array([ [[None for colour in range(3)] for i in range(y)] for j in range(x) ]).astype('float_')

    for row in range(x):
        for col in range(y):
            # fill in the colours
            output[row, col] = np.array(colours_dict[predictions[row, col]]).astype('float_')
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
load data from file
"""
def load_data(filename, mode, path=''):
    return pickle.load(open(path+filename, mode))



"""
perform prediction given a model

input:
    - (np.array) data: the pixels to be predicted

output:
    - (np.array) prediction
"""
def prediction(model, data):
    return model.predict(data)



"""
command line argument parser
"""
def parser():
    parser = argparse.ArgumentParser(description='obtain predictions and create predicted image')
    subparsers = parser.add_subparsers(help='different arguments for different activities')

    p_predict = subparsers.add_parser('predict', help='prediction')
    p_predict.add_argument('-img', '-image', action='store', type=int, dest='img', help='the image we are dealing with')
    p_predict.add_argument('-dim', '-dimension', action='store', type=int, dest='dim', help='dimension of a patch')
    p_predict.set_defaults(which='predict')

    # optional parameters
    p_gen = subparsers.add_parser('')

    p_gen.add_argument('-x', '-width', action='store', type=int, dest='x', help='width of image')
    p_gen.add_argument('-y', '-height', action='store', type=int, dest='y', help='height of image')
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
    # possible functions: pre, gen
    if args.which == 'predict':
        model = load_data(args.model+'.p', 'jl', path)
        patches = load_data('px_'+str(args.dim)+'_'+args.img+'.p', 'np', path)

        save_data(prediction(model, patches), 'pre_'+str(args.img)+'_all.npy', path+'prediction/')
        print 'saved prediction'

    elif args.which == 'gen':
        colours = load_data('colours_f.npy', 'np', path)
        pre = load_data('pre_'+str(args.dim)+'_'+str(args.img)+'.npy', 'np', path)

        generated = gen_image(pre, colours, args.dim, args.x, args.y) #\
            # if args.x is not None and args.y is not None \
            # else gen_image(pre, colours, args.dim)

        save_data(generated, 'gen_'+str(args.img)+'.p', path+'generated/')
        save_figure(generated, 'gen_'+str(args.img)+'.png', 150, path)
        print 'saved generated image'

    else:
        # error message
        print >> sys.stderr, 'possible inputs: pre, gen, pre_gen\n', \
                             '    pre     - predict and save prediction      , given image patches\n', \
                             '    gen     - generate and save image          , given prediction'
        sys.exit(1)


"""
run
"""
if __name__ == '__main__':
    main()

    # load data
    # svc_rbf    = pickle.load(open(path+'svc_rbf_6.p', 'rb'))
    # svc_linear = pickle.load(open(path+'svc_linear_6.p', 'rb'))
    # data       = pickle.load(open(path+'area_depths_6.p', 'rb'))


    """ RBF """
    ## see what elements are being predicted
    #rbf_predicted_2 = svc_rbf.predict(data_np)
    #print 'rbf length:', len(rbf_predicted_2)
    #print 'rbf set:   ', set(list(rbf_predicted_2))
    
    #pickle.dump(rbf_predicted_2, open(path+'rbf_predicted_2.p', 'wb'))
    #print 'written to file rbf_predicted.p'
    
    ## predicted colours
    #colours = gen_colours(rbf_predicted_2, 15) 
    #print 'colours for those pixels'

    ## predicted image matrix
    ##img = [list(i) for i in gen_image(630,480,15, colours).reshape(480,630)]
    #img = list([list(j) for j in gen_image(630,480,15,colours)])
    #pickle.dump(img, open(path+'rbf_predicted_img_2.p', 'wb'))
    #print 'stored predicted image matrix'
    #print 'done'

    """ linear """
    ## see what elements are being predicted
    #linear_predicted = svc_linear.predict(data_np)
    #print 'linear length:', len(linear_predicted)
    #print 'linaer set:   ', set(list(linear_predicted))
    
    #pickle.dump(linear_predicted, open(path+'linear_predicted.p', 'wb'))
    #print 'written to file linear_predicted.p'
    
    ## predicted colours
    #colours = gen_colours(linear_predicted, 15) 
    #print 'colours for those pixels'

    ## predicted image matrix
    #img = list(list(k) for k in gen_image(630,480,15, colours))
    ##img = gen_image(630,480,15, colours)
    #pickle.dump(img, open(path+'linear_predicted_img.p', 'wb'))
    #print 'stored predicted image matrix'
    #print 'done'

    """ confusion prediction - rbf """
    #linear_cf_model = pickle.load(open(path+'linear_cf_model.p','rb'))
    #linear_cf_test_data = pickle.load(open(path+'linear_cf_test_data.p', 'rb'))

    #pickle.dump(linear_cf_model.predict(linear_cf_test_data["features"]), open(path+'linear_cf_predicted.p', 'wb'))

    """ confusion prediction - linear """
    # rbf_cf_model = pickle.load(open(path+'rbf_cf_model.p','rb'))
    # rbf_cf_test_data = pickle.load(open(path+'rbf_cf_test_data.p', 'rb'))
    #
    # pickle.dump(rbf_cf_model.predict(rbf_cf_test_data["features"]), open(path+'rbf_cf_predicted.p', 'wb'))
