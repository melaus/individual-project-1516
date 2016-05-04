#!/usr/bin/env python

import numpy as np
import sys, argparse
from random import randrange
from matplotlib import pyplot as plt
from sklearn.externals import joblib as jl
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


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
def prediction(model, data):
    print('average score using .score():', model.score(data['features'], data['targets']))
    return model.predict(data['features'])

"""
precision-recall report
"""
def precision_recall(predicted, original):
    print classification_report(original, predicted)


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
    p_predict.set_defaults(which='predict')

    # optional parameters
    p_gen = subparsers.add_parser('gen', help='generate an image based on prediction')
    p_gen.add_argument('-img', '-image', action='store', type=int, dest='img', help='the image we are dealing with')
    p_gen.add_argument('-p_file', '-pre_file', action='store', dest='pre_file', help='the prediction file we need')
    p_gen.set_defaults(which='gen')

    p_pc = subparsers.add_parser('precall', help='precision-recall report')
    p_pc.add_argument('-p', '-predicted', action='store', dest='predicted', help='filename of predicted values')
    p_pc.add_argument('-o', '-original', action='store', dest='original', help='filename of original dataset')
    p_pc.set_defaults(which='precall')

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
        dataset = load_data(args.file, 'np', path+'lbl/').tolist()

        if args.s_flag == 1:
            save_data(prediction(model, dataset), args.savename, 'np', path+'prediction/')
        elif args.s_flag == 0:
            prediction(model, dataset)

    elif args.which == 'gen':
        # TODO: predict all patches, gen images with those labels
        colours = load_data('colours_f', 'np', path)
        predicted = load_data(args.pre_file, 'np', path)

        generated = gen_image(predicted, colours) #\
            # if args.x is not None and args.y is not None \
            # else gen_image(pre, colours, args.dim)

        save_data(generated, 'gen_'+str(args.img)+'.npy', 'np', path+'generated/')
        save_figure(generated, 'gen_'+str(args.img)+'.png', 150, path+'generated/')
        print 'saved generated image'

    elif args.which == 'precall':
        predicted = load_data(args.predicted, 'np', path+'prediction/').tolist()
        original = load_data(args.original, 'np', path+'lbl/').tolist()

        precision_recall(predicted, original)

    else:
        # error message
        print >> sys.stderr, 'possible inputs: predict, gen\n', \
                             '    predict - predict and save prediction , given image patches\n', \
                             '    gen     - predict, generate and save image , given prediction'
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
