#!/usr/bin/env python

import numpy as np
import cPickle as pickle
from random import randrange
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
def gen_image(predictions, colours_dict, dim):

    # size of x y
    x, y = 640-(dim-1), 480-(dim-1)

    predictions = predictions.reshape(y,x)
    output = np.array([ [None for i in range(y)] for j in range(x) ]).reshape(y,x)
    print output
    print ''
    print ''

    for row in range(x):
        for col in range(y):
            # fill in the colours
            # output[row, col] = colours_dict[predictions[row, col]]
            output[row, col] = predictions[row, col]

    return output


"""
save image
"""
def save_figure(img, filename, dpi_val, path=''):
    fig = plt.figure(frameon=False, dpi=dpi_val)
    ax  = plt.Axes(fig, [0.,0.,1.,1.])

    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(img, aspect='auto')
    fig.savefig(path+filename)

    print 'image saved'


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
ENTRY

perform prediction, and generate and store image
"""
def entry_predict_image(model, patches, colours, dim, path):
    print 'start prediction'
    return prediction(model, patches)

    # print 'start generated'
    # generated = gen_image(predicted, colours, dim)
    # print 'end generated'
    # print 'start save figure'
    # save_figure(generated, 'generated_pp_6', 150, path)
    # print 'end save figure'
    # print 'done prediction and generation'


"""
ENTRY

main function
"""
def main():
    # path = '/Users/melaus/repo/uni/individual-project/data/py-data/'
    path = '~/scratch/data/ip/'

    print 'start importing all files'
    svc_rbf = load_data('svc_rbf_6.p', 'rb', path)
    patches = load_data('img_6_per_pixel.p', 'rb', path)
    patches = patches.reshape(291716, 225)
    colours = load_data('colours.p', 'rb', path)
    # area_depths = load_data('area_depths_6.p', 'rb', path)
    print 'end importing all files'

    # save_data(entry_predict_image(svc_rbf, patches, colours, 15, path), 'prediction_img_6.p', path)
    save_data(svc_rbf.predict(patches), 'prediction_img_6_np.p', path)
    print 'done'


"""
run
"""
if __name__ == '__main__':
    path = '/Users/melaus/repo/uni/individual-project/data/py-data/'

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
