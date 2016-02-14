#!/usr/bin/env python

import numpy as np
import cPickle as pickle
#from matplotlib import pyplot as plt


"""
generate a matrix that can be drawn as an image
"""
def gen_colours(pts, dim):
    colours = {0:0, 5:255}
    return [[colours[pt]]*(dim*dim) for pt in pts]


"""
generate output image matrices
"""
def gen_image(img_x, img_y, dim, colours):
    # define the image size
    output = np.array([np.array([None for i in range(img_y)]) for j in range(img_x)])

    # initialising bounds
    x_lower, x_upper = 0, dim
    y_lower, y_upper = 0, dim

    # counters to get the correct colour value
    col_ctr = 0 
    item_ctr = 0 
    
    for y_group in range(img_y/dim):
        for x_group in range(img_x/dim):

            # get depths for these points
            for y in range(y_lower, y_upper):
                for x in range(x_lower, x_upper):
                    output[x][y] = (colours[col_ctr])[item_ctr]
                    item_ctr+=1
            col_ctr+=1
            item_ctr=0 # reset counter for next group

            # increase the y bounds to process the next group
            x_lower += dim
            x_upper += dim

        # reset x bounds
        x_lower = 0 
        x_upper = dim

        # increase y bounds to process the next group
        y_lower += dim
        y_upper += dim

    return output


"""
save image
"""
def save_figure(img, filename, dpi_val):
    fig = plt.figure(frameon=False, dpi=dpi_val)
    ax  = plt.Axes(fig, [0.,0.,1.,1.])

    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(img, aspect='auto')
    fig.savefig(filename)

    print 'image saved'


def prediction(model, dim, path=''):
    predicted = model.predict(data_np)


"""
run
"""
if __name__ == '__main__':
    path = '/Users/melaus/repo/uni/individual-project/data/py-data/'
    
    # load data
    svc_rbf    = pickle.load(open(path+'svc_rbf_6.p', 'rb'))
    svc_linear = pickle.load(open(path+'svc_linear_6.p', 'rb'))
    data       = pickle.load(open(path+'area_depths_6.p', 'rb'))
    
    # numpyify data
    data_np = np.array([np.array(i) for i in data])

    #""" RBF """
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
    # see what elements are being predicted
    linear_predicted = svc_linear.predict(data_np)
    print 'linear length:', len(linear_predicted)
    print 'linaer set:   ', set(list(linear_predicted))
    
    pickle.dump(linear_predicted, open(path+'linear_predicted.p', 'wb'))
    print 'written to file linear_predicted.p'
    
    # predicted colours
    colours = gen_colours(linear_predicted, 15) 
    print 'colours for those pixels'

    # predicted image matrix
    img = list(list(k) for k in gen_image(630,480,15, colours))
    #img = gen_image(630,480,15, colours)
    pickle.dump(img, open(path+'linear_predicted_img.p', 'wb'))
    print 'stored predicted image matrix'
    print 'done'
