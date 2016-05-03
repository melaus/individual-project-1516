#!/Users/melaus/virtualenv/env_python2/bin/python
import h5py
import numpy as np
import cPickle as pickle
from operator import *

def get_numbers(matlab, dataset):
    """
    get number MatLab data and convert it into numpy format
    """
    return np.array(matlab.get(dataset)) 


def get_strings(matlab, dataset):
    """
    get string MatLab data and convert it into numpy format
    """
    data = []

    print 'len(matlab[dataset]),', len(matlab[dataset])
    for column in matlab[dataset]:
        row_data = []
        for row_number in range(len(column)):            
            row_data.append(''.join(map(unichr, matlab[column[row_number]][:])))
        print 'row_data:', row_data
        data.append(row_data)

    #return np.transpose(data)
    #return np.ravel(data)
    return data



def write_to_file(filename, dataset):
    """
    write to file
    """
    #np.savetxt(filename, dataset, delimiter='\t', fmt=fmt)
    # pickle.dump(dataset, open(filename, 'w'))
    np.save(filename, dataset)


def create_names_map(dataset):
    """
    map names to numbers
    """
    return dict(zip(dataset, range(1,895)))

def run():
    """
    run everything required to gather data and output it to files
    """
    # where to get data from
    # matlab = h5py.File('nyu_depth_v2_labeled.mat', 'r')
    matlab = h5py.File('names_extract.mat', 'r')
    #number_datasets = [ 'accelData', 'depths',
                        #'images', 'instances', 
                        #'labels' ] 
    number_datasets = [ 'instances', 
                        'labels' ] 
    # other fields: namesToIds
    string_datasets = ['names_extract'] #, 'scenes']

    # get string datasets 
    for string in string_datasets:
        dataset = get_strings(matlab, string)
        print string, np.shape(dataset) # quickly check if things are okay
        write_to_file(string, dataset)
        
        # create names map
        if string == 'names':
            names_map = create_names_map(dataset.ravel())
            write_to_file(string+'_map', names_map)

    # get number datasets
    #for number in number_datasets:
        #dataset = get_numbers(matlab, number)
        #print number, np.shape(dataset) # quickly check if things are okay
        #print dataset[0]
        #write_to_file(number+'.p', dataset) 
    # print get_strings(matlab, 'names')

if __name__ == '__main__':
    run()
