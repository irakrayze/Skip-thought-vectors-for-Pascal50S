import os.path
import numpy as np
from scipy.misc import imread, imresize
import scipy.io
import caffe
from os import listdir
from os.path import isfile, join

"""
This code is based on https://github.com/karpathy/neuraltalk/blob/master/python_features/extract_features.py
The features were extracted from the 'relu7' and 'conv5_4' layers, based on 
https://github.com/elliottd/GroundedTranslation/tree/master/matlab_features_reference
"""

Use_CPU= True

def predict(in_data, net):
    """
    Get the features for a batch of data using network
    Inputs:
    in_data: data batch
    """

    out = net.forward(**{net.inputs[0]: in_data})
    features = out[net.outputs[0]]
    return features


def batch_predict(filenames, net):
    """
    Get the features for all images from filenames using a network
    Inputs:
    filenames: a list of names of image files
    Returns:
    an array of feature vectors for the images in that file
    """

    N, C, H, W = net.blobs[net.inputs[0]].data.shape
    F = (net.blobs[net.outputs[0]].data.shape[1])*14*14
    Nf = len(filenames)
    Hi, Wi, _ = imread(filenames[0]).shape
    allftrs = np.zeros((Nf, F))
    for i in range(0, Nf, N):
        in_data = np.zeros((N, C, H, W), dtype=np.float32)

        batch_range = range(i, min(i+N, Nf))
        batch_filenames = [filenames[j] for j in batch_range]
        Nb = len(batch_range)

        batch_images = np.zeros((Nb, 3, H, W))
        for j,fname in enumerate(batch_filenames):
            im = imread(fname)
            if len(im.shape) == 2:
                im = np.tile(im[:,:,np.newaxis], (1,1,3))
            # RGB -> BGR
            im = im[:,:,(2,1,0)]
            # mean subtraction
            im = im - np.array([103.939, 116.779, 123.68])  # VGG mean assumed
            # resize
            im = imresize(im, (H, W), 'bicubic')
            # get channel in correct dimension
            im = np.transpose(im, (2, 0, 1))
            batch_images[j,:,:,:] = im

        # insert into correct place
        in_data[0:len(batch_range), :, :, :] = batch_images

        # predict features
        ftrs = predict(in_data, net)

        for j in range(len(batch_range)):
            ftrs_new = np.reshape(ftrs[j,:], 512*14*14)
            allftrs[i+j,:] = ftrs_new

        print 'Done %d/%d files' % (i+len(batch_range), len(filenames))

    return allftrs


if Use_CPU:
    caffe.set_mode_cpu()
else:
    caffe.set_mode_gpu()

model_def='deploy_features-conv54.prototxt'
#model_def='deploy_features-fc7.prototxt'
weights_net='VGG_ILSVRC_19_layers.caffemodel'
net = caffe.Net(model_def, weights_net, caffe.TEST) 

filenames = []

mypath='/home/ira/Documents/Pascal/PASCAL_Images'

files='Features_Pascal/Image_Names.txt'
only_images = [f for f in listdir(mypath) if isfile(join(mypath, f)) and '.jpg' in f]
with open(files, 'w') as f:
    for item in only_images:
        f.write("%s\n" % item)

base_dir = os.path.dirname(files) #the name of the directory that this file is in

with open(files) as fp:
    for line in fp:
        filename = os.path.join(base_dir, line.strip().split()[0])
        filenames.append(filename)

allftrs = batch_predict(filenames, net)
mat_output='Pascal_vgg19_feats.mat'
scipy.io.savemat(os.path.join(base_dir, mat_output), mdict =  {'feats': np.transpose(allftrs)})
