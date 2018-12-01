from matplotlib.pyplot import imshow
from PIL import Image

import time
from pylab import *

def split_components(X, rX):
    train_x_lh = X[:,:,4:8,:]
    train_x_rh = X[:,:,8:12,:]
    train_x_ll = X[:,:,12:16,:]
    train_x_rl = X[:,:,16:20,:]
    train_x_t = X[:,:,0:4,:]

    rtrain_x_lh = rX[:,:,4:8,:]
    rtrain_x_rh = rX[:,:,8:12,:]
    rtrain_x_ll = rX[:,:,12:16,:]
    rtrain_x_rl = rX[:,:,16:20,:]
    rtrain_x_t = rX[:,:,0:4,:]

    rfinal_x = np.concatenate((rtrain_x_lh, rtrain_x_rh), axis=2)
    rfinal_x = np.concatenate((rfinal_x, rtrain_x_t), axis=2)
    rfinal_x = np.concatenate((rfinal_x, rtrain_x_rl), axis=2)
    rfinal_x = np.concatenate((rfinal_x, rtrain_x_ll), axis=2)


    final_x = np.concatenate((train_x_lh, train_x_rh), axis=2)
    final_x = np.concatenate((final_x, train_x_t), axis=2)
    final_x = np.concatenate((final_x, train_x_rl), axis=2)
    final_x = np.concatenate((final_x, train_x_ll), axis=2)

    return final_x, rfinal_x

def normalize_(X,rX):
    min_ = X.min(axis=(0,1,2)).reshape(1,1,1,-1)
    max_ = X.max(axis = (0,1,2)).reshape(1,1,1,-1)

    min1_ = min_
    max1_ = max_

    min_ = np.repeat(min_, 320, axis=0)
    min_ = np.repeat(min_, 150, axis=1)
    min_ = np.repeat(min_, 20, axis=2)

    max_ = np.repeat(max_, 320, axis=0)
    max_ = np.repeat(max_, 150, axis=1)
    max_ = np.repeat(max_, 20, axis=2)

    X_normed = 255*(X - min_)/(max_ - min_).astype(np.int64)

    rmin_ = rX.min(axis=(0,1,2)).reshape(1,1,1,-1)
    rmax_ = rX.max(axis = (0,1,2)).reshape(1,1,1,-1)

    rmin_ = np.repeat(min1_, 9, axis=0)
    rmin_ = np.repeat(min1_, 150, axis=1)
    rmin_ = np.repeat(min1_, 20, axis=2)

    rmax_ = np.repeat(max1_, 9, axis=0)
    rmax_ = np.repeat(max1_, 150, axis=1)
    rmax_ = np.repeat(max1_, 20, axis=2)

    rX_normed = 255*(rX - rmin_)/(rmax_ - rmin_).astype(np.int64)
    return X_normed, rX_normed

def load_dataset(x = 'x.npy', rX = 'cross_x.npy', y = 'labels.npy'):
    Y = np.load(Y)
    Y = Y-1
    X = np.load(X)
    X = np.array(X, dtype=float)

    rX = np.load(rX)

    X, rX = split_components(X,rX)

    X_normed, rX_normed = normalize_(X,rX)


    new_X = []
    for X_norm in X_normed:
        img = Image.fromarray(np.uint8(X_norm))
        img = img.resize((32, 32), Image.ANTIALIAS)
        img1 = img.transpose(Image.FLIP_LEFT_RIGHT)
        img2 = img.transpose(Image.FLIP_TOP_BOTTOM)
    #     imshow(img)
    #     show()
        time.sleep(0.0001)
        new_X.append(np.asarray( img, dtype="int32"))
        new_X.append(np.asarray( img1, dtype="int32"))
        new_X.append(np.asarray( img2, dtype="int32"))

    Y = np.repeat(Y,3,axis = 0)

    rnew_X = []
    for rX_norm in rX_normed:
        rimg = Image.fromarray(np.uint8(rX_norm))
        rimg = rimg.resize((32, 32), Image.ANTIALIAS)
        rimg1 = rimg.transpose(Image.FLIP_LEFT_RIGHT)
        rimg2 = rimg.transpose(Image.FLIP_TOP_BOTTOM)
    #     imshow(img)
    #     show()
        time.sleep(0.0001)
        rnew_X.append(np.asarray( rimg, dtype="int32"))
        rnew_X.append(np.asarray( rimg1, dtype="int32"))
        rnew_X.append(np.asarray( rimg2, dtype="int32"))
    new_X = np.array(new_X)
    rnew_X = np.array(rnew_X)

    return new_X, rnew_X, Y
