# Install libjpeg-dev, libX11, Pillow
"""
In this problem, you will be dividing the images into blocks of sizes (16; 16); (32; 32);
(64; 64) and performing principal component analysis in each case. For each case you
will visualize reconstructions using dierent number of principal components and also
visualize the top components.
"""
from os import walk
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from PIL import Image

import theano
import theano.tensor as T
from theano.tensor.nnet.neighbours import images2neibs


def plot_mul(c, D, im_num, mn, num_coeffs=range(5, 50, 5), n_blocks=4):
    '''
    Plots nine PCA reconstructions of a particular image using number
    of components specified by num_coeffs

    Parameters
    ---------------
    c: np.ndarray
        a n x m matrix  representing the coefficients of all the image blocks.
        n represents the maximum dimension of the PCA space.
        m is (number of images x n_blocks**2)

    D: np.ndarray
        an N x n matrix representing the basis vectors of the PCA space
        N is the dimension of the original space (number of pixels in a block)

    im_num: Integer
        index of the image to visualize

    X_mn: np.ndarray
        a matrix representing the mean block.

    num_coeffs: Iterable
        an iterable with 9 elements representing the number_of coefficients
        to use for reconstruction for each of the 9 plots

    n_blocks: Integer
        number of blocks comprising the image in each direction.
        For example, for a 256x256 image divided into 64x64 blocks, n_blocks will be 4
    '''

    f, axarr = plt.subplots(3, 3)

    print(c.shape, D.shape, n_blocks)
    for i in range(3):
        for j in range(3):
            nc = num_coeffs[i*3+j]
            plot(c[:nc, :], D[:, :nc], im_num, n_blocks, mn, axarr[i, j])
    #f.show()
    f.savefig('output/hw1a_{0}_im{1}.png'.format(n_blocks, im_num), bbox_inches='tight')
    plt.close(f)


def plot_top_16(D, sz, imname):
    '''
    Plots the top 16 components from the basis matrix D.
    Each basis vector represents an image block of shape (sz, sz)

    Parameters
    -------------
    D: np.ndarray
        N x n matrix representing the basis vectors of the PCA space
        N is the dimension of the original space (number of pixels in a block)
        n represents the maximum dimension of the PCA space (assumed to be atleast 16)

    sz: Integer
        The height and width of each block

    imname: string
        name of file where image will be saved.
    '''
    f, axarr = plt.subplots(4, 4)
    for i in range(axarr.shape[0]):
        for j in range(axarr.shape[1]):
            axarr[i, j].imshow(D[:, i*4+j].reshape((sz, sz)), cmap=cm.Greys_r)
            axarr[i, j].axis('off')

    f.savefig(imname, bbox_inches='tight')
    plt.close(f)


def plot(c, D, im_num, n_blocks, X_mn, ax):
    '''
    Plots a reconstruction of a particular image using D as the basis matrix and coeffiecient
    vectors from c

    Parameters
    ------------------------
        c: np.ndarray
            a l x m matrix  representing the coefficients of all blocks in a particular image
            l represents the dimension of the PCA space used for reconstruction
            m represents the number of blocks in an image

        D: np.ndarray
            an N x l matrix representing l basis vectors of the PCA space
            N is the dimension of the original space (number of pixels in a block)

        n_blocks: Integer
            number of blocks comprising the image in each direction.
            For example, for a 256x256 image divided into 64x64 blocks, n_blocks will be 4

        X_mn: basis vectors represent the divergence from the mean so this
            matrix should be added to all reconstructed blocks

        ax: the axis on which the image will be plotted
    '''
    X = None
    for i in range(n_blocks):
        Y = None
        for j in range(n_blocks):
            Y = np.append(Y, np.dot(D, c[:, im_num*n_blocks*n_blocks+i*n_blocks+j]).reshape(X_mn.shape) + X_mn, 1) \
                if Y is not None else np.dot(D, c[:, im_num*n_blocks*n_blocks+i*n_blocks+j]).reshape(X_mn.shape) + X_mn
        X = np.append(X, Y, 0) if X is not None else Y

    ax.imshow(X, cmap=cm.Greys_r)
    ax.axis('off')


if __name__ == '__main__':
    '''
    Read all images(grayscale) from jaffe folder and collapse each image
    to get an numpy array Ims with size (no_images, height*width)
    '''

    fnames = []
    for (_, _, f) in walk('jaffe/'):
        fnames.extend(f)

    fnames.sort()
    I = []
    for filename in fnames:
        try:
            I.append(np.asarray(Image.open("jaffe/{0}".format(filename)).convert('L')))
        except:
            continue

    Ims = np.asarray(I, np.float32)

    szs = [16, 32, 64]
    num_coeffs = [range(1, 10, 1), range(3, 30, 3), range(5, 50, 5)]

    for sz, nc in zip(szs, num_coeffs):
        '''
        Divide here each image into non-overlapping blocks of shape (sz, sz).
        Flatten each block and arrange all the blocks in a
        (no_images*n_blocks_in_image) x (sz*sz) matrix called X
        '''

        images = T.tensor4('images')
        neibs = images2neibs(images, neib_shape=(sz, sz))
        window_function = theano.function([images], neibs)
        X = window_function(Ims.reshape([Ims.shape[0], 1, Ims.shape[1], Ims.shape[2]]))

        X_mn = np.mean(X, 0)
        X = X - np.repeat(X_mn.reshape(1, -1), X.shape[0], 0)

        '''
        Perform eigendecomposition on X^T X and arrange the eigenvectors
        in decreasing order of eigenvalues into a matrix D
        '''

        w, v = np.linalg.eigh(np.dot(X.T, X))

        v = v[:, np.argsort(w)]
        D = np.fliplr(v)

        c = np.dot(D.T, X.T)

        for i in range(0, 200, 10):
            plot_mul(c, D, i, X_mn.reshape((sz, sz)),
                     num_coeffs=nc, n_blocks=int(256/sz))

        plot_top_16(D, sz, imname='output/hw1a_top16_{0}.png'.format(sz))
