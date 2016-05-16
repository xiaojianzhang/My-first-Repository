# Install libjpeg-dev, libX11, Pillow
"""
In this problem, you will be dividing the images into blocks of sizes (16; 16); (32; 32);
(64; 64) and performing principal component analysis in each case. For each case you
will visualize reconstructions using different number of principal components and also
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


def plot_mul(c, D, im_num, X_mn, num_coeffs):
    '''
    Plots nine PCA reconstructions of a particular image using number
    of components specified by num_coeffs

    Parameters
    ---------------
    c: np.ndarray
        a n x m matrix  representing the coefficients of all the images
        n represents the maximum dimension of the PCA space.
        m represents the number of images

    D: np.ndarray
        an N x n matrix representing the basis vectors of the PCA space
        N is the dimension of the original space (number of pixels in the image)

    im_num: Integer
        index of the image to visualize

    X_mn: np.ndarray
        a matrix representing the mean image

    num_coeffs: Iterable
        an iterable with 9 elements representing the number_of coefficients
        to use for reconstruction for each of the 9 plots
    '''
    f, axarr = plt.subplots(3, 3)

    for i in range(3):
        for j in range(3):
            nc = num_coeffs[i*3+j]
            cij = c[:nc, im_num]
            Dij = D[:, :nc]
            plot(cij, Dij, X_mn, axarr[i, j])

    f.savefig('output/hw1b_im{0}.png'.format(im_num), bbox_inches='tight')
    plt.close(f)


def plot_top_16(D, sz, imname):
    '''
    Plots the top 16 components from the basis matrix D.
    Each basis vector represents an image of shape (sz, sz)

    Parameters
    -------------
    D: np.ndarray
        an N x n matrix representing the basis vectors of the PCA space
        N is the dimension of the original space (number of pixels in the image)
        n represents the maximum dimension of the PCA space (assumed to be atleast 16)

    sz: Integer
        The height and width of a image

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


def plot(c, D, X_mn, ax):
    '''
    Plots a reconstruction of a particular image using D as the basis matrix and c as
    the coefficient vector
    Parameters
    -------------------
        c: np.ndarray
            a l x 1 vector  representing the coefficients of the image.
            l represents the dimension of the PCA space used for reconstruction

        D: np.ndarray
            an N x l matrix representing first l basis vectors of the PCA space
            N is the dimension of the original space (number of pixels in the image)

        X_mn: basis vectors represent the divergence from the mean so this
            matrix should be added to the reconstructed image

        ax: the axis on which the image will be plotted
    '''

    X = np.dot(D, c).reshape(X_mn.shape) + X_mn

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
    I = None
    for filename in fnames:
        try:
            I = np.append(I, np.asarray(Image.open("jaffe/{0}".format(filename)).convert('L')).reshape((1, -1)), axis=0) \
                if I is not None else np.asarray(Image.open("jaffe/{0}".format(filename)).convert('L')).reshape((1, -1))
        except:
            continue

    Ims = I.astype(np.float32)
    X_mn = np.mean(Ims, 0)
    X = Ims - np.repeat(X_mn.reshape(1, -1), Ims.shape[0], 0)

    '''
    Use theano to perform gradient descent to get top 16 PCA components
    Put them into a matrix D with decreasing order of eigenvalues
    '''

    n_eig = 16

    w = []
    v = []
    vs = []
    XX = T.dmatrix('XX')
    learning_rate = 0.1

    for i in range(n_eig):
        v.append(theano.shared(np.random.randn(X.shape[1]), name='v{0}'.format(i)))
        cost = T.dot(v[-1].T, T.dot(XX.T, T.dot(XX, v[-1])))
        for j in range(i):
            cost = cost - (w[j]*(T.dot(v[-1].T, v[j]))**2)
        cost = -cost
        gc_v = T.grad(cost, v[-1])

        dsc = theano.function([XX], cost, updates = [(v[-1], v[-1]-learning_rate*gc_v)])
        normalize = theano.function([], [], updates=[(v[-1], v[-1] / T.sqrt(T.dot(v[-1].T, v[-1])))])

        normalize()
        max_steps = 200
        cst = [np.inf]
        cst.append(dsc(X).item(0))
        normalize()

        j = 0
        while j<max_steps and cst[-2]-cst[-1]>1e-2:
            cst.append(dsc(X).item(0))
            normalize()
            if j%10 == 0:
                print i, j, cst[-1]
            j+=1

        vs.append(v[-1].get_value())
        w.append(np.dot(vs[-1].T, np.dot(X.T, np.dot(X, vs[-1]))))

    D = None
    for i in range(n_eig):
        D = np.append(D, vs[i].reshape(-1, 1), axis=1) if D is not None else vs[i].reshape(-1, 1)

    c = np.dot(D.T, X.T)

    for i in range(0, 200, 10):
        plot_mul(c, D, i, X_mn.reshape((256, 256)),
                 [1, 2, 4, 6, 8, 10, 12, 14, 16])

    plot_top_16(D, 256, 'output/hw1b_top16_256.png')
