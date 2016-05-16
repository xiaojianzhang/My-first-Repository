from __future__ import print_function
import six.moves.cPickle as pickle

from collections import OrderedDict
import copy
import numpy
import os
import random
import timeit

import theano
from theano import tensor as T

from hw4_utils import load_data, contextwin, shuffle, conlleval, check_dir, shared_dataset
from hw4_nn import myMLP, train_nn

import sys
sys.setrecursionlimit(5000)

def gen_parity_pair(nbit, num):
    """
    Generate binary sequences and their parity bits

    :type nbit: int
    :param nbit: length of binary sequence

    :type num: int
    :param num: number of sequences

    """
    numpy.random.seed(100)
    X = numpy.random.randint(2, size=(num,nbit))
    Y = numpy.mod(numpy.sum(X, axis=1), 2)
    return X,Y

#TODO: implement RNN class to learn parity function
class RNN(object):
    
    def __init__(self, nh, nc, nbit):
        """Initialize the parameters for the RNN

        :type nh: int
        :param nh: dimension of the hidden layer

        :type nc: int
        :param nc: number of classes

        :type bs: int
        :param bs: batch size
        """        
        
        # de = 1, ne = 2
        
        # parameters of the model                     
        self.wx = theano.shared(name='wx',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nh))
                                .astype(theano.config.floatX))
        self.wh = theano.shared(name='wh',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nh, nh))
                                .astype(theano.config.floatX))
        self.w = theano.shared(name='w',
                               value=0.2 * numpy.random.uniform(-1.0, 1.0,
                               (nh, nc))
                               .astype(theano.config.floatX))
        self.bh = theano.shared(name='bh',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.b = theano.shared(name='b',
                               value=numpy.zeros(nc,
                               dtype=theano.config.floatX))
        self.h0 = theano.shared(name='h0',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))

        # bundle
        self.params = [self.wx, self.wh, self.w,
                       self.bh, self.b, self.h0]

        x = T.dvector()
        y = T.ivector('y')  # labels

        def recurrence(x_t, h_tm1):
            h_t = T.nnet.sigmoid(T.dot(x_t, self.wx)
                                 + T.dot(h_tm1, self.wh) + self.bh)
            s_t = T.nnet.softmax(T.dot(h_t, self.w) + self.b)
            return [h_t, s_t]

        [h, s], _ = theano.scan(fn=recurrence,
                                sequences=x,
                                outputs_info=[self.h0, None],
                                n_steps=x.shape[0])
        # squeeze the singleton dimension, same as T.squeeze(s)
        # p_y_given_x = s[:,0,-1]
        
        p_y_given_x = T.squeeze(s)
        
        y_pred = T.argmax(T.squeeze(s[-1]))
        
        # cost and gradients and learning rate
        lr = T.scalar('lr')

        nll = -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]),y])
        gradients = T.grad(nll, self.params)
        updates = OrderedDict((p, p - lr*g)
                               for p, g in
                               zip(self.params, gradients))

        # theano functions to compile
        self.classify = theano.function(inputs=[x], outputs=y_pred)
        self.train_sub = theano.function(inputs=[x, y, lr],
                                              outputs=nll,
                                              updates=updates)
                                                  
    def train(self, x, y, learning_rate):
        x=numpy.asarray(x).astype(dtype=numpy.float64)
        self.train_sub(x, y, learning_rate)


#TODO: build and train a MLP to learn parity function
def test_mlp_parity(nbits=8,learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=100,
                    batch_size=20, n_hidden=200, n_hiddenLayers=3,verbose=False):
    # generate datasets
    train_set = gen_parity_pair(nbits, 1000)
    valid_set = gen_parity_pair(nbits, 500)
    test_set  = gen_parity_pair(nbits, 100)

    # Convert raw dataset to Theano shared variables.
    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)
    
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(1234)

    classifier = myMLP(rng, input=x, n_in=nbits, n_hidden=n_hidden, n_out=2, n_hiddenLayers=n_hiddenLayers)

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    train_nn(train_model, validate_model, test_model,
        n_train_batches, n_valid_batches, n_test_batches, n_epochs, verbose)
    
#TODO: build and train a RNN to learn parity function
def test_rnn_parity(**kwargs):
    """
    Wrapper function for training and testing RNN

    :type lr: float
    :param lr: learning rate used (factor for the stochastic gradient.

    :type nepochs: int
    :param nepochs: maximal number of epochs to run the optimizer.

    :type nhidden: int
    :param n_hidden: number of hidden units.

    :type emb_dimension: int
    :param emb_dimension: dimension of word embedding.

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to.

    :type decay: boolean
    :param decay: decay on the learning rate if improvement stop.

    """
    # process input arguments
    param = {
        'nbit':8,
        'lr': 0.1,
        'verbose': True,
        'decay': True,
        'nhidden': 15,
        'nepochs': 60}
    param_diff = set(kwargs.keys()) - set(param.keys())
    if param_diff:
        raise KeyError("invalid arguments:" + str(tuple(param_diff)))
    param.update(kwargs)

    if param['verbose']:
        for k,v in param.items():
            print("%s: %s" % (k,v))

    # load the dataset
    # print('... loading the dataset')
    # generate datasets
    train_set = gen_parity_pair(param['nbit'], 1000)
    valid_set = gen_parity_pair(param['nbit'], 500)
    test_set  = gen_parity_pair(param['nbit'], 100)    

    train_set_x = train_set[0]
    train_set_y = train_set[1]
    valid_set_x = valid_set[0]
    valid_set_y = valid_set[1]
    test_set_x = test_set[0]
    test_set_y = test_set[1]

    # instanciate the model
    # print('... building the model')
    rnn = RNN(
        nh=param['nhidden'],
        nc=2,
        nbit=param['nbit'])

    # train with early stopping on validation set
    # print('... training')
    best_val_acc0 = -numpy.inf
    param['clr'] = param['lr']
    for e in range(param['nepochs']):
        param['ce'] = e
        for i, (x, y) in enumerate(zip(train_set_x, train_set_y)):
            labels=numpy.zeros(x.shape[0]).astype('int32')
            labels[-1]=y
            for ky in range(2,x.shape[0]+1):
                labels[-ky]=numpy.mod((labels[-(ky-1)]+x[-(ky-1)]),2)
            rnn.train(x, labels, param['clr'])

        # evaluation // back into the real world : idx -> words
        predictions_test = numpy.asarray([rnn.classify(numpy.asarray(x).astype(theano.config.floatX))
                            for x in test_set_x])
        predictions_valid = numpy.asarray([rnn.classify(numpy.asarray(x).astype(theano.config.floatX))
                             for x in valid_set_x])
        # print(predictions_test)
                              
        best_val_acc = sum(predictions_valid==valid_set_y)/float(len(valid_set_y))*100.0
        if best_val_acc > best_val_acc0:

            best_rnn = copy.deepcopy(rnn)
            best_val_acc0 = best_val_acc #sum(modeval==valid_set_y)/float(len(valid_set_y))*100.0
            test_acc=sum(predictions_test==test_set_y)/float(len(test_set_y))*100.0
            if param['verbose']:
                print('NEW BEST: epoch', e,
                      'valid accuracy%', best_val_acc,
                      'test accuracy%', test_acc)

            param['vf1'], param['tf1'] =best_val_acc, test_acc
            param['be'] = e
            # print(predictions_test)
        else:
            if param['verbose']:
                print('')

        # learning rate decay if no improvement in 10 epochs
        if param['decay'] and abs(param['be']-param['ce']) >= 10:
            param['clr'] *= 0.5
            rnn = best_rnn

        if param['clr'] < 1e-5:
            break

    print('BEST RESULT: epoch', param['be'],
           'valid accuracy', param['vf1'],
           'test accuracy', param['tf1'])
    

if __name__ == '__main__':
    test_mlp_parity()
    test_rnn_parity()