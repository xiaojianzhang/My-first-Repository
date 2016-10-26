
from __future__ import print_function

__docformat__ = 'restructedtext en'

import cPickle
import gzip
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T

def load_data(dataset):
    
    print('loading data...')
    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.
    
    train_set_x, train_set_y = train_set
    valid_set_x, valid_set_y = valid_set
    test_set_x, test_set_y = test_set
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


class LogisticRgression(object):
    """
    """
    
    def __init__(self, X, y, dim_input, dim_output):
        
        self.W = theano.shared(value=numpy.zeros((dim_input, dim_output)).astype('float64'), name='W')
        self.b = theano.shared(value=numpy.zeros(dim_output).astype('float64'), name='b')
        z = T.dot(X, self.W) + self.b
        
        self.p_y_given_x = T.nnet.softmax(z)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]
        
        self.nll = -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y]) 
        
        self.error = T.mean(T.neq(self.y_pred.astype('int64'), y))
        

def sgd_logistic_regression_mnist(lr=0.13, 
                           n_epochs=1000,
                           dataset='mnist.pkl.gz',
                           batch_size=100):
    
    datasets = load_data('data/mnist.pkl.gz')

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    n_train_batches = train_set_x.shape[0] / batch_size
    n_valid_batches = valid_set_x.shape[0] / batch_size
    n_test_batches = test_set_x.shape[0] / batch_size

    X = T.matrix('X', dtype='float64')
    y = T.vector('y', dtype='int64')
    
    MyLR_classifier = LogisticRgression(X,y,dim_input=train_set_x.shape[1], dim_output=10)

    cost = MyLR_classifier.nll
    gparams = [T.grad(cost, param) for param in MyLR_classifier.params]
    updates = [
        (param, param - lr * gparam)
        for param, gparam in zip(MyLR_classifier.params, gparams)
        ]


    test_model = theano.function(
        inputs=[X,y],
        outputs=MyLR_classifier.error
    )
    validate_model = theano.function(
        inputs=[X,y],
        outputs=MyLR_classifier.error
    )
    train_model = theano.function(
        inputs=[X,y],
        outputs=cost,
        updates=updates
    )


    print('...training the model')
    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant
    validation_frequency = n_train_batches
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(
                train_set_x[minibatch_index*batch_size:(minibatch_index+1)*batch_size,:].astype('float64'),
                train_set_y[minibatch_index*batch_size:(minibatch_index+1)*batch_size].astype('int64')
            )
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(
                    valid_set_x[index*batch_size:(index+1)*batch_size,:].astype('float64'),
                    valid_set_y[index*batch_size:(index+1)*batch_size].astype('int64')    
                    ) for index in xrange(n_valid_batches)]
                
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation accuracy %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        (1.0-this_validation_loss) * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [test_model(
                    test_set_x[index*batch_size:(index+1)*batch_size,:].astype('float64'),
                    test_set_y[index*batch_size:(index+1)*batch_size].astype('int64')    
                    ) for index in xrange(n_valid_batches)]
                    
                    test_score = numpy.mean(test_losses)

                    print(
                        (
                            '     epoch %i, minibatch %i/%i, test accuracy of'
                            ' best model %f %%'
                        ) %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            (1.0-test_score) * 100.
                        )
                    )

                    # save the best model
                    with open('best_model.pkl', 'w') as f:
                        cPickle.dump(MyLR_classifier, f)

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % ((1.0-best_validation_loss) * 100., (1.0-test_score) * 100.)
    )
    print('The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time)))
    print(sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.1fs' % ((end_time - start_time))))

if __name__ == '__main__':
    sgd_logistic_regression_mnist()
