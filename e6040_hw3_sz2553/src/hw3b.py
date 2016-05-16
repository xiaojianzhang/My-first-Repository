import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample

from hw3_utils_sol import shared_dataset, load_data
from hw3_nn_sol import LogisticRegression, HiddenLayer, myMLP, LeNetConvPoolLayer, train_nn


def test_lenet(learning_rate=0.1, n_epochs=1000, nkerns=[16, 512],
               batch_size=200, verbose=False):
    """
    Wrapper function for testing LeNet on SVHN dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer

    :type batch_size: int
    :param batch_szie: number of examples in minibatch.

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to.

    """

    rng = numpy.random.RandomState(23455)

    datasets = load_data()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 3 * 32 * 32)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 3, 32, 32))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (32-5+1 , 32-5+1) = (28, 28)
    # maxpooling reduces this further to (28/3, 28/3) = (9, 9)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 9, 9)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(nkerns[0], 3, 5, 5),
        poolsize=(3, 3)
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (9-7+1, 9-7+1) = (3, 3)
    # maxpooling reduces this further to (3/2, 3/2) = (1, 1)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 1, 1)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 9, 9),
        filter_shape=(nkerns[1], nkerns[0], 7, 7),
        poolsize=(2, 2)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 2 * 2),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 1 * 1,
        n_out=500,
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
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
             n_train_batches, n_valid_batches, n_test_batches,
             n_epochs, verbose)


def test_convnet(learning_rate=0.1, n_epochs=1000, nkerns=[16, 512, 20],
        batch_size=200, verbose=False):
    """
    Wrapper function for testing Multi-Stage ConvNet on SVHN dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer

    :type batch_size: int
    :param batch_szie: number of examples in minibatch.

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to.

    """

    rng = numpy.random.RandomState(23455)

    datasets = load_data()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 3 * 32 * 32)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 3, 32, 32))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (32-5+1 , 32-5+1) = (28, 28)
    # maxpooling reduces this further to (28/3, 28/3) = (9, 9)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 9, 9)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(nkerns[0], 3, 5, 5),
        poolsize=(3, 3)
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (9-4+1, 9-4+1) = (3, 3)
    # maxpooling reduces this further to (3/1, 3/1) = (3, 3)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 3, 3)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 9, 9),
        filter_shape=(nkerns[1], nkerns[0], 4, 4),
        poolsize=(2,2)
    )

    # Combine Layer 0 output and Layer 1 output
    # downsample the first layer output to match the size of the second layer
    # output.
    layer0_output_ds = downsample.max_pool_2d(
            input=layer0.output,
            ds=(3,3),
            ignore_border=False
    )
    # concatenate layer
    layer2_input = T.concatenate([layer1.output, layer0_output_ds], axis=1)

    # Construct the third convolutional pooling layer
    # filtering reduces the image size to (3-3+1, 3-3+1) = (1, 1)
    # maxpooling reduces this further to (3/2, 3/2) = (3, 3)
    # 4D output tensor is thus of shape (batch_size, nkerns[2], 1, 1)
    layer2 = LeNetConvPoolLayer(
        rng,
        input=layer2_input,
        image_shape=(batch_size, nkerns[0] + nkerns[1], 3, 3),
        filter_shape=(nkerns[2], nkerns[0] + nkerns[1], 3, 3),
        poolsize=(1,1)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 1 * 1),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer3_input = layer2.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer3 = HiddenLayer(
        rng,
        input=layer3_input,
        n_in=nkerns[2] * 1 * 1,
        n_out=20,
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    layer4 = LogisticRegression(input=layer3.output, n_in=20, n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer4.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
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


def test_CDNN(learning_rate=0.1, n_epochs=1000, nkerns=[16, 512],
               batch_size=200, verbose=False):
    """
    Wrapper function for testing CNN in cascade with DNN

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer

    :type batch_size: int
    :param batch_szie: number of examples in minibatch.

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to.

    """

    rng = numpy.random.RandomState(23455)

    datasets = load_data()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 3 * 32 * 32)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 3, 32, 32))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (32-5+1 , 32-5+1) = (28, 28)
    # maxpooling reduces this further to (28/3, 28/3) = (9, 9)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 9, 9)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(nkerns[0], 3, 5, 5),
        poolsize=(3, 3)
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (9-7+1, 9-7+1) = (3, 3)
    # maxpooling reduces this further to (3/2, 3/2) = (1, 1)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 1, 1)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 9, 9),
        filter_shape=(nkerns[1], nkerns[0], 7, 7),
        poolsize=(2, 2)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 2 * 2),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer2_input = layer1.output.flatten(2)

    hidden_neurons = [500, 200, 100]
    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 1 * 1,
        n_out=hidden_neurons[0],
        activation=T.tanh
    )

    layer3 = HiddenLayer(
        rng,
        input=layer2.output,
        n_in=hidden_neurons[0],
        n_out=hidden_neurons[1],
        activation=T.tanh
    )

    layer4 = HiddenLayer(
        rng,
        input=layer3.output,
        n_in=hidden_neurons[1],
        n_out=hidden_neurons[2],
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    layer5 = LogisticRegression(input=layer4.output, n_in=hidden_neurons[2],
                                n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer5.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer5.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer5.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = (layer5.params + layer4.params + layer3.params +
              layer2.params + layer1.params + layer0.params)

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
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
             n_train_batches, n_valid_batches, n_test_batches,
             n_epochs, verbose)

if __name__ == "__main__":
    test_lenet(n_epochs=10, verbose=True)
    test_convnet(n_epochs=10, verbose=True)
    test_CDNN()

