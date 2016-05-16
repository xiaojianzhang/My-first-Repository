import os
os.environ['THEANO_FLAGS']='device=gpu0'
import matplotlib
import numpy as np
np.random.seed(123)
import matplotlib.pyplot as plt
import lasagne
import theano
import theano.tensor as T
from spatialtransformerlayer import SpatialTransformerLayer
import timeit

conv = lasagne.layers.Conv2DLayer
pool = lasagne.layers.MaxPool2DLayer
dense = lasagne.layers.DenseLayer
relu = lasagne.nonlinearities.rectify


def test_mnist_stcnn(datadir='',modalname='TC',DIM=60,LEARNING_RATE = 0.001,NUM_EPOCHS=50):

    BATCH_SIZE = 256
    NUM_CLASSES = 10
    mnist_cluttered = "mnist_cluttered_60x60_6distortions.npz"
    from scipy.io import loadmat

    def load_data():
        # data = np.load(mnist_cluttered)
        
        data=loadmat(datadir+'train_'+modalname+'.mat')
        X_train0 = data['train_'+modalname].transpose()
        
        data=loadmat(datadir+'test_'+modalname+'.mat')
        X_test = data['test_'+modalname].transpose()

        data=loadmat(datadir+'train_labels.mat')
        y_train0 = np.squeeze(data['train_labels'])
        
        data=loadmat(datadir+'test_labels.mat')
        y_test = np.squeeze(data['test_labels'])
        
        X_train = X_train0[:50000]
        y_train = y_train0[:50000]
        
        X_valid = X_train0[-10000:]
        y_valid = y_train0[-10000:]
        #print X_train.shape
        
        # reshape for convolutions
        X_train = X_train.reshape((X_train.shape[0], 1, DIM, DIM)).transpose((0,1,3,2))
        X_valid = X_valid.reshape((X_valid.shape[0], 1, DIM, DIM)).transpose((0,1,3,2))
        X_test = X_test.reshape((X_test.shape[0], 1, DIM, DIM)).transpose((0,1,3,2))
        
        print "Train samples:", X_train.shape
        print "Validation samples:", X_valid.shape
        print "Test samples:", X_test.shape
        
        return dict(
                    X_train=X_train.astype(theano.config.floatX),
                    y_train=y_train.astype('int32'),
                    X_valid=X_valid.astype(theano.config.floatX),
                    y_valid=y_valid.astype('int32'),
                    X_test=X_test.astype(theano.config.floatX),
                    y_test=y_test.astype('int32'),
                    num_examples_train=X_train.shape[0],
                    num_examples_valid=X_valid.shape[0],
                    num_examples_test=X_test.shape[0],
                    input_height=X_train.shape[2],
                    input_width=X_train.shape[3],
                    output_dim=10,)
    data = load_data()


    plt.figure(figsize=(7,7))
    plt.imshow(data['X_train'][101].reshape(DIM, DIM), cmap='gray', interpolation='none')
    plt.title('Cluttered MNIST', fontsize=20)
    plt.axis('off')
    plt.show()


    def build_model(input_width, input_height, output_dim,
                    batch_size=BATCH_SIZE):
        ini = lasagne.init.HeUniform()
        l_in = lasagne.layers.InputLayer(shape=(None, 1, input_width, input_height),)
        
        # Localization network
        b = np.zeros((2, 3), dtype=theano.config.floatX)
        b[0, 0] = 1
        b[1, 1] = 1
        b = b.flatten()
        
        loc_l0 = pool(l_in, pool_size=(2, 2))
        loc_l1 = conv(loc_l0, num_filters=20, filter_size=(5, 5), W=ini)
        loc_l2 = pool(loc_l1, pool_size=(2, 2))
        loc_l3 = conv(loc_l2, num_filters=20, filter_size=(5, 5), W=ini)
        loc_l4 = dense(loc_l3, num_units=20, W=lasagne.init.HeUniform('relu'))
        loc_out = dense(loc_l4, num_units=6, b=b, W=lasagne.init.Constant(0.0),nonlinearity=lasagne.nonlinearities.identity)

        # Transformer network
        l_trans1 = SpatialTransformerLayer([l_in, loc_out], ds_rate=2.0)
        print "Transformer network output shape: ", l_trans1.output_shape

        # Classification network
        # 32 ~ 64
        class_l1 = conv(l_trans1,num_filters=48,filter_size=(7, 7),nonlinearity=relu,W=ini)
        class_l2 = pool(class_l1, pool_size=(2, 2),stride=2)
        class_l3 = conv(class_l2,num_filters=48,filter_size=(5, 5),nonlinearity=relu,W=ini)
        class_l4 = pool(class_l3, pool_size=(2, 2),stride=2)
        
        l_out = dense(class_l4,num_units=output_dim,nonlinearity=lasagne.nonlinearities.softmax,W=ini)
        
        return l_out, l_trans1

    model, l_transform = build_model(DIM, DIM, NUM_CLASSES)
    model_params = lasagne.layers.get_all_params(model, trainable=True)


    X = T.tensor4()
    y = T.ivector()

    # training output
    output_train = lasagne.layers.get_output(model, X, deterministic=False)

    # evaluation output. Also includes output of transform for plotting
    output_eval, transform_eval = lasagne.layers.get_output([model, l_transform], X, deterministic=True)

    sh_lr = theano.shared(lasagne.utils.floatX(LEARNING_RATE))
    cost = T.mean(T.nnet.categorical_crossentropy(output_train, y))
    updates = lasagne.updates.adam(cost, model_params, learning_rate=sh_lr)

    train = theano.function([X, y], [cost, output_train], updates=updates)
    eval = theano.function([X], [output_eval, transform_eval])



    def train_epoch(X, y):
        num_samples = X.shape[0]
        num_batches = int(np.ceil(num_samples / float(BATCH_SIZE)))
        costs = []
        correct = 0
        for i in range(num_batches):
            idx = range(i*BATCH_SIZE, np.minimum((i+1)*BATCH_SIZE, num_samples))
            X_batch = X[idx]
            y_batch = y[idx]
            cost_batch, output_train = train(X_batch, y_batch)
            costs += [cost_batch]
            preds = np.argmax(output_train, axis=-1)
            correct += np.sum(y_batch == preds)
        
        return np.mean(costs), correct / float(num_samples)


    def eval_epoch(X, y):
        output_eval, transform_eval = eval(X)
        preds = np.argmax(output_eval, axis=-1)
        acc = np.mean(preds == y)
        return acc, transform_eval


    valid_accs, train_accs, test_accs = [], [], []
    try:
        for n in range(NUM_EPOCHS):
            start_time = timeit.default_timer()
            train_cost, train_acc = train_epoch(data['X_train'], data['y_train'])
            valid_acc, valid_trainsform = eval_epoch(data['X_valid'], data['y_valid'])
            test_acc, test_transform = eval_epoch(data['X_test'], data['y_test'])
            valid_accs += [valid_acc]
            test_accs += [test_acc]
            end_time = timeit.default_timer()
            
            if (n+1) % 20 == 0:
                new_lr = sh_lr.get_value() * 0.7
                print "New LR:", new_lr
                sh_lr.set_value(new_lr.astype(theano.config.floatX))
        
            print "Epoch {0}: Train cost {1}, Train acc {2}, val acc {3}, test acc {4} in {5} min" \
            .format(n, train_cost,train_acc, valid_acc, test_acc,(end_time-start_time)/60.0)
    except KeyboardInterrupt:
        pass

    '''
    plt.figure(figsize=(9,9))
    plt.plot(1-np.array(train_accs), label='Training Error')
    plt.plot(1-np.array(valid_accs), label='Validation Error')
    plt.legend(fontsize=20)
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Error', fontsize=20)
    plt.show()


    
    plt.figure(figsize=(7,14))
    for i in range(3):
        plt.subplot(321+i*2)
        plt.imshow(data['X_test'][i].reshape(DIM, DIM), cmap='gray', interpolation='none')
        if i == 0:
            plt.title('Original 60x60', fontsize=20)
        plt.axis('off')
        plt.subplot(322+i*2)
        plt.imshow(test_transform[i].reshape(DIM//2, DIM//2), cmap='gray', interpolation='none')
        if i == 0:
            plt.title('Transformed 20x20', fontsize=20)
        plt.axis('off')
    plt.tight_layout()
    '''

