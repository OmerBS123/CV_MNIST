from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt


# load train and test dataset
def load_dataset():
    # load dataset
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    # reshape dataset to have a single channel
    train_x = train_x.reshape((train_x.shape[0], 28, 28, 1))
    test_x = test_x.reshape((test_x.shape[0], 28, 28, 1))
    # one hot encode target values
    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)
    return train_x, train_y, test_x, test_y


def show_data():
    # load dataset
    (trainX, trainy), (testX, testy) = mnist.load_data()
    # summarize loaded dataset
    print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
    print('Test: X=%s, y=%s' % (testX.shape, testy.shape))
    # plot first few images
    for i in range(9):
        # define subplot
        plt.subplot(330 + 1 + i)
        # plot raw pixel data
        plt.imshow(trainX[i], cmap=plt.get_cmap('gray'))
    # show the figure


def prep_pixels(train, test):
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images
    return train_norm, test_norm
