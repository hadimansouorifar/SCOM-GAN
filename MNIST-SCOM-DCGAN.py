#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install pydot')
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Flatten, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D, Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import random
#from tqdm import tqdm_notebook
import math
from keras import layers
from keras.utils.vis_utils import plot_model
import scipy as sp
#from tqdm import tqdm_notebook
from math import floor
from numpy import ones
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import std
from numpy import exp
from numpy import resize
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input

classnum=15
# assumes images have the shape 299x299x3, pixels in [0,255]
def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape)
        # store
        images_list.append(new_image)
    return np.asarray(images_list)
def calculate_inception_score(images, n_split=10, eps=1E-16):
    # load inception v3 model
    model = InceptionV3()
    # convert from uint8 to float32
    processed = images.astype('float32')
    # pre-process raw images for inception v3 model
    processed = preprocess_input(processed)
    # predict class probabilities for images
    yhat = model.predict(processed)
    # enumerate splits of images/predictions
    scores = list()
    n_part = floor(images.shape[0] / n_split)
    for i in range(n_split):
        # retrieve p(y|x)
        ix_start, ix_end = i * n_part, i * n_part + n_part
        p_yx = yhat[ix_start:ix_end]
        # calculate p(y)
        p_y = expand_dims(p_yx.mean(axis=0), 0)
        # calculate KL divergence using log probabilities
        kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
        # sum over classes
        sum_kl_d = kl_d.sum(axis=1)
        # average over images
        avg_kl_d = mean(sum_kl_d)
        # undo the log
        is_score = exp(avg_kl_d)
        # store
        scores.append(is_score)
    # average across images
    is_avg, is_std = mean(scores), std(scores)
    return is_avg, is_std
# calculate frechet inception distance
def get_fid(act1, act2):
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sp.linalg.sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid
# Dataset of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images.
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# print(X_train.shape)

z_dim = 100

X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

X_train = X_train[0:30000]

nch = 20
g_input = Input(shape=[100])
H1 = Dense(nch * 14 * 14, init='glorot_normal')(g_input)
H = BatchNormalization()(H1)
H = Activation('relu')(H)
H = Reshape([nch, 14, 14])(H)
H = UpSampling2D(size=(2, 2))(H)
H = Convolution2D(int(nch / 2), 3, 3, border_mode='same', init='glorot_uniform')(H)
H = BatchNormalization()(H)
H = Activation('relu')(H)
H = Convolution2D(int(nch / 4), 3, 3, border_mode='same', init='glorot_uniform')(H)
H = BatchNormalization()(H)
H = Activation('relu')(H)
H = Convolution2D(1, 1, 1, border_mode='same', init='glorot_uniform')(H)
g_V = Activation('sigmoid')(H)
generator = Model(g_input, g_V)
generator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
generator.summary()

# Generator
adam = Adam(lr=0.0002, beta_1=0.5)

g = Sequential()
layer = g.add(Dense(7 * 7 * 112, input_dim=z_dim))
g.add(Reshape((7, 7, 112)))
g.add(BatchNormalization())
g.add(Activation(LeakyReLU(alpha=0.2)))
g.add(Conv2DTranspose(56, 5, strides=2, padding='same'))
g.add(BatchNormalization())
g.add(Activation(LeakyReLU(alpha=0.2)))
g.add(Conv2DTranspose(1, 5, strides=2, padding='same', activation='sigmoid'))
g.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
g.summary()

d = Sequential()
d.add(Conv2D(56, 5, strides=2, padding='same', input_shape=(28, 28, 1), activation=LeakyReLU(alpha=0.2)))
d.add(Conv2D(112, 5, strides=2, padding='same'))
g.add(BatchNormalization())
g.add(Activation(LeakyReLU(alpha=0.2)))
d.add(Conv2D(224, 5, strides=2, padding='same'))
g.add(Activation(LeakyReLU(alpha=0.2)))
d.add(Flatten())
d.add(Dense(112, activation=LeakyReLU(alpha=0.2)))

input = Input([28,28,1])



features = d(input)

fake = Dense(1, activation='sigmoid', name='generation')(features)
aux = Dense(classnum+1, activation='softmax', name='auxiliary')(features)
discriminator=Model(input, [fake, aux])
discriminator.compile(loss=['binary_crossentropy','categorical_crossentropy'],loss_weights=[1., 0.4], optimizer=adam, metrics=['accuracy'])
discriminator.summary()
plot_model(discriminator, to_file='mnist-discriminator.png', show_shapes=True, show_layer_names=True)
discriminator.trainable = False
inputs = Input(shape=(z_dim,))
hidden = g(inputs)
fake, aux = discriminator(hidden)
gan = Model(inputs, [fake, aux])

gan.compile(loss=['binary_crossentropy','categorical_crossentropy'],loss_weights=[1., 0.4], optimizer=adam, metrics=['accuracy'])
gan.summary()


def plot_loss(losses):
    """
    @losses.keys():
        0: loss
        1: accuracy
    """
    d_loss = [v[0] for v in losses["D"]]
    g_loss = [v[0] for v in losses["G"]]

    plt.figure(figsize=(6.4, 4.8))
    plt.plot(d_loss, color='red', label="Discriminator loss")
    plt.plot(g_loss, color='green', label="Generator loss")
    plt.title("GAN : MNIST dataset")

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss.png')
    # plt.show()


def plot_generated(n_ex=20, dim=(2, 10), figsize=(48, 8)):
    noise = np.random.normal(0, 1, size=(n_ex, z_dim))
    generated_images = g.predict(noise)
    generated_images = generated_images.reshape(generated_images.shape[0], 28, 28)
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generated_images[i, :, :], interpolation='nearest', cmap='gray_r')
        
        sss = str(i)

        plt.imsave(sss, generated_images[i, :, :], cmap='gray_r')

        plt.axis('off')
    plt.tight_layout()
    plt.plot()
    plt.show()


# Set up a vector (dict) to store the losses
losses = {"D": [], "G": []}
samples = []


mfid=[]
def train(d, epochs=1, plt_frq=1, BATCH_SIZE=128):
    batchCount = int(X_train.shape[0] / BATCH_SIZE)
    #batchCount=100
    print('Epochs:', epochs)
    print('Batch size:', BATCH_SIZE)
    print('Batches per epoch:', batchCount)

    d_v = []
    for e in range(1, epochs + 1):
        if e == 1 or e % plt_frq == 0:
            print('-' * 15, 'Epoch %d' % e, '-' * 15)
        for _ in range(batchCount):  # tqdm_notebook(range(batchCount), leave=False):
            # Create a batch by drawing random index numbers from the training set
            image_batch = X_train[np.random.randint(0, X_train.shape[0], size=BATCH_SIZE)]
            image_batch = image_batch.reshape(image_batch.shape[0], image_batch.shape[1], image_batch.shape[2], 1)
            # print(image_batch.shape)
            # Create noise vectors for the generator
            noise = np.random.normal(0, 1, size=(BATCH_SIZE, z_dim))

            # Generate the images from the noise
            generated_images = g.predict(noise)
            samples.append(generated_images)
            X = np.concatenate((image_batch, generated_images))
            # Create labels
            y = np.zeros(2 * BATCH_SIZE)
            y[:BATCH_SIZE] = 0.9  # One-sided label smoothing
            
            ys = np.zeros((2 * BATCH_SIZE,classnum+1))
            ys[:BATCH_SIZE,0] = 1
            ys[BATCH_SIZE:2*BATCH_SIZE,(e%classnum)+1] = 1

            # Train discriminator on generated images
            discriminator.trainable = True
            d_loss = discriminator.train_on_batch(X, [y,ys])
            # Train generator
            noise = np.random.normal(0, 1, size=(BATCH_SIZE, z_dim))
            y2 = np.ones(BATCH_SIZE)
            ys2 = np.zeros((BATCH_SIZE,classnum+1))
            
            ys2[:BATCH_SIZE,(e%classnum)+1] = 1
            discriminator.trainable = False
            g_loss = gan.train_on_batch(noise, [y2,ys2])
            weights = []
            ccc = 0

        #weights = g.layers[0].get_weights()[0]
        #w = 3
        #weights = weights.reshape(548800)
        #bin = [-0.08, -0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06, 0.08]

        #px,py,_ = plt2.hist(weights, bins=100)
        #print(str(np.argmax(px)) + '--' + str(np.max(py)))

        sss = str(e)
        # plt2.savefig("plt" +sss +".png")
        # plt2.clf()
        #print(len(weights))

        # Only store losses from final batch of epoch
        #images1 = scale_images(generated_images, (299,299,3))
        #print(calculate_inception_score(images1))
        
        
        image_batch = image_batch.reshape(BATCH_SIZE,784)
        generated_images = generated_images.reshape(BATCH_SIZE,784)
        temp=get_fid(generated_images, image_batch)
        print('fid : ' + str(temp))
        
        mfid.append(temp)
        losses["D"].append(d_loss)
        losses["G"].append(g_loss)

        # Update the plots
        if e == 1 or e % plt_frq == 0:
            plot_generated()
    # print((weights.shape ))
    

    # for fff in range(0,100):
    # print(losses["D"][fff][0])
    # print("-------")
    # for fff in range(0,100):
    # print(losses["G"][fff][0])


train(d, epochs=100, plt_frq=20, BATCH_SIZE=128)
for i in range(0,100):
            print(losses["G"][i][0])


# In[ ]:




