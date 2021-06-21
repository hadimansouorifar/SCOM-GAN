#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pydot')
import numpy as np


import matplotlib.pyplot as plt

from keras.datasets import cifar10
from keras.models import Sequential, Model
from keras.layers import Input, Dense, LeakyReLU, BatchNormalization, ReLU
from keras.layers import Conv2D, Conv2DTranspose, Reshape, Flatten
from keras.optimizers import Adam
from keras import initializers
from keras.utils import plot_model, np_utils
from keras import backend as K
import scipy as sp
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.utils.vis_utils import plot_model

adam = Adam(lr=0.0004, beta_1=0.5)

def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = np.resize(image, new_shape)
        # store
        images_list.append(new_image)
    return np.asarray(images_list)


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


# load dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 3, 32, 32)
    X_test = X_test.reshape(X_test.shape[0], 3, 32, 32)
    input_shape = (3, 32, 32)
else:
    X_train = X_train.reshape(X_train.shape[0], 32, 32, 3)
    X_test = X_test.reshape(X_test.shape[0], 32, 32, 3)
    input_shape = (32, 32, 3)

classnum=10
num_classes = len(np.unique(y_train))
class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, num_classes)
Y_test = np_utils.to_categorical(y_test, num_classes)

# the generator is using tanh activation, for which we need to preprocess
# the image data into the range between -1 and 1.

X_train = np.float32(X_train)
X_train = (X_train / 255 - 0.5) * 2
X_train = np.clip(X_train, -1, 1)

X_test = np.float32(X_test)
X_test = (X_train / 255 - 0.5) * 2
X_test = np.clip(X_test, -1, 1)

print('X_train reshape:', X_train.shape)
print('X_test reshape:', X_test.shape)


# latent space dimension
latent_dim = 100
adam_lr = 0.0004
adam_beta_1 = 0.5

init = initializers.RandomNormal(stddev=0.02)

# Generator network
generator = Sequential()

# FC: 2x2x512
generator.add(Dense(2*2*512, input_shape=(latent_dim,), kernel_initializer=init))
generator.add(Reshape((2, 2, 512)))
generator.add(BatchNormalization())
generator.add(LeakyReLU(0.2))

# # Conv 1: 4x4x256
generator.add(Conv2DTranspose(256, kernel_size=5, strides=2, padding='same'))
generator.add(BatchNormalization())
generator.add(LeakyReLU(0.2))

# Conv 2: 8x8x128
generator.add(Conv2DTranspose(128, kernel_size=5, strides=2, padding='same'))
generator.add(BatchNormalization())
generator.add(LeakyReLU(0.2))

# Conv 3: 16x16x64
generator.add(Conv2DTranspose(64, kernel_size=5, strides=2, padding='same'))
generator.add(BatchNormalization())
generator.add(LeakyReLU(0.2))

# Conv 4: 32x32x3
generator.add(Conv2DTranspose(3, kernel_size=5, strides=2, padding='same',
                              activation='tanh'))

generator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0004, beta_1=0.5))
generator.summary()


# imagem shape 32x32x3
img_shape = X_train[0].shape

# Discriminator network
discriminator = Sequential()

# Conv 1: 16x16x64
discriminator.add(Conv2D(64, kernel_size=5, strides=2, padding='same',
                         input_shape=(img_shape), kernel_initializer=init))
discriminator.add(LeakyReLU(0.2))

# Conv 2:
discriminator.add(Conv2D(128, kernel_size=5, strides=2, padding='same'))
discriminator.add(BatchNormalization())
discriminator.add(LeakyReLU(0.2))

# Conv 3:
discriminator.add(Conv2D(256, kernel_size=5, strides=2, padding='same'))
discriminator.add(BatchNormalization())
discriminator.add(LeakyReLU(0.2))

# Conv 3:
discriminator.add(Conv2D(512, kernel_size=5, strides=2, padding='same'))
discriminator.add(BatchNormalization())
discriminator.add(LeakyReLU(0.2))

# FC
discriminator.add(Flatten())

# Output


image = Input(shape=(32, 32, 3))

features = discriminator(image)

# first output (name=generation) is whether or not the discriminator
    # thinks the image that is being shown is fake, and the second output
    # (name=auxiliary) is the class that the discriminator thinks the image
    # belongs to.
fake = Dense(1, activation='sigmoid', name='generation')(features)
aux = Dense(classnum+1, activation='softmax', name='auxiliary')(features)
discriminator=Model(image, [fake, aux])




discriminator.summary()

discriminator.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss=['binary_crossentropy','categorical_crossentropy'],loss_weights=[1., 0.1])



# d_g = discriminador(generador(z))
discriminator.trainable = False

inputs = Input(shape=(latent_dim, ))


f = generator(inputs)

    # we only want to be able to train generation for the combined model

fake, aux = discriminator(f)

gan = Model(inputs, [fake, aux])



gan.compile(loss=['binary_crossentropy','categorical_crossentropy' ], optimizer=Adam(lr=0.0004, beta_1=0.5), metrics=['binary_accuracy'],loss_weights=[1., 0.1])


plot_model(generator, to_file='cifar-g-model.png', show_shapes=True, show_layer_names=True)
plot_model(discriminator, to_file='cifar-d-model.png', show_shapes=True, show_layer_names=True)


gan.summary()

epochs = 100
batch_size = 32
smooth = 0.1



d_loss = []
g_loss = []

mfid=[]
for e in range(0,epochs + 1):
    for i in range(len(X_train) // batch_size):

        
        
        yr = np.zeros((batch_size))
        yr[:batch_size] = 1
        yr2 = np.zeros((batch_size,classnum+1))
        yr2[:batch_size,0] = 1

        yf = np.zeros((batch_size))
        yf[:batch_size] = 0
        yf2 = np.zeros((batch_size,classnum+1))
        yf2[:batch_size,(e%classnum)+1] = 1
        
        # Train Discriminator weights
        discriminator.trainable = True
        
        # Real samples
        X_batch = X_train[i*batch_size:(i+1)*batch_size]
        d_loss_real = discriminator.train_on_batch(x=X_batch,
                                                   y=[yr,yr2])
        
        # Fake Samples
        z = np.random.normal(loc=0, scale=1, size=(batch_size, latent_dim))
        X_fake = generator.predict_on_batch(z)
        d_loss_fake = discriminator.train_on_batch(x=X_fake, y=[yf,yf2])
         
        # Discriminator loss
        d_loss_batch = 0.5 * (d_loss_real[0] + d_loss_fake[0])
        
        # Train Generator weights
        discriminator.trainable = False
        yf2 = np.zeros((batch_size,classnum+1))
        yf2[:batch_size,(e%classnum)+1] = 1
        g_loss_batch = gan.train_on_batch(x=z, y=[yr,yf2])
        



       # print(
            #'epoch = %d/%d, batch = %d/%d, d_loss=%.3f, g_loss=%.3f' % (e + 1, epochs, i, len(X_train) // batch_size, d_loss_batch, g_loss_batch),
            #100*' ',
            #end='\r'
        #)


    #model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))
    #images1 = scale_images(X_batch, (299,299,3))
    #images2 = scale_images(X_fake, (299,299,3))
    #images1 = preprocess_input(images1)
    #images2 = preprocess_input(images2)
    #act1 = model.predict(images1)
    #act2 = model.predict(images2)

    act1 = X_batch.reshape(batch_size,3072 )
    act2 = X_fake.reshape(batch_size,3072 )
    temp=get_fid(act1, act2)
    print('epoch ' + str(e)+' fid : ' + str(temp))
    
    mfid.append(temp)
    #d_loss.append(d_loss)
    #g_loss.append(g_loss[0])
    #print('epoch = %d/%d, d_loss=%.3f, g_loss=%.3f' % (e + 1, epochs, d_loss[-1], g_loss[-1]), 100*' ')

    if e % 2 == 0:
        samples = 10
        x_fake = generator.predict(np.random.normal(loc=0, scale=1, size=(samples, latent_dim)))

        for k in range(samples):
            plt.subplot(2, 5, k + 1, xticks=[], yticks=[])
            plt.imshow(((x_fake[k] + 1)* 127).astype(np.uint8))

        plt.tight_layout()
        #plt.show()
    

for i in range(0,100):
    print(mfid[i])


# In[ ]:




