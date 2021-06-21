#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!unzip test
get_ipython().system('pip install tqdm')
#Import everything that is needed from Keras library.
from keras.layers import Input, Reshape, Dropout, Dense, Flatten, BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
import glob
#matplotlib will help with displaying the results
import matplotlib.pyplot as plt
#numpy for some mathematical operations
import numpy as np
#PIL for opening,resizing and saving images
from PIL import Image
#tqdm for a progress bar when loading the dataset
from tqdm import tqdm
import scipy as sp
#os library is needed for extracting filenames from the dataset folder.
import os

classnum=3
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


class FaceGenerator:
    #RGB-images: 3-channels, grayscale: 1-channel, RGBA-images: 4-channels
    def __init__(self,image_width,image_height,channels):
        self.image_width = image_width
        self.image_height = image_height

        self.channels = channels

        self.image_shape = (self.image_width,self.image_height,self.channels)

        #Amount of randomly generated numbers for the first layer of the generator.
        self.random_noise_dimension = 100

        #Just 10 times higher learning rate would result in generator loss being stuck at 0.
        optimizer = Adam(0.0002,0.5)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile( loss=['binary_crossentropy','categorical_crossentropy'],loss_weights=[1., 0.2],
        optimizer=optimizer,metrics=["accuracy"])
        self.generator = self.build_generator()

        #A placeholder for the generator input.
        random_input = Input(shape=(self.random_noise_dimension,))

        #Generator generates images from random noise.
        generated_image = self.generator(random_input)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        #Discriminator attempts to determine if image is real or generated
        fake, aux  = self.discriminator(generated_image)

        #Combined model = generator and discriminator combined.
        #1. Takes random noise as an input.
        #2. Generates an image.
        #3. Attempts to determine if image is real or generated.
        self.combined = Model(random_input,[fake, aux ])
        self.combined.compile( loss=['binary_crossentropy','categorical_crossentropy'],loss_weights=[1., 0.2],optimizer=optimizer)

    def get_training_data(self,datafolder):
        print("Loading training data...")

        training_data = []
        #Finds all files in datafolder

        image_list = []
        for filename in glob.glob('*.jpg'): #assuming gif
              
              im=Image.open(filename)
              image = Image.open(filename)
              #Resizes to a desired size.
              image = image.resize((self.image_width,self.image_height),Image.ANTIALIAS)
              #Creates an array of pixel values from the image.
              pixel_array = np.asarray(image)

              training_data.append(pixel_array)
       
        
        
        #training_data is converted to a numpy array
        training_data = np.reshape(training_data,(-1,self.image_width,self.image_height,self.channels))
        return training_data


    def build_generator(self):
        #Generator attempts to fool discriminator by generating new images.
        model = Sequential()

        model.add(Dense(256*4*4,activation="relu",input_dim=self.random_noise_dimension))
        model.add(Reshape((4,4,256)))

        #Four layers of upsampling, convolution, batch normalization and activation.
        # 1. Upsampling: Input data is repeated. Default is (2,2). In that case a 4x4x256 array becomes an 8x8x256 array.
        # 2. Convolution: If you are not familiar, you should watch this video: https://www.youtube.com/watch?v=FTr3n7uBIuE
        # 3. Normalization normalizes outputs from convolution.
        # 4. Relu activation:  f(x) = max(0,x). If x < 0, then f(x) = 0.


        model.add(UpSampling2D())
        model.add(Conv2D(256,kernel_size=3,padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(UpSampling2D())
        model.add(Conv2D(256,kernel_size=3,padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(UpSampling2D())
        model.add(Conv2D(128,kernel_size=3,padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(UpSampling2D())
        model.add(Conv2D(128,kernel_size=3,padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))


        # Last convolutional layer outputs as many featuremaps as channels in the final image.
        model.add(Conv2D(self.channels,kernel_size=3,padding="same"))
        # tanh maps everything to a range between -1 and 1.
        model.add(Activation("tanh"))

        # show the summary of the model architecture
        model.summary()

        # Placeholder for the random noise input
        input = Input(shape=(self.random_noise_dimension,))
        #Model output
        generated_image = model(input)

        #Change the model type from Sequential to Model (functional API) More at: https://keras.io/models/model/.
        return Model(input,generated_image)


    def build_discriminator(self):
        #Discriminator attempts to classify real and generated images
        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.image_shape, padding="same"))
        #Leaky relu is similar to usual relu. If x < 0 then f(x) = x * alpha, otherwise f(x) = x.
        model.add(LeakyReLU(alpha=0.2))

        #Dropout blocks some connections randomly. This help the model to generalize better.
        #0.25 means that every connection has a 25% chance of being blocked.
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        #Zero padding adds additional rows and columns to the image. Those rows and columns are made of zeros.
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.25))
        model.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.25))
        #Flatten layer flattens the output of the previous layer to a single dimension.
        model.add(Flatten())
        #Outputs a value between 0 and 1 that predicts whether image is real or generated. 0 = generated, 1 = real.
        
        input = Input(shape=self.image_shape)


        features = model(input)

        # first output (name=generation) is whether or not the discriminator
        # thinks the image that is being shown is fake, and the second output
        # (name=auxiliary) is the class that the discriminator thinks the image
        # belongs to.
        fake = Dense(1, activation='sigmoid', name='generation')(features)
        aux = Dense(classnum+1, activation='softmax', name='auxiliary')(features)
        discriminator=Model(input, [fake, aux])

        
        
        
        model.summary()

        

        return discriminator

    def train(self, datafolder ,epochs,batch_size,save_images_interval):
        #Get the real images
        training_data = self.get_training_data(datafolder)

        #Map all values to a range between -1 and 1.
        training_data = training_data / 127.5 - 1.

        #Two arrays of labels. Labels for real images: [1,1,1 ... 1,1,1], labels for generated images: [0,0,0 ... 0,0,0]
        
        
        
        mfid=[]
        for epoch in range(epochs):
            yr = np.ones((batch_size,1))
            yr2=np.zeros((batch_size,classnum+1))
            yr2[0:batch_size,0]=1
            yf = np.zeros((batch_size,1))
            yf2=np.zeros((batch_size,classnum+1))
            yf2[0:batch_size,(epoch%classnum)+1]=1
        
        
            ym = np.ones((batch_size,1))
            ym2=np.zeros((batch_size,classnum+1))
            ym2[0:batch_size,(epoch%classnum)+1]=1
        
            # Select a random half of images
            
            indices = np.random.randint(0,len(training_data),batch_size)
            real_images = training_data[indices]

            #Generate random noise for a whole batch.
            random_noise = np.random.normal(0,1,(batch_size,self.random_noise_dimension))
            #Generate a batch of new images.
            generated_images = self.generator.predict(random_noise)

            #Train the discriminator on real images.
            discriminator_loss_real = self.discriminator.train_on_batch(real_images,[yr,yr2])
            #Train the discriminator on generated images.
            discriminator_loss_generated = self.discriminator.train_on_batch(generated_images,[yf,yf2])
            #Calculate the average discriminator loss.
            discriminator_loss = 0.5 * np.add(discriminator_loss_real,discriminator_loss_generated)

            #Train the generator using the combined model. Generator tries to trick discriminator into mistaking generated images as real.
            generator_loss = self.combined.train_on_batch(random_noise,[ym,ym2])
            print ("%d [Discriminator loss: %f, acc.: %.2f%%] [Generator loss: %f]" % (epoch, discriminator_loss[0], 100*discriminator_loss[1], generator_loss[0]))
            
            images1 = scale_images(real_images, (16,16,3))
            images2 = scale_images(generated_images, (16,16,3))

            act1 = images1.reshape(batch_size,768  )
            act2 = images2.reshape(batch_size,768  )
            temp=get_fid(act1, act2)
            print('fid : ' + str(temp))
    
            mfid.append(temp)

            if epoch % save_images_interval == 0:
                self.save_images(epoch)
                

        #Save the model for a later use
        for i in range(0,5000):
            print(mfid[i])
        #self.generator.save("saved_models/facegenerator.h5")

    def save_images(self,epoch):
        #Save 25 generated images for demonstration purposes using matplotlib.pyplot.
        rows, columns = 5, 5
        noise = np.random.normal(0, 1, (rows * columns, self.random_noise_dimension))
        generated_images = self.generator.predict(noise)

        generated_images = 0.5 * generated_images + 0.5

        figure, axis = plt.subplots(rows, columns)
        image_count = 0
        for row in range(rows):
            for column in range(columns):
                axis[row,column].imshow(generated_images[image_count, :], cmap='spring')
                axis[row,column].axis('off')
                image_count += 1
        figure.savefig('./Untitled Folder 1/generated_' + str(epoch)+ '.png')
        #figure.show()
        plt.close()

    def generate_single_image(self,model_path,image_save_path):
        noise = np.random.normal(0,1,(1,self.random_noise_dimension))
        model = load_model(model_path)
        generated_image = model.predict(noise)
        #Normalized (-1 to 1) pixel values to the real (0 to 256) pixel values.
        generated_image = (generated_image+1)*127.5
        print(generated_image)
        #Drop the batch dimension. From (1,w,h,c) to (w,h,c)
        generated_image = np.reshape(generated_image,self.image_shape)

        image = Image.fromarray(generated_image,"RGB")
        #image.save(image_save_path)

if __name__ == '__main__':
    facegenerator = FaceGenerator(64,64,3)
    facegenerator.train(datafolder="test",epochs=5000, batch_size=32, save_images_interval=100)
    #facegenerator.generate_single_image("saved_models/facegenerator.h5","test.png")
    


# In[ ]:




