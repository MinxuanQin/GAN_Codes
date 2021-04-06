from keras.layers import Input, Lambda, LSTM, LeakyReLU, Activation, BatchNormalization, Dense, Reshape

from keras.models import Model
from keras.initializers import RandomNormal
from keras.optimizers import Adam, RMSprop
from keras.utils import plot_model

import tensorflow as tf
import numpy as np
import os
import pickle as pkl
import matplotlib.pyplot as plt

class WAVE_GAN():
    def __init__(self
                , input_dim
                , discriminator_lstm_units
                , discriminator_batch_norm_momentum
                , discriminator_activation
                , discriminator_learning_rate
                , generator_initial_dense_layer_size
                , generator_lstm_units
                , generator_batch_norm_momentum
                , generator_activation
                , generator_learning_rate
                , optimizer
                , z_dim):
                self.name = 'wave_gan'

                self.input_dim = input_dim
                self.discriminator_lstm_units = discriminator_lstm_units
                self.discriminator_batch_norm_momentum = discriminator_batch_norm_momentum
                self.discriminator_activation = discriminator_activation
                self.discriminator_learning_rate = discriminator_learning_rate

                self.generator_initial_dense_layer_size = generator_initial_dense_layer_size
                self.generator_lstm_units = generator_lstm_units           
                self.generator_batch_norm_momentum = generator_batch_norm_momentum
                self.generator_activation = generator_activation
                self.generator_learning_rate = generator_learning_rate

                self.optimizer = optimizer
                self.z_dim = z_dim

                self.weight_init = RandomNormal(mean = 0., stddev=0.02)

                self.d_losses = []
                self.g_losses = [] 

                self.epoch = 0

                self._build_discriminator()
                self._build_generator()

                self._build_adversarial()

    def get_activation(self, activation):
        if activation == 'leaky_relu':
            layer = LeakyReLU(alpha=0.2)
        else:
            layer = Activation(activation)
        return layer

    def _build_discriminator(self):

        discriminator_input = Input(shape = self.input_dim, name='discriminator_input')

        x = discriminator_input

        #probieren mit Lambda

        x = Lambda(lambda x:tf.expand_dims(x, axis=-1))(x)

        x = LSTM(self.discriminator_lstm_units, return_sequences=True)(x)

        x = LSTM(self.discriminator_lstm_units)(x)

        if self.discriminator_batch_norm_momentum > 0:
            x = BatchNormalization(momentum=self.discriminator_batch_norm_momentum)(x)

        x = self.get_activation(self.discriminator_activation)(x)

        discriminator_output = Dense(1, activation = 'sigmoid', kernel_initializer=self.weight_init)(x)

        self.discriminator = Model(discriminator_input, discriminator_output)

    def _build_generator(self):

        generator_input = Input(shape = (self.z_dim,), name='generator_input')

        x = generator_input
        '''
        x = Dense(np.prod(self.generator_initial_dense_layer_size))(x)

        if self.generator_batch_norm_momentum:
            x = BatchNormalization(momentum=self.generator_batch_norm_momentum)(x)

        x = self.get_activation(self.generator_activation)(x)

        x = Reshape(self.generator_initial_dense_layer_size)(x)
        '''
        x = Lambda(lambda x:tf.expand_dims(x, axis = -1))(x)

        x = LSTM(self.generator_lstm_units, return_sequences=True)(x)
        x = LSTM(self.generator_lstm_units)(x)

        if self.generator_batch_norm_momentum:
            x = BatchNormalization(momentum=self.generator_batch_norm_momentum)(x)
        x = Activation('tanh')(x)

        x = Dense(self.generator_initial_dense_layer_size, activation='tanh', kernel_initializer=self.weight_init)(x)


        generator_output = x

        self.generator = Model(generator_input, generator_output)

    def get_opti(self, lr):
        if self.optimizer == 'adam':
            opti = Adam(lr=lr, beta_1=0.5)
        elif self.optimizer == 'rmsprop':
            opti = RMSprop(lr=lr)
        else:
            opti = Adam(lr=lr)
        
        return opti

    def set_trainable(self, m, val):
        m.trainable = val
        for l in m.layers:
            l.trainable = val

    def _build_adversarial(self):

        self.discriminator.compile(
            optimizer = self.get_opti(self.discriminator_learning_rate), 
            loss = 'binary_crossentropy',
            metrics = ['accuracy']
        )

        self.set_trainable(self.discriminator, False)

        model_input = Input(shape=(self.z_dim,), name='model_input')
        model_output = self.discriminator(self.generator(model_input))
        self.model = Model(model_input, model_output)

        self.model.compile(
            optimizer = self.get_opti(self.generator_learning_rate),
            loss = 'binary_crossentropy',
            metrics = ['accuracy']
        )

        self.set_trainable(self.discriminator, True)

    def train_discriminator(self, x_train, batch_size, using_generator):

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        if using_generator:
            true_imgs = next(x_train)[0]
            if true_imgs.shape[0] != batch_size:
                true_imgs = next(x_train)[0]
        else:
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            true_imgs = x_train[idx]

        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        gen_imgs = self.generator.predict(noise)

        d_loss_real, d_acc_real = self.discriminator.train_on_batch(true_imgs, valid)
        d_loss_fake, d_acc_fake = self.discriminator.train_on_batch(gen_imgs, fake)

        d_loss = 0.5 * (d_loss_fake + d_loss_real)
        d_acc = 0.5 * (d_acc_fake + d_acc_real)

        return [d_loss, d_loss_real, d_loss_fake, d_acc, d_acc_real, d_acc_fake]

    def train_generator(self, batch_size):
        valid = np.ones((batch_size, 1))
        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        return self.model.train_on_batch(noise, valid)

    def train(self, x_train, batch_size, epochs, run_folder
               , print_every_n_batches = 20
               , using_generator = False):

        for epoch in range(self.epoch, self.epoch + epochs):
            d = self.train_discriminator(x_train, batch_size, using_generator)
            g = self.train_generator(batch_size)

            print("%d [D loss: (%.3f)(R %.3f, F %.3f)] [D acc: (%.3f)(%.3f, %.3f) [G loss: %.3f] [G acc: %.3f]" % (epoch, d[0], d[1], d[2], d[3], d[4], d[5], g[0], g[1]))

            self.d_losses.append(d)
            self.g_losses.append(g)

            if epoch % print_every_n_batches == 0:
                self.sample_images(run_folder)
                self.model.save_weights(os.path.join(run_folder, 'weights/weights-%d.h5' % (epoch)))
                self.model.save_weights(os.path.join(run_folder, 'weights/weights.h5'))
                self.save_model(run_folder)
                
                #new (still debugging)
                #self.save_losses(run_folder)
            
            self.epoch += 1

    def sample_images(self, run_folder):
        r, c = 5, 5   # rows and columns
        noise = np.random.normal(0, 1, (r*c, self.z_dim))
        gen_imgs = self.generator.predict(noise)
        '''
        ## rescaling
        gen_imgs = 0.5 * (gen_imgs + 1)
        gen_imgs = np.clip(gen_imgs, 0, 1)

        fig, axs = plt.subplots(r, c, figsize=(15,15))
        cnt = 0

        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(np.squeeze(gen_imgs[cnt,:,:,:]), cmap = 'gray')
                axs[i, j].axis('off')
                cnt += 1
        '''

        fig, axs = plt.subplots(r,c, figsize = (15,15))
        cnt = 0
        #need change every time!
        t = np.arange(0,5,0.1)

        for i in range(r):
            for j in range(c):
                axs[i,j].plot(t, gen_imgs[cnt,:])
                cnt += 1

        
        fig.savefig(os.path.join(run_folder, "images/sample_%d.png" % self.epoch))
        plt.close()

    def plot_model(self, run_folder):
        plot_model(self.model, to_file=os.path.join(run_folder, 'viz/model.png'), show_shapes=True, show_layer_names=True)
        plot_model(self.discriminator, to_file=os.path.join(run_folder, 'viz/discriminator.png'), show_shapes=True, show_layer_names=True)
        plot_model(self.generator, to_file=os.path.join(run_folder, 'viz/generator.png'), show_shapes=True, show_layer_names=True)

    def save(self, folder):

        ## 以二进制格式打开一个文件只用于写入。如果该文件已存在则打开文件，并从开头开始编辑，即原有内容会被删除。
        # 如果该文件不存在，创建新文件。一般用于非文本文件如图片等。
        with open(os.path.join(folder, 'params.pkl'), 'wb') as f:
            pkl.dump([
                self.input_dim
                , self.discriminator_lstm_units
                , self.discriminator_batch_norm_momentum
                , self.discriminator_activation
                , self.discriminator_learning_rate
                , self.generator_initial_dense_layer_size
                , self.generator_lstm_units
                , self.generator_batch_norm_momentum
                , self.generator_activation
                , self.generator_learning_rate
                , self.optimizer
                , self.z_dim
            ], f)

        self.plot_model(folder)

    def save_model(self, run_folder):
        self.model.save(os.path.join(run_folder, 'model.h5'))
        self.discriminator.save(os.path.join(run_folder, 'discriminator.h5'))
        self.generator.save(os.path.join(run_folder, 'generator.h5'))
        #pkl.dump(self, open(os.path.join(run_folder, 'obj.pkl'), 'wb'))

    def load_weight(self, filepath):
        self.model.load_weights(filepath)

    def save_losses(self, run_folder):
        #loss in green, acc in red
        #generator in -, discriminator in :
        epochs = np.arange(self.epoch)
        fig, axs = plt.subplots(2,2)
        axs[0,0].plot(epochs, self.d_losses[:,0], 'g:')
        axs[0,1].plot(epochs, self.d_losses[:,3], 'r:')
        axs[1,0].plot(epochs, self.g_losses[:,0], 'g-')
        axs[1,1].plot(epochs, self.g_losses[:,1], 'r:')
        fig.savefig(os.path.join(run_folder, "losses_to_%d.png" % self.epoch))
        plt.close()