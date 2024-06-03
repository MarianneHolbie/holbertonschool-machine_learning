#!/usr/bin/env python3
"""
    Define WGAN clip class
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class WGAN_clip(keras.Model):

    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=.005):
        """
            class constructor
        :param generator: generator network
        :param discriminator: discriminator network
        :param latent_generator: input (latent vector) for generator
        :param real_examples: real example for training
        :param batch_size: hyperparameter, number sample to generate
        :param disc_iter: number of iter for discriminator in one iter of
        training
        :param learning_rate: learning rate for training
        """
        super().__init__()  # run the __init__ of keras.Model first.
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = .5  # standard value, but can be changed if necessary
        self.beta_2 = .9  # standard value, but can be changed if necessary

        # define the generator loss and optimizer:
        self.generator.loss = lambda fake_output: -tf.reduce_mean(fake_output)
        self.generator.optimizer = (
            keras.optimizers.Adam(learning_rate=self.learning_rate,
                                  beta_1=self.beta_1,
                                  beta_2=self.beta_2))
        self.generator.compile(optimizer=generator.optimizer,
                               loss=generator.loss)

        # define the discriminator loss and optimizer:
        self.discriminator.loss = lambda real_output, fake_output: tf.reduce_mean(fake_output - real_output)
        self.discriminator.optimizer = (
            keras.optimizers.Adam(learning_rate=self.learning_rate,
                                  beta_1=self.beta_1,
                                  beta_2=self.beta_2))
        self.discriminator.compile(optimizer=discriminator.optimizer,
                                   loss=discriminator.loss)

    # generator of fake samples of size batch_size
    def get_fake_sample(self, size=None, training=False):
        """
            function to generate fake sample

        :param size: size of corpus to generate
        :param training: False to obtain sample generated by model
         in eval mode
        :return: corpus of fake sample
        """
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    # generator of real samples of size batch_size
    def get_real_sample(self, size=None):
        """
            function to return corpus real sample
        :param size: size of corpus
        :return: corpus of real sample
        """
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    # overloading train_step()
    def train_step(self, useless_argument):
        """
            function to train GAN
        :param useless_argument: normaly pass data, but here
        data=self.real_example
        :return: training model with loss for generator and discriminator
        following Wasserstein GAN
        """
        for _ in range(self.disc_iter):
            # compute the loss for the discriminator in a tape watching the
            # discriminator's weights
            # get a real sample
            real_sample = self.get_real_sample()
            # get a fake sample
            fake_sample = self.get_fake_sample(training=True)
            # compute the loss discr_loss of the discriminator on real
            # and fake samples
            with tf.GradientTape() as disc_tape:
                disc_real_output = self.discriminator(real_sample, training=True)
                disc_fake_output = self.discriminator(fake_sample, training=True)

                discr_loss = self.discriminator.loss(real_output=disc_real_output,
                                                     fake_output=disc_fake_output)
            # apply gradient descent once to the discriminator
            disc_gradients = (
                disc_tape.gradient(discr_loss,
                                   self.discriminator.trainable_variables))

            self.discriminator.optimizer.apply_gradients(zip(
                disc_gradients,
                self.discriminator.trainable_variables))

            # clip the weights (of the discriminator) between -1 and 1    # <----- new !
            for weight in self.discriminator.trainable_weights:
                weight.assign(tf.clip_by_value(weight,
                                               clip_value_min=-1,
                                               clip_value_max=1))

        # compute the loss for the generator in a tape watching
        # the generator's weights
        with tf.GradientTape() as gen_tape:
            # get a fake sample
            fake_sample = self.get_fake_sample(training=True)
            gen_out = self.discriminator(fake_sample, training=True)

            # compute the loss gen_loss of the generator on this sample
            gen_loss = self.generator.loss(gen_out)

            # apply gradient descent to the discriminator
        gen_gradients = gen_tape.gradient(gen_loss,
                                          self.generator.trainable_variables)

        self.generator.optimizer.apply_gradients(zip(
            gen_gradients, self.generator.trainable_variables))

        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
