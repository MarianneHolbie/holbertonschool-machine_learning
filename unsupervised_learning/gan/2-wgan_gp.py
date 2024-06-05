#!/usr/bin/env python3
"""
    Define WGAN with gradient penalty
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class WGAN_GP(keras.Model):
    """
        Wasserstein GANs class with gradient penalty
    """

    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=.005, lambda_gp=10):
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
        :param lambda_gp: new hyperparameter for gradient penalty
        """
        super().__init__()  # run the __init__ of keras.Model first.
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = .3  # standard value, but can be changed if necessary
        self.beta_2 = .9  # standard value, but can be changed if necessary

        self.lambda_gp = lambda_gp
        self.dims = self.real_examples.shape
        self.len_dims = tf.size(self.dims)
        self.axis = tf.range(1, self.len_dims, delta=1, dtype='int32')
        self.scal_shape = self.dims.as_list()
        self.scal_shape[0] = self.batch_size
        for i in range(1, self.len_dims):
            self.scal_shape[i] = 1
        self.scal_shape = tf.convert_to_tensor(self.scal_shape)

        # define the generator loss and optimizer:
        self.generator.loss = lambda fake_output: -tf.reduce_mean(fake_output)
        self.generator.optimizer = (
            keras.optimizers.Adam(learning_rate=self.learning_rate,
                                  beta_1=self.beta_1,
                                  beta_2=self.beta_2))
        self.generator.compile(optimizer=generator.optimizer,
                               loss=generator.loss)

        # define the discriminator loss and optimizer:
        self.discriminator.loss = lambda real_output, fake_output: \
            tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
        self.discriminator.optimizer = (
            keras.optimizers.Adam(learning_rate=self.learning_rate,
                                  beta_1=self.beta_1,
                                  beta_2=self.beta_2))
        self.discriminator.compile(optimizer=discriminator.optimizer,
                                   loss=discriminator.loss)

    # generator of real samples of size batch_size
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

    # generator of fake samples of size batch_size
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

    # generator of interpolating samples of size batch_size
    def get_interpolated_sample(self, real_sample, fake_sample):
        """
            generate interpolated sample
        :param real_sample: corpus of real samples
        :param fake_sample: corpus of fake samples
        :return: interpolated sample
        """
        u = tf.random.uniform(self.scal_shape)
        v = tf.ones(self.scal_shape) - u
        return u * real_sample + v * fake_sample

    # computing the gradient penalty
    def gradient_penalty(self, interpolated_sample):
        """
            compute gradient penalty to stabilize training
        :param interpolated_sample: interpolated sample
        :return: value of gradient penalty
        """
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_sample)
            pred = self.discriminator(interpolated_sample, training=True)
        grads = gp_tape.gradient(pred, [interpolated_sample])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=self.axis))

        return tf.reduce_mean((norm - 1.0) ** 2)

        # overloading train_step()

    def train_step(self, useless_argument):
        """
            function to train WGAN_GP
        :param useless_argument: normaly pass data, but here
        data=self.real_example
        :return: training model with loss for generator and discriminator
        following Wasserstein GAN with gradient penalty
        """
        for _ in range(self.disc_iter):
            # compute the penalized loss for the discriminator
            # in a tape watching the discriminator's weights
            with (tf.GradientTape() as disc_tape):
                # get a real sample
                real_sample = self.get_real_sample()
                # get a fake sample
                fake_sample = self.get_fake_sample()
                # get the interpolated sample (between real and fake)
                interpoled_sample = self.get_interpolated_sample(real_sample,
                                                                 fake_sample)

                # compute the old loss discr_loss of the discriminator on real and fake samples
                disc_real_output = self.discriminator(real_sample)
                disc_fake_output = self.discriminator(fake_sample)
                old_discr_loss = self.discriminator.loss(
                    real_output=disc_real_output,
                    fake_output=disc_fake_output)

                # compute the gradient penalty gp
                gp = self.gradient_penalty(interpoled_sample)

                # compute the sum new =discr_loss+self.lambda_gp*gp
                new_discr_loss = old_discr_loss + gp * self.lambda_gp

            # apply gradient descent with respect to new_discr_loss
            # once to the discriminator
            disc_gradients = (
                disc_tape.gradient(new_discr_loss,
                                   self.discriminator.trainable_variables))
            self.discriminator.optimizer.apply_gradients(zip(
                disc_gradients,
                self.discriminator.trainable_variables))

        # compute the loss for the generator in a tape watching
        # the generator's weights
        with tf.GradientTape() as gen_tape:
            # get a fake sample
            fake_sample = self.get_fake_sample(training=True)
            gen_out = self.discriminator(fake_sample, training=False)
            # compute the loss gen_loss of the generator on this sample
            gen_loss = self.generator.loss(gen_out)

        # apply gradient descent to the discriminator
        gen_gradients = gen_tape.gradient(gen_loss,
                                          self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(zip(
            gen_gradients, self.generator.trainable_variables))

        return {"discr_loss": old_discr_loss, "gen_loss": gen_loss, "gp": gp}
