#!/usr/bin/env python3
"""
    Function to train transformer
"""
import tensorflow as tf

Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
        Custom learning rate schedule : learning rate function
        described in the original Transformer paper
        LR increased linearly for the first "warmup_steps"
        training steps, and then decreased proportionally to
        the inverse square root of the step number

        Args:
            d_model (int): dimensionality of the model
            warmup_steps (int) number of steps talen to increase
            the lr linearly.

        Attributes:
            d_model (float) dimensionality of the model as float
            warmup_steps (int) number of steps talen to increase
            the lr linearly.

        Methods:
            __call__(step): returns the learning rate at the given step

        Returns:
            The learning rate at the given step
    """

    def __init__(self, d_model, warmup_steps):
        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        """
            return learning rate at the given step

        :param step: int, current training step

        :return: learning rate at the given step as float32
        """
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """
        creates and trains a transformer model for machine translation
        of Portuguese to English

    :param N: number of blocks in encoder/decoder
    :param dm: dimensionality of the model
    :param h: number of heads
    :param hidden: number of hidden units in the fully connected layers
    :param max_len: max number of tokens per sequence
    :param batch_size: batch size for training
    :param epochs: number of epochs to train for

    :return: trained model
    """

    data = Dataset(batch_size, max_len)
    train_data, val_data = data.data_train, data.data_valid

    transformer = Transformer(
        N=N,
        dm=dm,
        h=h,
        hidden=hidden,
        input_vocab=data.tokenizer_pt.vocab_size + 2,
        target_vocab=data.tokenizer_en.vocab_size + 2,
        max_seq_input=max_len,
        max_seq_target=max_len
    )

    # calculate learning rate following original description
    learning_rate = CustomSchedule(d_model=dm, warmup_steps=4000)

    optimizer = tf.keras.optimizers.Adam(learning_rate,
                                         beta_1=0.9,
                                         beta_2=0.98,
                                         epsilon=1e-9)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction='none'
    )
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

    # training step function
    @tf.function
    def train_step(inp, tar):

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar)

        with tf.GradientTape() as tape:
            predictions, _ = transformer(inp, tar, True, enc_padding_mask, combined_mask, dec_padding_mask)
            loss = loss_function(tar, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_loss(loss)
        train_accuracy(tar, predictions)

    for epoch in range(epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()

        for (batch, (inp, tar)) in enumerate(data.data_train):
            train_step(inp, tar)

            if batch % 50 == 0:
                print(
                    f'Epoch {epoch + 1}, Batch {batch}: Loss {train_loss.result()}, Accuracy {train_accuracy.result()}')

        print(f'Epoch {epoch + 1}: Loss {train_loss.result()}, Accuracy {train_accuracy.result()}')

    return transformer
