from keras.layers import Input, Dense, Reshape, Flatten, BatchNormalization, Concatenate, Embedding, Dropout
from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.models import Model, load_model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from utils import DataPipe
import numpy as np
import random

img_rows = 32
img_cols = 32
channels = 1
img_shape = (img_rows, img_cols, channels)


def build_generator():
    in_label = Input(shape=(1,))
    x1 = Embedding(29, 200)(in_label)
    x1 = Reshape((200,))(x1)

    noise_shape = (100,)
    in_noise = Input(shape=noise_shape)
    x2 = Dense(206)(in_noise)

    merge = Concatenate(axis=-1)([x1, x2])
    x = Dense(356)(merge)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(1024)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(np.prod(img_shape), activation='tanh')(x)
    out_layer = Reshape(img_shape)(x)
    return Model([in_label, in_noise], out_layer, name='generator')


def build_discriminator():
    in_label = Input(shape=(1,))
    x1 = Embedding(29, 200)(in_label)

    x1 = Reshape((200,))(x1)

    in_image = Input(shape=img_shape)
    x2 = Flatten()(in_image)

    merge = Concatenate(axis=-1)([x1, x2])
    x = Dense(662)(merge)
    x = Dropout(0.1)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(256)(x)
    x = Dropout(0.1)(x)
    x = LeakyReLU(alpha=0.2)(x)
    out_layer = Dense(1, activation='softmax')(x)

    return Model([in_label, in_image], out_layer, name='discriminator')


def train(epochs, start=0, batch_size=128, save_interval=200):
    """
    :param epochs:the number of epochs
    :type epochs: int
    :param batch_size:the batch
    :param save_interval:the number frequency which it will save the model and output images
    """
    optimizer = Adam(0.0002, 0.5)
    try:
        generator = load_model('models/generator_model.h5')
        discriminator = load_model('models/discriminator_model.h5')
        combined = load_model('models/combined_model.h5')
        print("loading...")
    except OSError:
        print("creating new model...")
        discriminator = build_discriminator()
        generator = build_generator()
        discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        generator.compile(loss='binary_crossentropy', optimizer=optimizer)
        discriminator.trainable = False
        l, n = generator.inputs
        i = generator.output
        v = discriminator([l, i])
        combined = Model([l, n], v)
    discriminator.summary()
    generator.summary()
    combined.summary()
    combined.compile(loss='binary_crossentropy', optimizer=optimizer)
    half_batch = int(batch_size / 2)

    dp = DataPipe()
    dp.load_all_images()
    for epoch in range(start, epochs):
        images, labels = dp.get_butch(half_batch)

        noise = np.random.normal(0, 1, (half_batch, 100))
        gen_images = generator.predict([labels, noise])

        d_labels = np.ones((half_batch, 1))
        d_loss_real = discriminator.train_on_batch([labels, images], d_labels)
        d_loss_fake = discriminator.train_on_batch([labels, gen_images], np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise1 = np.random.normal(0, 1, (batch_size, 100))
        valid_y = np.ones((batch_size, 1))  # np.array([1] * batch_size)
        fake_labels = np.random.randint(0, 29, batch_size)
        g_loss = combined.train_on_batch([fake_labels, noise1], valid_y)

        print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

        # If at save interval => save generated image samples
        if epoch % save_interval == 0:
            save_images(epoch, generator)
            generator.save('models/generator_model.h5')
            discriminator.save('models/discriminator_model.h5')
            combined.save('models/combined_model.h5')


def save_images(epoch, generator):
    """
    generate an number/latter image with the generator and save it in the batches_output_images folder
    :param epoch:the number of epochs
    :type epoch: int
    :param generator:the generator model
    :type generator: Model
    """
    n = 4
    noise = np.random.normal(0, 1, (n * n, 100))
    classes = np.array([random.randint(0, 28) for p in range(0, n * n)])

    gen_images = generator.predict([classes, noise])
    gen_images = 0.5 * gen_images + 0.5
    for i in range(n * n):
        plt.subplot(n, n, i + 1)
        plt.axis("off")
        plt.imshow(gen_images[i], cmap='gray')
    filename = f"batches_output_images/generated_plot_epoch-{epoch}.png"
    plt.savefig(filename)
    plt.close()


def generate_letter(x):
    """
    generate an number/latter image with the generator and save it in the new_predictions folder
    """
    if not 29 > x >= 0:
        return None
    generator_ = load_model('models/generator_model.h5')
    noise = np.random.normal(0, 1, (1, 100))
    gen_image = generator_.predict([np.array([x]), noise])
    gen_image = 0.5 * gen_image + 0.5
    plt.axis("off")
    plt.imshow(gen_image[0], cmap='gray')
    plt.savefig(f"new_predictions/p{x}____{random.randint(0, 999)}.png")
    plt.close()



'''
0 #
1 $
2 &
3 @
4 a
5 b
6 c
7 d
8 e
9 f
10 g
11 h
12 i
13 j
14 k
15 l
16 m
17 w
18 p
19 q
20 r
21 s
22 t
23 u
24 v
25 w
26 x
27 y
28 z
'''
