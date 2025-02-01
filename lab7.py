import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

latent_dim = 2
data_dir = "/Users/jedrzej/Desktop/programowanie/pythonProject/venv/img"

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "/Users/jedrzej/Desktop/programowanie/pythonProject/venv/img",
    labels=None,
    label_mode=None,
    color_mode="rgb",
    image_size=(128,128),
    shuffle=True,
    seed=None
)

# normalizacja
normalization_layer = layers.Rescaling(1./255)
dataset = dataset.map(lambda x: (normalization_layer(x), normalization_layer(x)))

# augmentacja danych
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# enkoder
inputs = tf.keras.Input(shape=(128, 128, 3))
x = data_augmentation(inputs)
x = layers.Conv2D(32, 3, activation='relu', padding='same', strides=2)(x)
x = layers.Conv2D(64, 3, activation='relu', padding='same', strides=2)(x)
x = layers.Conv2D(128, 3, activation='relu', padding='same', strides=2)(x)
x = layers.Conv2D(256, 3, activation='relu', padding='same', strides=2)(x)

shape_before_flattening = tf.keras.backend.int_shape(x)[1:]
x = layers.Flatten()(x)
latent = layers.Dense(latent_dim, name='latent_vector')(x)

encoder = models.Model(inputs, latent, name="encoder")
encoder.summary()

# dekoder
latent_inputs = tf.keras.Input(shape=(latent_dim,))
x = layers.Dense(np.prod(shape_before_flattening), activation='relu')(latent_inputs)
x = layers.Reshape(shape_before_flattening)(x)
x = layers.Conv2DTranspose(256, 3, activation='relu', padding='same', strides=2)(x)
x = layers.Conv2DTranspose(128, 3, activation='relu', padding='same', strides=2)(x)
x = layers.Conv2DTranspose(64, 3, activation='relu', padding='same', strides=2)(x)
x = layers.Conv2DTranspose(32, 3, activation='relu', padding='same', strides=2)(x)
outputs = layers.Conv2D(3, 3, activation='sigmoid', padding='same')(x)

decoder = models.Model(latent_inputs, outputs, name="decoder")
decoder.summary()

# autoenkoder
autoencoder_inputs = tf.keras.Input(shape=(128, 128, 3))
encoded = encoder(autoencoder_inputs)
decoded = decoder(encoded)

autoencoder = models.Model(autoencoder_inputs, decoded, name="autoencoder")
autoencoder.summary()

# kompilacja modelu
autoencoder.compile(optimizer='adam', loss='mse')

# trenowanie modelu
autoencoder.fit(dataset, epochs=100)

# rysowanie
for batch in dataset.take(1):
    original_images = batch[0].numpy()
    break

encoded_images = encoder.predict(original_images)
decoded_images = autoencoder.predict(original_images)


def plot_images(original, decoded, n=5):
    plt.figure(figsize=(15, 5))
    for i in range(n):
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(original[i])
        plt.title("Original")
        plt.axis("off")

        ax = plt.subplot(3, n, 2 * n + i + 1)
        plt.imshow(decoded[i])
        plt.title("Decoded")
        plt.axis("off")
    plt.show()

plot_images(original_images, decoded_images)
