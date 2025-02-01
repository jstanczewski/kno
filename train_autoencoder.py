import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K

# Wymiar przestrzeni latentnej
latent_dim = 2

# **1. Dataset**
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    ".\\img",
    labels=None,
    label_mode=None,
    color_mode="rgb",
    image_size=(128, 128),
    shuffle=True,
    seed=42  # Ustawienie seed dla powtarzalności
)

# Normalizacja
normalization_layer = layers.Rescaling(1. / 255)
dataset = dataset.map(lambda x: (normalization_layer(x), normalization_layer(x)))

# **2. Transfer Learning - Encoder z ResNet50**
base_model = tf.keras.applications.ResNet50(
    weights="imagenet",  # Transfer learning z ImageNet
    include_top=False,  # Usuwamy warstwy klasyfikacyjne
    input_shape=(128, 128, 3)
)
base_model.trainable = False  # Zamrażamy wagi

# Augmentacja (opcjonalna)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),  # Większa rotacja
    layers.RandomZoom(0.2),      # Większy zoom
    layers.RandomBrightness(0.2) # Dodatkowa zmiana jasności
])

inputs = tf.keras.Input(shape=(128, 128, 3))
x = data_augmentation(inputs)  # Augmentacja przed ResNet50
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)  # Redukcja wymiarów
latent = layers.Dense(latent_dim, name="latent_vector")(x)  # Latent space

encoder = models.Model(inputs, latent, name="encoder")
encoder.summary()

# **3. Dekoder**
latent_inputs = tf.keras.Input(shape=(latent_dim,))
x = layers.BatchNormalization()(latent_inputs)  # Normalizacja latent space
x = layers.Dense(16 * 16 * 256, activation="relu")(x)
x = layers.Reshape((16, 16, 256))(x)
x = layers.Conv2DTranspose(256, 3, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(128, 3, activation="relu", padding="same", strides=2)(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", padding="same", strides=2)(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", padding="same", strides=2)(x)
outputs = layers.Conv2D(3, 3, activation="sigmoid", padding="same")(x)  # Sigmoid dla wartości 0-1

decoder = models.Model(latent_inputs, outputs, name="decoder")
decoder.summary()

# **4. Autoencoder**
autoencoder_inputs = tf.keras.Input(shape=(128, 128, 3))
encoded = encoder(autoencoder_inputs)
decoded = decoder(encoded)

def perceptual_loss(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred)) + (1 - tf.image.ssim(y_true, y_pred, max_val=1.0))

autoencoder = models.Model(autoencoder_inputs, decoded, name="autoencoder")
autoencoder.compile(optimizer='adam', loss=perceptual_loss)
autoencoder.summary()

# **5. Trenowanie modelu**
autoencoder.fit(dataset, epochs=300)
decoder.save("decoder_model.keras")
latent_vectors = encoder.predict(dataset.take(1))
print("Minimalna wartość latent space:", latent_vectors.min())
print("Maksymalna wartość latent space:", latent_vectors.max())

# **6. Testowanie dekodera na losowym punkcie latent space**
generated_image = decoder.predict(np.array([[0.2, 0.2]]))[0]  # Generowanie obrazu

# Wyświetlenie wygenerowanego obrazu
plt.imshow(generated_image)
plt.axis("off")
plt.title("Generated Image from Random Latent Vector")
plt.show()
