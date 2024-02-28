import tensorflow as tf
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

def build_generator(latent_dim):
    model = tf.keras.Sequential([
        layers.Dense(128, activation="relu", input_dim=latent_dim),
        layers.Dense(256, activation="relu"),
        layers.Dense(512, activation="relu"),
        layers.Dense(1024, activation="relu"),
        layers.Dense(28*28, activation="sigmoid"),
        layers.Reshape((28, 28))
    ])
    return model

def build_discriminator(img_shape):
    model = tf.keras.Sequential([
        layers.Flatten(input_shape=img_shape),
        layers.Dense(512, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    return model

def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan = tf.keras.Sequential([
        generator,
        discriminator
    ])
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan

# def load_real_data():
#     (X_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
#     X_train = X_train / 255.0
#     X_train = np.expand_dims(X_train, axis=-1)
#     return X_train

def train(generator, discriminator, gan, dataset, latent_dim, epochs=1000, batch_size=128):
    for epoch in range(epochs):
        # Train the discriminator
        idx = np.random.randint(0, dataset.shape[0], batch_size)
        real_imgs = dataset[idx]
        fake_imgs = generator.predict(np.random.normal(0, 1, (batch_size, latent_dim)))
        
        d_loss_real = discriminator.train_on_batch(real_imgs, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_imgs, np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Train the generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_y = np.array([1] * batch_size)
        g_loss = gan.train_on_batch(noise, valid_y)
        
        if epoch % 1000 == 0:
            print(f"Epoch: {epoch}, D Loss: {d_loss}, G Loss: {g_loss}")
      
def generate_synthetic_data(generator, latent_dim, num_samples):
    noise = np.random.normal(0, 1, (num_samples, latent_dim))
    synthetic_data = generator.predict(noise)
    return synthetic_data


latent_dim = 100
generator = build_generator(latent_dim)
discriminator = build_discriminator((28, 28, 1))
gan = build_gan(generator, discriminator)

# Load real data
dataset = load_real_data()

# Train GAN
train(generator, discriminator, gan, dataset, latent_dim)

# Generate synthetic data
synthetic_data = generate_synthetic_data(generator, latent_dim, 10)
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(synthetic_data[i, :, :, 0], cmap='gray')
    plt.axis
