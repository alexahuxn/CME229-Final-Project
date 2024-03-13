import numpy as np
import pandas as pd
import torch
from torch import nn
import tensorflow as tf
from keras import layers
import matplotlib.pyplot as plt

#implement the generator
class Generator(nn.Module):
    def __init__(self):
         super().__init__()
         self.model = nn.Sequential(
             nn.Linear(2, 16),
             nn.ReLU(),
             nn.Linear(16, 32),
             nn.ReLU(),
             nn.Linear(32, 2),
         )

    def forward(self, x):
         output = self.model(x)
         return output
     

#implement the discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 256), #the input is two-dimensional
            nn.ReLU(),
            nn.Dropout(0.3), #droput layers reduce overfitting
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),#sigmoid activation to represent probability
        )

    def forward(self, x):
        output = self.model(x)
        return output

#set the training parameters
#prepare th training data
torch.manual_seed(111)
train_data_length = 1024
train_data = torch.zeros((train_data_length, 2))
train_labels = torch.zeros(train_data_length)
train_set = [
    (train_data[i], train_labels[i]) for i in range(train_data_length)
]
plt.plot(train_data[:, 0], train_data[:, 1], ".")
lr = 0.001
num_epochs = 1000
loss_function = nn.BCELoss() 


# def build_generator(latent_dim):
#     model = tf.keras.Sequential([
#         layers.Dense(128, activation="relu", input_dim=latent_dim),
#         layers.Dense(256, activation="relu"),
#         layers.Dense(512, activation="relu"),
#         layers.Dense(1024, activation="relu"),
#         layers.Dense(28*28, activation="sigmoid"),
#         layers.Reshape((28, 28))
#     ])
#     return model

# def build_discriminator(img_shape):
#     model = tf.keras.Sequential([
#         layers.Flatten(input_shape=img_shape),
#         layers.Dense(512, activation="relu"),
#         layers.Dense(256, activation="relu"),
#         layers.Dense(1, activation="sigmoid")
#     ])
#     return model

# def build_gan(generator, discriminator):
#     discriminator.trainable = False
#     gan = tf.keras.Sequential([
#         generator,
#         discriminator
#     ])
#     gan.compile(loss='binary_crossentropy', optimizer='adam')
#     return gan


# def train(generator, discriminator, gan, return_data, latent_dim, epochs=1000, batch_size=128):
#     for epoch in range(epochs):
#         # ---------------------
#         #  Train Discriminator
#         # ---------------------
#         half_batch = batch_size // 2
#         discriminator.trainable = True

#         # Compile the discriminator again if it's trained separately within the loop
#         discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
#         idx = np.random.randint(0, return_data.shape[0], half_batch)
#         real_changes = return_data[idx].astype('float32')

#         # Generate a half batch of new 'Pct Change' values
#         noise = np.random.normal(0, 1, (half_batch, latent_dim)).astype('float32')
#         gen_changes = generator.predict(noise)

#         # Train the discriminator
#         d_loss_real = discriminator.train_on_batch(real_changes, np.ones((half_batch, 1)))
#         d_loss_fake = discriminator.train_on_batch(gen_changes, np.zeros((half_batch, 1)))
#         d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

#         # ---------------------
#         #  Train Generator
#         # ---------------------
        
#         noise = np.random.normal(0, 1, (batch_size, latent_dim)).astype('float32')
#         valid_y = np.array([1] * batch_size)

#         # Train the generator
#         g_loss = gan.train_on_batch(noise, valid_y)

#         # Print progress
#         print(f"Epoch: {epoch}/{epochs}, D Loss: {d_loss}, G Loss: {g_loss}")
        
      
# def generate_synthetic_data(generator, latent_dim, num_samples):
#     noise = np.random.normal(0, 1, (num_samples, latent_dim))
#     synthetic_data = generator.predict(noise)
#     return synthetic_data


# latent_dim = 100
# generator = build_generator(latent_dim)
# discriminator = build_discriminator((28, 28, 1))
# gan = build_gan(generator, discriminator)

# # Load real data
# dataset = pd.read_csv('real_data/googl.csv')
# return_data = dataset['Pct Change']

# # Train GAN
# train(generator, discriminator, gan, return_data, latent_dim)

# Generate synthetic data
# synthetic_data = generate_synthetic_data(generator, latent_dim, 10)
# synthetic_data.to_csv('synthetic_data/gan_data.csv', index=False)

# for i in range(10):
#     plt.subplot(2, 5, i+1)
#     plt.imshow(synthetic_data[i, :, :, 0], cmap='gray')
#     plt.axis
