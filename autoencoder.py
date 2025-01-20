import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

from skimage.metrics import structural_similarity as ssim

# Load the Fashion MNIST dataset
(x_train, _), (x_test, _) = fashion_mnist.load_data()

# Normalize the data to the range [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Visualize the first 4 training images
plt.figure()
for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.imshow(x_train[i], cmap="gray")
    plt.axis("off")
plt.show()

# Flatten the data for use in the autoencoder
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# Define the input dimension and encoding dimension
input_dim = x_train.shape[1]  # Size of input layer (28*28 = 784)
encoding_dim = 64  # Size of encoding layer

# Build the autoencoder architecture
input_image = Input(shape=(input_dim,))

# Encoder layers
encoded = Dense(512, activation="relu")(input_image)
encoded = Dense(256, activation="relu")(encoded)
encoded = Dense(128, activation="relu")(encoded)
encoded = Dense(encoding_dim, activation="relu")(encoded)

# Decoder layers
decoded = Dense(128, activation="relu")(encoded)
decoded = Dense(256, activation="relu")(decoded)
decoded = Dense(512, activation="relu")(decoded)
decoded = Dense(input_dim, activation="sigmoid")(decoded)

# Define the autoencoder model
autoencoder = Model(input_image, decoded)

# Compile the autoencoder
autoencoder.compile(optimizer=Adam(), loss="binary_crossentropy")

# Train the autoencoder
history = autoencoder.fit(
    x_train,
    x_train,
    epochs=10,
    batch_size=64,
    shuffle=True,
    validation_data=(x_test, x_test),
    verbose=1
)

# Split the autoencoder into encoder and decoder
# Define the encoder
encoder = Model(input_image, encoded)

# Define the decoder
encoded_input = Input(shape=(encoding_dim,))
decoder_layer1 = autoencoder.layers[-4](encoded_input)
decoder_layer2 = autoencoder.layers[-3](decoder_layer1)
decoder_layer3 = autoencoder.layers[-2](decoder_layer2)
decoder_output = autoencoder.layers[-1](decoder_layer3)

decoder = Model(encoded_input, decoder_output)

# Encode and decode the test images
encoded_images = encoder.predict(x_test)
decoded_images = decoder.predict(encoded_images)

# Visualize original and reconstructed images
n = 10  # Number of images to display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Original image
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Reconstructed (decoded) image
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_images[i].reshape(28, 28), cmap="gray")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

# Define a function to compute SSIM (Structural Similarity Index)
def compute_ssim(original, reconstructed):
    original = original.reshape(28, 28)
    reconstructed = reconstructed.reshape(28, 28)
    return ssim(original, reconstructed, data_range=1)

# Compute SSIM for the first 100 test images
ssim_scores = []
for i in range(100):
    original_image = x_test[i]
    reconstructed_image = decoded_images[i]
    score = compute_ssim(original_image, reconstructed_image)
    ssim_scores.append(score)

# Calculate the average SSIM score
average_ssim = np.mean(ssim_scores)
print("SSIM: ", average_ssim)
