# Autoencoder with Fashion MNIST

This project implements an autoencoder to compress and reconstruct images from the Fashion MNIST dataset. The autoencoder is built using TensorFlow/Keras and visualizes the reconstruction quality. Additionally, the project calculates the Structural Similarity Index (SSIM) to evaluate the performance of the model.

## Features
- Data preprocessing for the Fashion MNIST dataset.
- Autoencoder with dense layers for encoding and decoding.
- Visualization of original and reconstructed images.
- SSIM computation for evaluating reconstruction quality.

## Prerequisites

Ensure you have the following packages installed:

- `numpy`
- `matplotlib`
- `tensorflow`
- `scikit-image`

You can install the necessary packages using pip:
```bash
pip install numpy matplotlib tensorflow scikit-image
```

## Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/KonyD/fashion-mnist-autoencoder.git
   cd fashion-mnist-autoencoder
   ```

2. Run the Python script:
   ```bash
   python autoencoder_fashion_mnist.py
   ```

3. The script will display:
   - Sample original and reconstructed images.
   - Average SSIM score printed in the console.

## Code Overview

- **Data Preprocessing**:
  - Normalizes the Fashion MNIST images to a range of [0, 1].
  - Reshapes the dataset for compatibility with dense layers.

- **Model Architecture**:
  - Encoder: Compresses the input images into a lower-dimensional latent space.
  - Decoder: Reconstructs the images from the compressed latent space.

- **Evaluation**:
  - Visualizes original vs. reconstructed images.
  - Calculates SSIM scores for evaluating reconstruction quality.

## Example Output

- **Visualization of Original and Reconstructed Images**:

  ![Example Images](./Figure%202025-01-20%20115328.png)  

- **SSIM Score**:
  ```bash
  SSIM:  0.7793266204555716  # Your results may vary
  ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset: [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)
- Libraries: TensorFlow, Keras, NumPy, Matplotlib, Scikit-Image

Feel free to contribute or raise issues if you encounter any problems!
