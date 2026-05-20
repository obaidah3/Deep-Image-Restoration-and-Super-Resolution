# Walkthrough: Codebase Audit and Comparative Analysis

This walkthrough documents the reverse-engineering steps, mathematical calculations, and audit results performed on the two image restoration codebases:
1. **The Sequential Pipeline (`CNN&SRGAN.py`)**: Symmetric U-Net Denoiser ($3.47\text{M}$ parameters) followed by an optimized 16-ResBlock SRGAN Generator ($1.55\text{M}$ parameters).
2. **The Unified Pipeline (`div2k-esrgan.py`)**: A 23-block RRDBNet Generator ($8.61\text{M}$ parameters) featuring Channel Attention, optimized via a multi-layer VGG, LPIPS, and FFT frequency loss framework against a spatial U-Net Discriminator ($3.50\text{M}$ parameters) with Spectral Normalization.

---

## 1. Summary of Verification Activities

To verify all claims and conclusions in the CVPR report, the following analytical and computational steps were executed:

### A. Architectural Auditing & Reverse-Engineering
* **Line-by-Line Code Parsing**: Reviewed `CNN&SRGAN.py` and `div2k-esrgan.py` to identify every layer configuration, kernel size, stride, padding, normalization, and activation layer.
* **Structural Anomaly Detection**: Identified three major engineering defects:
  1. A **reduced 3-convolution DenseBlock design** in `div2k-esrgan.py` (line 431-447) instead of the standard 5-convolution design, which cuts representational capacity by half.
  2. A **dead-code defect** where `SpatialAttention` (line 414-428) is defined but never instantiated or integrated into the Generator trunk.
  3. A **dynamic dynamic scaling parameter** (`lambda_vgg`) in `CNN&SRGAN.py` (line 399) that is highly volatile, causing training instability during the adversarial phase.

### B. Mathematical Verification of Loss Formulations
* Mapped the exact mathematical formulations of the pixel-level losses (L1 and L2), multi-layer VGG perceptual loss, LPIPS perceptual similarity, Relativistic GAN (RaGAN) with softplus optimization, and FFT-based Frequency loss (real 2D FFT spectral amplitude alignment) directly from their implementations in the source code.

### C. Parameter Footprint Validation
* Designed a custom diagnostic script (`count_params.py`) to instantiate the PyTorch modules exactly as defined in the source code.
* Computed the exact parameter counts to verify the model capacity tables in the report:
  * **UNetAutoencoder**: 3,474,627 trainable parameters
  * **SRGAN Generator**: 1,546,435 trainable parameters
  * **SRGAN Discriminator**: 9,936,705 trainable parameters
  * **ESRGAN Generator (RRDBNet)**: 8,611,971 trainable parameters
  * **ESRGAN Discriminator**: 3,499,329 trainable parameters

---

## 2. Walkthrough of Audited Codebase Structure

### A. The Sequential Pipeline (`CNN&SRGAN.py`)
* **U-Net Denoiser** [L89-147]: Implements a 3-stage symmetric encoder-decoder structure with skip connections via `torch.cat`. Uses `BatchNorm2d` and `LeakyReLU(0.2)`. The output features a `Sigmoid` activation [L130].
* **SRGAN Generator** [L272-291]: Implements 16 residual blocks without BatchNorm to preserve high-frequency details. Upsampling is performed via two sequential `PixelShuffle` stages.
* **SRGAN Discriminator** [L294-320]: Standard PatchGAN-like convolutional encoder followed by an `AdaptiveAvgPool2d(4,4)` and a massive fully connected classifier (`Linear(8192, 1024) -> Linear(1024, 1) -> Sigmoid`), representing a significant parameter bottleneck.
* **VGG Feature Extractor** [L332-353]: Extracts features at `relu2_2` and `relu5_4` layers. Implements ImageNet normalization.

### B. The Unified Pipeline (`div2k-esrgan.py`)
* **Generator (RRDBNet)** [L463-531]: Cascades 23 RRDB blocks, each containing 3 DenseBlock modules with dense skip connections. Integrates Squeeze-and-Excitation Channel Attention (`ChannelAttention`) [L397-412] after the trunk. Features **no output activation**.
* **Discriminator (UNetDiscriminator)** [L559-610]: A fully convolutional spatial U-Net that outputs a single-channel score map for localized patch-level supervision. Features **Spectral Normalization** on all convolutional layers.
* **Degradation Pipeline** [L228-281]: A stochastic, high-order, 2-stage degradation pipeline that applies Gaussian blur, random downsampling (`bicubic`, `bilinear`, or `nearest`), additive Gaussian noise, and JPEG compression recursively.
* **Regularization & Optimizers** [L733-796, 867-888]: Implements the Two-Timescale Update Rule (TTUR) with doubled discriminator learning rate ($\text{lr}_g = 10^{-4}, \text{lr}_d = 20^{-4}$), Cosine Annealing with Warm Restarts, and Exponential Moving Average (EMA) with a cold-start warmup ramp.
