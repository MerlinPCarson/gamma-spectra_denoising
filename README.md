

# Gamma-Spectra denoising
### by Merlin Carson
Algorithms for removing background, Compton scatter and detector noise from gamma-ray spectra.

This repo contains python scripts for simulating clean and noisy gamma-ray spectra, trainable Convolutional Neural Networks for denoising gamma-ray spectra, and a convex optimization based radionuclide classifier. The training script trains two variants of the denoising algorithm, one that predicts a noise mask that is then subtracted from the noisy spectra in post processing and one that directly predicts the denoised spectra. For each of these two variants there are two versions of Convolutional Neural Network models that can be used. The first model, a traditional Convolution Neural Network with a configurable number of layers and the second, a variant of the first with residual connections between every other layer, allowing for the training of ultra-deep (20+ layer) models.  

# Requirements
* Pytorch
* Sklearn
* Scikit-Image
* cvxpy
* h5py

# Scripts
- gen_templates.py 

  Generate templates from NNDC tables based on radionuclides and detector parameters in configuration file (default=config.json). Creates h5 file (default=data/templates.h5) use --savefigs arguments to generate plots of all templates.
  
- load_templates.py

  Load template data file (default=data/templates.h5) and displays all data.
  
- spectra_utils.py

  Helper functions for gamma-spectra processing.
 
- compton.py
  
  Functions for predicting Compton scatter.
  
- build_dataset.py
  
  Generates training data with varying level of noise and Compton scatter along with target data which is a radionuclide's template.
  
- load_data.py
 
  Loads and views a training HD5F file created by build_dataset.
  
- model.py

  Contains Convolutional model classes. DnCNN (denoising CNN) and DnCNN-Res (denoising CNN with residual blocks).
  * DnCNN is an image denoising CNN developed by Zhang el al. ([Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising](https://arxiv.org/pdf/1608.03981.pdf)) which I've implemented using Pytorch and converted the 2D-Convolutional layers to 1D. The DnCNN-Res is an amalgamation of the DnCNN model and the Res-Net model ([Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)) I developed by adding skip connections between every other Convolutional layer, which allows for the training of ultra-deep models without encountering the problem of vanishing gradients.
  
- train.py

  Trains a Convolutional model for denoising gamma-spectra using training set created by build_dataset (default model predicts clean spectra, "--gennoise" flag trains model that predicts noise mask).
  
- denoise.py

  Tests trained model on validation data, generates plots to view the results (default is clean generative model, "--gennoise" flag tests noise mask model).  
  
- idCVX.py

  Radionuclide classification script using convex optimization. Uses templates.h5 created by gen_templates.py script.

- cvx.py

  Convex optimization script adopted from Bart Massey's [sd code](https://github.com/BartMassey/sd) presented under the MIT license.
  
# Training

## Noise Mask Model (DnCNN)
While I did not have enough time to do a significant amount of hyperparameter tuning, I found that the Noise Mask model didn't perform well with over 20 convolutional layers. Therefore, for this model, I used the DnCNN with 20 convolutional layers since the DnCNN-Res is intended for training with more than 20 convolutional layers.

![](/figs/GenNoise_Train.png)

## Generate Spectrum Model (DnCNN-Res)
For the Generate Spectrum model I experimented with both the DnCNN model with up to 20 convolutional layers and the DnCNN-Res model with as many as 80 convolutional layers. I found the optimal number of layers to be 30. Thus, for this model I'm used the DnCNN-Res since the DnCNN does not perform well with more than 20 layers.

![](/figs/GenSpec_Train.png)

# Results

## Denoising

Average PSNR of validation set before and after denoising.

| | No Denoising| Noise Mask DnCNN | Gen Spectrum DnCNN-Res|
|:--|:--:|:--:|:--:|
|PSNR|32.82dB|58.21dB (+25.39dB)|66.69dB (+33.87dB)|

## Classification

Radionuclide classification using Convex Optimization before and after denoising.

| | No Denoising| Noise Mask DnCNN | Gen Spectrum DnCNN-Res|
|:--|:--:|:--:|:--:|
|Accuracy|90.99%|???%|96.21%|

# Examples of the Gen Spectrum DnCNN-Res model

### Europium-152 Template Spectrum for High Purity Germanium (HPGE) Detector Type

![](/figs/eu152_template.png)

### Simulated Spectrum for Europium-152 with Compton Scatter and Exponentially Decaying Noise, and applying Detector's Efficiency

![](/figs/eu152_noisy.png)

### Denoised Simulated Noisy Spectrum

![](/figs/eu152_denoised.png)

### Comparison of Denoised Spectrum and Template Spectrum

![](/figs/eu152_target_denoised.png)

### Ytterbium-169 Template Spectrum for High Purity Germanium (HPGE) Detector Type

![](/figs/yb169_template.png)
  
### Simulated Spectrum for Ytterbium-169 with Compton Scatter and Exponetially Decaying Noise, and applying Detector's Efficiency

![](/figs/yb169_noisy.png)

### Denoised Simulated Noisy Spectrum

![](/figs/yb169_denoised.png)

### Comparison of Denoised Spectrum and Template Spectrum

![](/figs/yb169_target_denoised.png)

# Roadmap

- [x] Build simulated datasets based on real detector properties
- [x] Train Deep Learning models
  - [x] Generative convolutional model for clean gamma-spectra
  - [x] Generative convolutional model for noise mask
- [x] Build radionuclide classifier with Convex Optimization (cvxpy)
- [x] Optimize algorithm parameters
  - [x] Clean geneartive convolutional model
  - [x] Noise mask generative convolutional model
  - [x] Convex Optimization
- [x] Compare results of approaches

# Future Work
- [ ] Use GADRAS or MCNP to generate more realistic clean and noisy spectra

## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/mpc6/gamma-spectra_denoising/blob/master/LICENSE.txt)
This work is released under MIT license.
