

# Gamma-Spectra denoising
### by Merlin Carson
Algorithms for removing background, Compton scatter and detector noise from gamma-ray spectra.

This repo contains python scripts for simulating clean and noisy gamma-ray spectra for 3 types of gamma-ray detectors (HPGe, NaI, CZT), trainable Convolutional Neural Networks for denoising gamma-ray spectra, and a convex optimization based radionuclide classifier. The training script trains two variants of the denoising algorithm, one that predicts a noise mask (Noise Mask Model) that is then subtracted from the noisy spectra in post processing and one that directly predicts the denoised spectra (Generate Spectrum Model). For each of these two variants there are two versions of Convolutional Neural Network that can be used. The first, a traditional Convolution Neural Network with ReLU activation and batch normalization for each convolutional layer, and a configurable number of these blocks. The second is a variant of the first with residual connections between every other convolutional layer block, allowing for the training of ultra-deep (20+ layer) models.  

# Requirements
* Pytorch
* Sklearn
* Scikit-Image
* cvxpy
* h5py

<pre>  An NVIDIA GPU: 

  The training and denoising scripts are written to support the use of multiple GPUs. 
  While they will run correctly on a single GPU, they will not run on the CPU. 
  All other scripts run solely on the CPU.
  
</pre>

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
  * DnCNN is an image denoising CNN developed by Zhang el al. ([Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising](https://arxiv.org/pdf/1608.03981.pdf)) which I've implemented using Pytorch and converted the 2D-convolutional layers to 1D. The DnCNN-Res is an amalgamation of the DnCNN model and the Res-Net model ([Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)) that I developed by adding skip connections between every other convolutional layer block, which allows for the training of ultra-deep models without encountering the problem of vanishing gradients.
  
- train.py

  Trains a Convolutional model for denoising gamma-ray spectra using the training set created by build_dataset. Default model predicts clean spectra (Generate Spectrum Model), "--gennoise" flag trains model that predicts noise in spectrum (Noise Mask Model). The DnCNN network with 20 convolutional layer blocks is used by default, use "--num_layers" to change the number of layer blocks, use "--res" flag for DnCNN-Res 
  
- denoise.py

  Tests trained model on validation data, generates plots to view the results. The model type and network parameters are loaded from the model history file (best_model.npy) generated during training.  
  
- idCVX.py

  Radionuclide classification script using convex optimization. Uses templates.h5 created by gen_templates.py script for the bases. Loads spectra from a .h5 file created by build_dataset or denoising script.

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

Average PSNR of validation set (lowest PSNR is 13.34dB, highest PSNR is 51.85dB) before and after denoising.

| | No Denoising| Noise Mask DnCNN | Gen Spectrum DnCNN-Res|
|:--|:--:|:--:|:--:|
|PSNR|32.82dB|58.21dB (+25.39dB)|66.69dB (+33.87dB)|

## Classification

Radionuclide classification using Convex Optimization before and after denoising.

| | No Denoising| Noise Mask DnCNN | Gen Spectrum DnCNN-Res|
|:--|:--:|:--:|:--:|
|Accuracy|90.99%|96.17%|96.21%|

# Examples of the Gen Spectrum DnCNN-Res model

### Europium-152 Template Spectrum for High Purity Germanium (HPGE) using Detector's Efficiency and Resolution

![](/figs/eu152_template.png)

### Simulated Spectrum for Europium-152 with Compton Scatter and Exponentially Decaying Noise

![](/figs/eu152_noisy.png)

### Denoised Simulated Noisy Spectrum (+25.92dB PSNR)

![](/figs/eu152_denoised.png)

### Comparison of Denoised Spectrum and Template Spectrum

![](/figs/eu152_target_denoised.png)

### Ytterbium-169 Template Spectrum for High Purity Germanium (HPGE) using Detector's Efficiency and Resolution

![](/figs/yb169_template.png)
  
### Simulated Spectrum for Ytterbium-169 with Compton Scatter and Exponetially Decaying Noise

![](/figs/yb169_noisy.png)

### Denoised Simulated Noisy Spectrum  (+35.40dB PSNR)

![](/figs/yb169_denoised.png)

### Comparison of Denoised Spectrum and Template Spectrum

![](/figs/yb169_target_denoised.png)


# Conclusion

It is clear that the Generate Spectrum model performs superiorly, averaging an almost 8.5dB PSNR improvment over the Noise Mask model. This is most likely due to the fact that photoelectric peaks are gaussian like in shape, which is a much easier distribution for a neural network to learn than the poisson distributed exponentially decaying noise. The batch normalization layers between each convolutional layer also help mitigate internal covariate shift, which keeps the parameters centered around a mean of zero with unit variance. Thus, making models more capable of learning normal distributions. 

It is clear that there is some amount of data lost during the denoising process for very small photopeaks and when there is a spike in noise around a photopeak. However, the majority and the strongest photopeaks are clearly maintained almost perfectly, while the rest of the spectrum is almost entirely surpressed. These strong, presistent photopeaks are the clearest identifiers for the classification of radionuclides and therefore the convex optimization algorithm performs better after denoising than before, despite loss of information. The denoising also results in sparse representations of the spectra which makes classification more efficient.

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
