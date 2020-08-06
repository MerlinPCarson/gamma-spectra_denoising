

# Gamma-Spectra denoising
Algorithms for removing background, Compton scatter and detector noise from gamma-ray spectra.

# Requirements
* Pytorch
* Sklearn
* Scikit-Image

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
  
# Roadmap

- [x] Build simulated datasets based on real detector properties
- [x] Train Deep Learning models
  - [x] Generative convolutional model for clean gamma-spectra
  - [x] Generative convolutional model for noise mask
- [x] Build radionuclide classifier with Convex Optimization (cvxpy)
- [ ] Optimize algorithm parameters
  - [ ] Clean geneartive convolutional model
  - [ ] Noise mask generative convolutional model
  - [ ] Convex Optimization
- [ ] Compare results of approaches

## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/mpc6/gamma-spectra_denoising/blob/master/LICENSE.txt)
This work is released under MIT license.
