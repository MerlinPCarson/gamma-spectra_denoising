

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
  
- train.py

  Trains a model Convolutional model for denoising gamma-spectra using training set created by build_dataset.
  
- denoise.py

  Tests trained model on validation data, generates plots to view the results.  
  
# Roadmap

- [x] Build simulated datasets based on real detector properties
- [ ] Train Deep Learning models
  - [x] Generative convolutional model for clean gamma-spectra
  - [ ] Generative convolutional model for noise mask
- [ ] Experiment with Convex Optimization 
- [ ] Optimize algorithm parameters
  - [ ] Clean geneartive convolutional model
  - [ ] Noise mask generative convolutional model
  - [ ] Convex Optimization
- [ ] Compare results of both approaches

## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/mpc6/gamma-spectra_denoising/blob/master/LICENSE.txt)
This work is released under MIT license.
