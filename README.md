[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/mpc6/AudioRNN/blob/master/LICENSE.txt)

# gamma-spectra_denoising
Algorithms for removing background and noise from gamma-ray spectra.

# Scripts
- gen_templates.py 

  Generate templates from NNDC tables based on radionuclides and detector parameters in configuration file (default=config.json). Creates h5 file (default=data/templates.h5) use --savefigs arguments to generate plots of all templates.
- load_templates.py

  Load template data file (default=data/templates.h5) and displays all data.
- spectra_utils.py

  Helper functions for gamma-spectra processing.
  
# Roadmap

- [ ] Build simulated datasets based on real detector properties
- [ ] Experiment with Deep Learning models
- [ ] Experiment with Convex Optimization 
- [ ] Compare results of both approaches

## License
This work is released under MIT license.
