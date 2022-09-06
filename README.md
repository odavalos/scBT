# scBT

**EXPERIMENTAL**

Contrastive learning model for scRNA-Seq data using [BarlowTwins](https://github.com/facebookresearch/barlowtwins) loss

#### Install Package Locally
Make sure to be in the same directory as `setup.py`. Then, using `pip`, run:

````bash
pip install -e .
````


## TO DO LIST

- [ ] Update dataloader to use experimental [Annloader](https://anndata-tutorials.readthedocs.io/en/latest/annloader.html)
- [ ] Figure out how to extract latent space if I call autoencoder class in model.py as the backbone.
- [ ] Update training code
- [ ] Test data masking methods (currently using `dropout p=0.10``)


