# scBT

**EXPERIMENTAL**

Contrastive learning model for scRNA-Seq data using [BarlowTwins](https://github.com/facebookresearch/barlowtwins) loss

#### Install Package Locally
Make sure to be in the same directory as `setup.py`. Then, using `pip`, run:

````bash
pip install -e .
````


## TO DO LIST

- [x] Update dataloader to use experimental [Annloader](https://anndata-tutorials.readthedocs.io/en/latest/annloader.html)
- [ ] Figure out how to extract latent space if I call autoencoder class in model.py as the backbone.
- [ ] Update training code
  - [ ] Add early stopping
  - [ ] Add model saving (best epoch)
- [ ] Test data masking methods (currently using `dropout p=0.10``)
- [ ] Add different loss functions ([InfoNCE](https://arxiv.org/abs/1807.03748), [Soft-Nearest Neighbors Loss](https://arxiv.org/abs/1902.01889))?
- [ ] Try shallower networks


