from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as f:
    readme = f.read()


setup(
      name="scBT",
      version="0.0.1",
      author="Oscar Davalos",
      author_email="odavalos2@ucmerced.edu",
      description="A contrastive learning model for scRNA-Seq data using BarlowTwins loss",
      long_description=readme,
      long_description_content_type="text/markdown",
      license="MIT",
      url="https://github.com/odavalos/scBT",
      download_url="https://github.com/odavalos/scBT",
      packages=find_packages(),
      keywords=['Single-cell RNASeq', 'scRNA-Seq','Clustering', 'Neural Networks', 'Autoencoders', 'Tabular Data'],
      install_requires=[
                        'scanpy>=1.9.1',
                        'numpy>=1.21.5',
                        'torch>=1.12.1',
                        'scikit-learn>=1.1.2'
                        ]
)
