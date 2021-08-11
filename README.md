[![Build Status](https://travis-ci.org/AlexsLemonade/fancySVD.svg?branch=master)](https://travis-ci.org/AlexsLemonade/fancySVD) [![Coverage Status](https://coveralls.io/repos/github/AlexsLemonade/fancySVD/badge.svg?branch=master)](https://coveralls.io/github/AlexsLemonade/fancySVD?branch=master) [![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.51773.svg)](http://dx.doi.org/10.5281/zenodo.51773)


![plot](https://user-images.strikinglycdn.com/res/hrscywv4p/image/upload/c_limit,fl_lossy,h_1440,w_720,f_auto,q_auto/174108/654579_364680.png)


A fork of https://github.com/iskandr/fancyimpute which only includes its IterativeSVD algorithm.
The original fancyimpute project had many more algorithms which required installing Tensorflow, which was difficult to do because of dependency issues.
This minimal project is much simpler to install.

To install:

`pip install fancySVD`

## Important Caveats

(1) This project is in "bare maintenance" mode. That means we are not planning on adding more imputation algorithms or features (but might if we get inspired). Please do report bugs, and we'll try to fix them. Also, we are happy to take pull requests for more algorithms and/or features.


## Usage

```python
from fancySVD import IterativeSVD

imputed_matrix = IterativeSVD().fit_transform(matrix)
```

## Algorithms

* `IterativeSVD`: Matrix completion by iterative low-rank SVD decomposition. Should be similar to SVDimpute from [Missing value estimation methods for DNA microarrays](http://www.ncbi.nlm.nih.gov/pubmed/11395428) by Troyanskaya et. al.

## Citation

As this project was forked from work done in https://github.com/iskandr/fancyimpute, if you use `fancySVD` in your academic publication, please cite the original authors as follows:
```bibtex
@software{fancyimpute,
  author = {Alex Rubinsteyn and Sergey Feldman},
  title={fancyimpute: An Imputation Library for Python},
  url = {https://github.com/iskandr/fancyimpute},
  version = {0.5.4},
  date = {2016},
}
```
