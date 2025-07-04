This code is an expanding of  [surfinBH](https://github.com/vijayvarma392/surfinBH).
Paper: **Phys.Rev.D 104 (2021) 6, 064056**
# SurfinBH

_**surfinBH**_ provides _**sur**rogate **fin**al **B**lack_ _**H**ole_
properties for mergers of binary black holes (BBH).

If you find this package useful in your work, Please also cite [surfinBH](https://github.com/vijayvarma392/surfinBH) and relativate papers in the ReadMe file of [surfinBH](https://github.com/vijayvarma392/surfinBH).

This package is compatible with both python2 and python3.

# This code

We expand surfinBH so that it can estimate the memory an BBH event leaves in spacetime. 

This fit is described in the following paper:

[1] Zhi-Chao Zhao, Xiaolin Liu, Zhoujian Cao, et al., Phys.Rev.D 104 (2021) 6, 064056.

Note: the result of this code is ```log10(memory)``` and the GPR error of ```log10(memory)```

## Installation


### From source

```shell
git clone https://github.com/Zhi-ChaoZhao/NRSurMemory_7qd4
cd NRSurMemory_7qd4
git submodule init
git submodule update
python setup.py install
```

If you do not have root permissions, replace the last step with
`python setup.py install --user`


## Dependencies
All of these can be installed through pip or conda.
* [numpy](https://docs.scipy.org/doc/numpy/user/install.html)
* [scipy](https://www.scipy.org/install.html)
* [h5py](http://docs.h5py.org/en/latest/build.html)
* [scikit-learn](http://scikit-learn.org/stable/install.html) (at least 0.19.1)
* [lalsuite](https://pypi.org/project/lalsuite) (at least 6.70)
* [gwsurrogate](https://pypi.org/project/gwsurrogate)

## Usage

```python
import surfinBH
```


```python
fit_name = 'NRSurMemory_7dq4Remnant'

surfinBH.fits_collection[fit_name].desc
>>> 'Fits for the gravitational wave memory for generically precessing BBH systems up to mass ratio 4.'

surfinBH.fits_collection[fit_name].refs
>>> 'arxiv:ToBePublished'
```

### Load the fit
This only needs to be done **once** at the start of your script.
If the fit data is not already downloaded, this will also download the data.

```python
fit = surfinBH.LoadFits(fit_name)
>>> Loaded NRSurMemory_7dq4Remnant fit.
```
### Evaluation
The evaluation of each fit is different, so be sure to read the documentation.
This also describes the frames in which different quantities are defined.

```python
help(fit)
```

We also provide ipython example for usage of this fit:

[NRSurMemory_7dq4Remnant](https://github.com/vijayvarma392/surfinBH/blob/master/examples/example_Mem_7dq4.ipynb) (Ref. [1])



