# Kaggle House Prices Competition

## Overview

House Prices (https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) is a Kaggle competition whose purpose is to practices skills in regression problems and acquire new advanced skills solving regression problems.
This project contains modelisations and various tests for predicting prices from test set.

## Requirements

Python-3.7.12 which is installable with `pyenv` (see below)

The following install procedure requires `pyenv` and its plugin `pyenv-virtualenv`. Check eventually their repo to see how to install them

* pyenv : https://github.com/pyenv/pyenv
* pyenv-virtualenv : https://github.com/pyenv/pyenv-virtualenv


## Install

1. clone the repository

```
$ git clone https://github.com/mikachou/house-prices
```

Then enter the folder
```
$ cd house-prices
```

2. install right python version with pyenv :
```
$ pyenv local
```

You may check that you have install the right python version

```
$ python -V
Python 3.7.12
```

Create a virtualenv for this project

```
$ pyenv virtualenv house-prices
```

Enter in the just created virtualenv

```
$ pyenv activate house-prices
```

3. install required packages
```
$ pip install -r requirements.txt
```

## Usage

### EDA

The files for EDA are already generated and included in repo in `eda/` directory. However here are instructions in case you might want to generate them again.

#### Pandas Profiler report

The report are already generated and located in `eda/report_train.html` and `ede/report_test.html`

If you still want to generate them use `pandas_profiler` command and generate report for train and test data
```
$ pandas_profiling data/train.csv eda/report_train.html

$ pandas_profiling data/test.csv eda/report_test.html
```
#### Pair plots

Image of pairplots is located in `eda/pairplot_train.png`

To generate visualisation of pairplots execute the following command
```
$ python eda_pairplot.py
```

#### Target transformers visualization

Image of effect of various transformers on target variable (SalePrice) is `eda/dist_saleprice.png`

To generate it type the following command :

```
$ python eda_transformer.py
```

#### Outliers

To display outliers from train set type the following command :

```
$ python eda_outliers
```

### Modelisation

Different outliers treatments, preprocessings, and models are implemented.
Once these three parameters selected you must also choose number of trials for hyperparameters tuning and cross-validation.

Typical simulation would be called regarding this pattern :

```
$ python main.py <outliers> <preprocessor> <model> <number>
```

* The `outliers` parameter must be choosen after the filenames from `inc/outliers/` (without `.py` extension): it can be `outliers0` (no outliers treatment) or `outliers1` (some outliers are retired from set before preprocessing)
* The `preprocessor` parameter must be choosen after the filenames from `inc/preprocessor/` (without `.py` extension): it can be `preprocessor1` or `preprocessor2`, but are almost sames (details are given anywhere else). However for Kaggle competition `preprocessor1` was choosen most of times
* The `model` parameter must be choosen after the filenames from `inc/model/` (without `.py` extension). Each file contains an instanciated scikit-learn model and a set of hyper-parameters with a selection range for each one of them
* The `number` parameter is number of cross-val trials with hyper-parameters tuning.

For instance to try 50 times xgboost with log-transformed target you may type :

```
$ python main.py outliers1 preprocessor1 log_xgboost 50
```

After each execution a log file is generated in `logs/` directory and the submission file for kaggle competition is generated in `submissions/` directory

### Blend

Aside stacking and blending models (see `inc/model/stack*` files) there is a dummy blend model that you may execute with :
```
$ python blend.py
```
A submission file is generated in `submissions/` but result is not that good.