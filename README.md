Python GLMNET
=============

>glmnet: Lasso and Elastic-Net Regularized Generalized Linear Models.

This is a Python wrapper for the fortran library used in the R package
[`glmnet`](http://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html).
While the library includes linear, logistic, Cox, Poisson, and multiple-response
Gaussian, only linear and logistic are implemented in this package.

The API follows the conventions of [Scikit-Learn](http://scikit-learn.org/stable/),
so it is expected to work with tools from that ecosystem.

Installation
------------
`glmnet` depends on numpy, scikit-learn and scipy. A working Fortran compiler
is also required to build the package, for Mac users, `brew install gcc` will
take care of this requirement.

```bash
git clone git@github.com:civisanalytics/python-glmnet.git
cd python-glmnet
python setup.py install
```

Alternatively, one can install the zipped project directly from GitHub.
```bash
pip3 install https://github.com/HIVERY/python-glmnet/archive/master.zip
```

Usage
-----

### General

By default, `LogitNet` and `ElasticNet` fit a series of models using the lasso
penalty (α = 1) and up to 100 values for λ (determined by the algorithm). In
addition, after computing the path of λ values, performance metrics for each
value of λ are computed using 3-fold cross validation. The value of λ
corresponding to the best performing model is saved as the `lambda_max_`
attribute and the largest value of λ such that the model performance is within
`cut_point * standard_error` of the best scoring model is saved as the
`lambda_best_` attribute.

The `predict` and `predict_proba` methods accept an optional parameter `lamb`
which is used to select which model(s) will be used to make predictions. If
`lamb` is omitted, `lambda_best_` is used.

Both models will accept dense or sparse arrays.

### Regularized Logistic Regression

```python
from glmnet import LogitNet

m = LogitNet()
m = m.fit(x, y)
```

Prediction is similar to Scikit-Learn:
```python
# predict labels
p = m.predict(x)
# or probability estimates
p = m.predict_proba(x)
```

### Regularized Linear Regression

```python
from glmnet import ElasticNet

m = ElasticNet()
m = m.fit(x, y)
```

Here are some of the features available in glmnet.

```python
import numpy as np
m = ElasticNet(alpha=0.01)  # set the alpha values
m = ElasticNet(fit_intercept=False)  # disable the intercept fit
m = ElasticNet(lambda_path=np.exp(np.linspace(-1, -4, num=20))  # set a decreasing sequence of lambda values for glmnet to use
m = ElasticNet(lower_limits=np.array([0, -np.inf, 0]), upper_limits=np.array([np.inf, 3, np.inf]))  # set limits on the coefficients
m = ElasticNet(n_splits=30)  # set the number of cross-validation splits - note that CV is enabled by default.
```

Once the model is defined, we can set weights for each observation in the fit method.

```python
m.fit(X=x, y=y, sample_weights=np.array([1, 3, 1])  # the weights get standardised internally
```

Predict:
```python
p = m.predict(x)
```
