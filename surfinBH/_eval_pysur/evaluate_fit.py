#!/usr/bin/python
import numpy as np
from math import pi, ceil
import warnings
from random import sample as rand_sample

from sklearn import gaussian_process
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn import linear_model

#######################     GPR fit functions       #######################

# These are the things we need to save from a GPR fit in order to reproduce it
# NOTE: If you get errors like "AttributeError: 'GaussianProcessRegressor'
# object has no attribute -----", try adding that attribute to
# GPR_SAVE_ATTRS_DICT
GPR_SAVE_ATTRS_DICT = ['kernel_', 'X_train_', 'alpha_',
        '_y_train_mean', '_y_train_std', 'L_']

# These are the things we need to save from a LinearRegression fit in order
# to reproduce it
# NOTE: If you get errors like "AttributeError: 'LinearRegression'
# object has no attribute -----", try adding that attribute to
# LINREG_SAVE_ATTRS_DICT
LINREG_SAVE_ATTRS_DICT = ['coef_', 'intercept_']

# -------------------------------------------------------------------------
class GPRPredictor:
    def __init__(self, res, **kwargs):
        """
        Class to evaluate GPR fits constructed by GPRFitter class in
        pySurrogate/fit_gpr.py
        """

        self.data_mean = res['data_mean']
        self.data_std = res['data_std']

        # load GPR fit
        self.GPR_obj = GaussianProcessRegressor()
        GPR_params =  res['GPR_params']
        self._set_gpr_params(self.GPR_obj, GPR_params)

        # load LinearRegression fit
        lin_reg_params = res['lin_reg_params']
        if lin_reg_params is not None:
            self.linearModel = linear_model.LinearRegression()
            self._set_lin_reg_params(self.linearModel, lin_reg_params)
        else:
            self.linearModel = None

    def _set_kernel_params(self, kernel_params):
        """ Recursively sets paramters for a kernel and returns the final
            kernel.
        """

        # copy so as to not overwrite
        kernel_params = kernel_params.copy()

        # initialize kernel
        name = kernel_params['name']
        kernel = getattr(gaussian_process.kernels, name)
        del kernel_params['name']

        params = {}
        for key in kernel_params.keys():
            if type(kernel_params[key]) == dict:
                # recursively set kernels
                params[key] = self._set_kernel_params(kernel_params[key])
            else:
                params[key] = kernel_params[key]

        if name == 'Sum' or name == 'Product':
            kernel = kernel(params['k1'], params['k2'])
        else:
            kernel = kernel()

        kernel.set_params(**params)
        return kernel


    def _set_gpr_params(self, gp_obj, gp_params):
        """ Sets the fitted hyperparameter for a GPR object.
            This can be used to load a previously constructed fit.

            NOTE: If you get errors like:
            "AttributeError: 'GaussianProcessRegressor' object has
            no attribute -----",
            try adding that attribute to GPR_SAVE_ATTRS_DICT
        """
        for attr in GPR_SAVE_ATTRS_DICT:
            if attr == 'kernel_':
                param = self._set_kernel_params(gp_params[attr])
            elif attr == '_y_train_std':
                # In scikit-learn versions before 0.23, there was no
                # _y_train_std, which is the same as saying _y_train_std=1. If
                # this fit was constructed using an earlier version of
                # scikit-learn, it would not have an attribute called
                # _y_train_std, so we just set it to 1. This way, the fit can
                # be evaluated with any version of scikit-learn.
                if '_y_train_std' not in gp_params.keys():
                    param = 1
                else:
                    param = gp_params[attr]
            else:
                param = gp_params[attr]
            setattr(gp_obj, attr, param)


    def _set_lin_reg_params(self, lr_obj, lr_params):
        """ Sets the fitted parameters for a LinearRegression object.
            This can be used to load a previously constructed fit.

            NOTE: If you get errors like:
            "AttributeError: 'LinearRegression' object has
            no attribute -----",
            try adding that attribute to LINREG_SAVE_ATTRS_DICT
        """
        for attr in LINREG_SAVE_ATTRS_DICT:
            param = lr_params[attr]
            setattr(lr_obj, attr, param)


    def _reconstruct_normalized_data(self, data_normed, data_normed_err):
        """
        The inverse operation of 'GPRFitter._normalize()'
        Returns the reconstructed data and error estimate.
        """
        return data_normed * self.data_std + self.data_mean, \
            data_normed_err * self.data_std

    def GPR_predict(self, x, estimate_err=False):
        """
        Evaluates a GPR fit.
        First evalutates the GPR fit to get the prediction for the normalized
        data. Then reconstructs the un-normalized data.
        Finally adds the linear model fit if it was done in GPRFitter.
        """

        # Get fit prediction and error estimate for normalized data
        fit_res = self.GPR_obj.predict(x, return_cov=estimate_err)

        if estimate_err:
            y_normalized_pred, cov_normalized_pred = fit_res
            err_normalized_pred = np.sqrt(cov_normalized_pred.flatten())
        else:
            y_normalized_pred = fit_res
            err_normalized_pred = fit_res * 0

        # Reconstruct to get un-normalized prediction
        y_pred, err_pred = self._reconstruct_normalized_data( \
                                            y_normalized_pred, \
                                            err_normalized_pred)

        if self.linearModel is not None:
            # Add the linear prediction that was subtracted before
            # doing the fit
            y_pred = y_pred + self.linearModel.predict(x)


        val_dict = {
            'y': y_pred,
        }

        if estimate_err:
            val_dict['y_gprErr'] = err_pred

        return val_dict


#######################     Greedt fit functions       #######################
class BasisFunction:
    def __init__(self, name, func, minX, maxX):
        """name: A name (for example, 'cosine')
        func: f(n, x) gives the n'th basis function evaluated at x
            for example, lambda n, x: np.cos(n*x)
        minX, maxX: The default domain (for example, minX=0, maxX=pi)
            If the user's domain is different, linearly maps to the default.
            Can give None, in which case the domain is not mapped.
        """
        self.name = name
        self.func = func
        self.minX = minX
        if len([tmp for tmp in [minX, maxX] if tmp is None]) == 1:
            raise Exception('Must have both or neither minX=None, maxX=None')
        if self.minX is None:
            self.width = None
        else:
            self.width = maxX - minX

    def __call__(self, n, x):
        return self.func(n, x)

    def mappedFunc(self, n, minX, maxX):
        if self.minX is None:
            return lambda x: self.func(n, x)
        def newFunc(x):
            return self(n, self.minX + (x-minX)*self.width/(maxX-minX))
        return newFunc

## Define some commonly used BasisFunctions
_polynomial = BasisFunction(
        'polynomial',
        lambda n, x: 1. + 0.*x if n==0 else x**n,
        -1, 1)

_complexPeriodic = BasisFunction(
        'complexPeriodic',
        lambda n, x: np.exp(1.j*n*x/2) if n%2==0 else np.exp(-1.j*(n+1)*x/2),
        0, 2*pi)

_periodic = BasisFunction(
        'periodic',
        lambda n, x: np.cos(n*x/2) if n%2==0 else np.sin((n+1)*x/2),
        0, 2*pi)

_chebychev = BasisFunction(
        'chebyshev',
        lambda n, x: np.polynomial.Chebyshev([0]*n + [1])(x),
        -1, 1)

bf_dict = {
            'polynomial': _polynomial,
            'complexPeriodic': _complexPeriodic,
            'periodic': _periodic,
            'chebyshev': _chebychev,
          }

def basisFunction(typeStr, n, minVal, maxVal, BF_dict=bf_dict):
    return BF_dict[typeStr].mappedFunc(n, minVal, maxVal)


#######################   Fit evaluator functions   #######################
def getFitEvaluator(res):
    if res is None:
        return None

    def greedyfitEvaluator(x):
        basisFuncEvals = np.array([
                np.prod(
                  np.array([
                    basisFunction(
                      res['bfTypes'][j],
                      orders[j],
                      res['minVals'][j],
                      res['maxVals'][j])(x[j])
                    for j in range(len(x))]), 0)
                for orders in res['bfOrders']])
        return res['coefs'].dot(basisFuncEvals)

    def gprfitEvaluator(x):
        GPR_eval  = gpr_predictObject.GPR_predict([x])
        return GPR_eval['y'][0] # This probaly needs to change if len(x) != 1

    if 'fitType' in res and res['fitType'] == 'GPR':
        gpr_predictObject = GPRPredictor(res)
        return gprfitEvaluator
    else:
        return greedyfitEvaluator


# Evalutates the GPR error estimation function.
# TODO: Currently this just reevaluates the fit, consider incorporating into
# getFitEvaluator() somehow
def getGPRErrorEvaluator(res):

    def gprErrEval(x):
        GPR_eval  = gpr_predictObject.GPR_predict([x], estimate_err=True)
        return GPR_eval['y_gprErr'][0]

    if 'fitType' in res and res['fitType'] == 'GPR':
        gpr_predictObject = GPRPredictor(res)
        return gprErrEval
    else:
        raise Exception('getGPRErrorPrediction requires fitType=GPR.')


def getGPRFitAndErrorEvaluator(res):

    def gprEval(x):
        GPR_eval  = gpr_predictObject.GPR_predict([x], estimate_err=True)
        return GPR_eval['y'][0], GPR_eval['y_gprErr'][0]

    if 'fitType' in res and res['fitType'] == 'GPR':
        gpr_predictObject = GPRPredictor(res)
        return gprEval
    else:
        raise Exception('getGPRErrorPrediction requires fitType=GPR.')
