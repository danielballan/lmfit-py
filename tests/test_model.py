import unittest
from lmfit.utilfuncs import gauss, assert_results_close
import numpy as np

from lmfit import Model, Parameter

class TestModel(unittest.TestCase):

    def setUp(self):
        self.x = np.linspace(-10, 10, num=1000)
        self.noise = 0.01*np.random.randn(*self.x.shape)
        self.true_values = lambda: dict(amp=7, cen=1, wid=3)
        self.guess = lambda: dict(amp=5, cen=2, wid=4)  # return a fresh copy
        self.model = Model(gauss, ['x'])
        self.data = gauss(x=self.x, **self.true_values()) + self.noise

    def test_fit_with_keyword_params(self):
        result = self.model.fit(self.data, x=self.x, **self.guess())
        assert_results_close(result.values, self.true_values())

    def test_fit_with_parameters_obj(self):
        params = self.model.params()
        for param_name, value in self.guess().items():
            params[param_name].value = value
        result = self.model.fit(self.data, params, x=self.x) 
        assert_results_close(result.values, self.true_values())

    def test_missing_param_raises_error(self):

        # using keyword argument parameters
        guess_missing_wid = self.guess()
        del guess_missing_wid['wid']
        f = lambda: self.model.fit(self.data, x=self.x, **guess_missing_wid)
        self.assertRaises(ValueError, f)

        # using Parameters
        params = self.model.params()
        for param_name, value in guess_missing_wid.iteritems():
            params[param_name].value = value
        f = lambda: self.model.fit(self.data, params, x=self.x)

    def test_missing_independent_variable_raises_error(self):
        f = lambda: self.model.fit(self.data, **self.guess())
        self.assertRaises(KeyError, f)

    def test_bounding(self):
        guess = self.guess()
        guess['cen'] = Parameter(value=2, min=1.3)
        true_values = self.true_values()
        true_values['cen'] = 1.3  # as close as it's allowed to get
        result = self.model.fit(self.data, x=self.x, **guess)
        assert_results_close(result.values, true_values, rtol=0.05)

    def test_vary_false(self):
        guess = self.guess()
        guess['cen'] = Parameter(value=1.3, vary=False)
        true_values = self.true_values()
        true_values['cen'] = 1.3
        result = self.model.fit(self.data, x=self.x, **guess)
        assert_results_close(result.values, true_values, rtol=0.05)
