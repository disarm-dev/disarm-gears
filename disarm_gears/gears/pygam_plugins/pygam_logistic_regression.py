from disarm_gears.validators import *
import pygam


class PyGAMLogisticRegression:

    def __init__(self):
        '''
        This is a wrapper of regression models in pygam.
        See: TODO
        '''
        pass

    def _base_instance(self):
        return pygam.LogisticGAM()


    def _log_likelihood(self):
        return self.base.loglikelihood(X=self._X_train, y=self._y_train, weights=self._weights_train)


    def fit(self, y, X, weights, **kwargs):
        validate_1d_array(y)
        validate_2d_array(X, n_rows=y.size, n_cols=None)
        self._y_train = y
        self._X_train = X
        self._weights_train = weights
        self.n_dim = X.shape[1]
        self.base = self._base_instance()
        self.base.gridsearch(y=y, X=X, weights=weights)


    def predict(self, X, phi=True, threshold=.5, **kwargs):
        validate_2d_array(X, n_cols=self.n_dim)
        mu = self.base.predict_mu(X)
        if not phi:
            assert 0 < threshold and threshold < 1., 'Invalid threshold.'
            _mu = np.zeros_like(mu)
            _mu[mu > threshold] = 1
            mu = _mu
        return mu


    def posterior_samples(self, X, n_samples=100, phi=True, **kwargs):
        validate_2d_array(X, n_cols=self.n_dim)
        if phi:
            samples = self.base.sample(y=self._y_train, X=self._X_train, weights=self._weights_train,
                                       sample_at_X=X, quantity='mu', n_draws=n_samples)
        else:
            samples = self.base.sample(y=self._y_train, X=self._X_train, weights=self._weights_train,
                                       sample_at_X=X, quantity='y', n_draws=n_samples)
        return samples

