from disarm_gears.gears.stan_plugins import GPStanRegression
from disarm_gears.gears.stan_plugins import stan_compilers
from disarm_gears.validators import *


class GPStanPoissonRegression(GPStanRegression):

    def __init__(self, stan_models=None):
        '''
        This is a wrapper of a model implemented in pystan.
        See: https://mc-stan.org
        '''
        super(GPStanPoissonRegression, self).__init__(stan_models=stan_models)


    def _compile_models(self):
        return stan_compilers.gp_poisson_compiler()


    def _make_train_dict(self, X, y, mu_prior, exposure, **kwargs):

        validate_integer_array(y)
        validate_non_negative_array(y)
        validate_positive_array(exposure)
        return {'X_data': X, 'y_data': y.astype(int), 'n_data': y.size, 'n_dim': X.shape[1],
                'mu_data': mu_prior, 'exposure_data': exposure}


    def _make_pred_dict(self, X, mu_prior, exposure, **kwargs):

        assert hasattr(self, 'train_dict')
        validate_positive_array(exposure)
        new_dict = self.train_dict.copy()
        new_dict.update(self.params)
        new_dict.update({'X_pred': X, 'n_pred': X.shape[0], 'mu_pred': mu_prior,
                         'exposure_pred': exposure})

        return new_dict
