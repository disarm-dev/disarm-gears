import pystan


def gp_regression_compiler():
    train_script = 'disarm_gears/gears/stan_plugins/stan_scripts/gp_gaussian_fit_hyper.stan'
    prediction_script = 'disarm_gears/gears/stan_plugins/stan_scripts/gp_gaussian_predict.stan'
    pystan1 = pystan.StanModel(file=train_script)
    pystan2 = pystan.StanModel(file=prediction_script)
    return pystan1, pystan2


def gp_binomial_compiler():
    train_script = 'disarm_gears/gears/stan_plugins/stan_scripts/gp_binomial_fit_hyper.stan'
    prediction_script = 'disarm_gears/gears/stan_plugins/stan_scripts/gp_binomial_predict.stan'
    pystan1 = pystan.StanModel(file=train_script)
    pystan2 = pystan.StanModel(file=prediction_script)
    return pystan1, pystan2


def gp_poisson_compiler():
    train_script = 'disarm_gears/gears/stan_plugins/stan_scripts/gp_poisson_fit_hyper.stan'
    prediction_script = 'disarm_gears/gears/stan_plugins/stan_scripts/gp_poisson_predict.stan'
    pystan1 = pystan.StanModel(file=train_script)
    pystan2 = pystan.StanModel(file=prediction_script)
    return pystan1, pystan2
