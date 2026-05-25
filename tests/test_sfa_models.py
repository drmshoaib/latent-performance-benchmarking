from __future__ import annotations

import numpy as np

from sfa.loaders import design_matrix
from sfa.sfa_halfnormal import HalfNormalSFA


def test_half_normal_likelihood_and_invalid_params(synthetic_dataset):
    group = synthetic_dataset.data.query("portfolio == 'SMALL LoBM'")
    y = group["excess_return"].to_numpy(float)
    X = design_matrix(group, synthetic_dataset.factor_cols)
    model = HalfNormalSFA(y, X)
    theta = model._ols_start()

    assert np.isfinite(model._neg_loglik(theta))
    assert np.isinf(model._neg_loglik(np.full_like(theta, np.nan)))


def test_half_normal_fit_returns_required_fields(synthetic_dataset):
    group = synthetic_dataset.data.query("portfolio == 'SMALL LoBM'")
    y = group["excess_return"].to_numpy(float)
    X = design_matrix(group, synthetic_dataset.factor_cols)

    fit = HalfNormalSFA(y, X).fit(maxiter=80)
    summary = fit.summary()

    required = {
        "beta",
        "alpha",
        "sigma_v",
        "sigma_u",
        "lambda",
        "log_likelihood",
        "AIC",
        "BIC",
        "converged",
    }
    assert required <= set(summary)
    assert np.all(fit.AE > 0)
    assert np.all(fit.AE <= 1)
    assert fit.residuals.shape == y.shape
    assert fit.fitted_values.shape == y.shape
