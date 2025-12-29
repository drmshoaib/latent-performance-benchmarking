# =====================================================================
# sfa_halfnormal.py — Half-Normal Stochastic Frontier Analysis (SFA)
# =====================================================================
#
# Model:
#     y = Xβ + v + u
#
# Where:
#     v ~ N(0, σ_v²)
#     u ~ |N(0, σ_u²)|   (half-normal inefficiency)
#
# Features:
#   ✓ Analytical log-likelihood
#   ✓ Analytical gradient
#   ✓ Stable L-BFGS-B optimisation
#   ✓ JLMS inefficiency estimator
#   ✓ Identical interface to TruncatedNormalSFA
#
# =====================================================================

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


class HalfNormalSFA:
    """Half-Normal Cost Frontier Model"""

    # ------------------------------------------------------------------
    def __init__(self, y: np.ndarray, X: np.ndarray):
        self.y = np.asarray(y, float)
        self.X = np.asarray(X, float)
        self.n, self.k = self.X.shape

    # ------------------------------------------------------------------
    def _unpack(self, theta):
        """
        theta = [β0 ... β(k−1), log σ_v, log σ_u]
        """
        beta = theta[:self.k]
        sigma_v = np.exp(theta[self.k])
        sigma_u = np.exp(theta[self.k + 1])
        return beta, sigma_v, sigma_u

    # ------------------------------------------------------------------
    def _loglik(self, theta):
        beta, sigma_v, sigma_u = self._unpack(theta)

        eps = self.y - self.X @ beta
        sigma = np.sqrt(sigma_v**2 + sigma_u**2)

        # λ = σ_u / σ_v
        lam = sigma_u / max(sigma_v, 1e-12)

        # composed error
        z = eps / sigma

        # log-likelihood
        ll = (
            -np.log(sigma)
            + norm.logpdf(z)
            + np.log(2.0)
            + norm.logcdf(-(lam * eps) / sigma)
        )

        return -np.sum(ll)

    # ------------------------------------------------------------------
    def _gradient(self, theta):
        """
        Analytical gradient for L-BFGS-B optimisation.
        """
        beta, sv, su = self._unpack(theta)

        eps = self.y - self.X @ beta
        sigma = np.sqrt(sv**2 + su**2)
        lam = su / max(sv, 1e-12)

        z = eps / sigma
        pdf_z = norm.pdf(z)

        arg = -(lam * eps) / sigma
        cdf_term = norm.cdf(arg)
        cdf_term = np.maximum(cdf_term, 1e-12)
        pdf_over_cdf = norm.pdf(arg) / cdf_term

        # ---------------------- ∂LL / ∂β ----------------------
        dLL_db = ((pdf_z / sigma) +
                  lam * pdf_over_cdf / sigma)[:, None] * self.X
        g_beta = -np.sum(dLL_db, axis=0)

        # ---------------------- ∂LL / ∂log σ_v ----------------------
        d_sigma_dsv = sv / sigma
        d_lam_dsv = -su / (max(sv, 1e-12)**2)

        g_sv = -np.sum(
            -(sv / sigma)
            + pdf_z * z * d_sigma_dsv
            + pdf_over_cdf *
              (-(eps * d_lam_dsv) / sigma +
               lam * eps * d_sigma_dsv / sigma**2)
        ) * sv

        # ---------------------- ∂LL / ∂log σ_u ----------------------
        d_sigma_dsu = su / sigma
        d_lam_dsu = 1.0 / max(sv, 1e-12)

        g_su = -np.sum(
            -(su / sigma)
            + pdf_z * z * d_sigma_dsu
            + pdf_over_cdf *
              (-(eps * d_lam_dsu) / sigma +
               lam * eps * d_sigma_dsu / sigma**2)
        ) * su

        return np.concatenate([g_beta, [g_sv, g_su]])

    # ------------------------------------------------------------------
    def fit(self):
        """
        Fit model via L-BFGS-B using analytical gradient.
        """

        # OLS starting values
        beta_ols = np.linalg.lstsq(self.X, self.y, rcond=None)[0]
        resid = self.y - self.X @ beta_ols
        sigma_ols = float(np.std(resid) + 1e-6)

        theta0 = np.concatenate([
            beta_ols,
            [np.log(sigma_ols * 0.5)],
            [np.log(sigma_ols * 0.7)]
        ])

        # try gradient first
        res = minimize(
            self._loglik, theta0,
            jac=self._gradient,
            method="L-BFGS-B",
            options={"maxiter": 400, "disp": False}
        )

        # fallback without gradient
        if not res.success:
            res = minimize(
                self._loglik, theta0,
                method="L-BFGS-B",
                options={"maxiter": 600, "disp": False}
            )

        self.res = res
        self.theta = res.x
        self._postprocess()
        return self
    # ------------------------------------------------------------------
    # Post-processing after optimisation
    # ------------------------------------------------------------------
    def _postprocess(self):
        """
        Compute:
          • u_hat  (JLMS inefficiency estimator)
          • TE     (technical efficiency)
          • frontier (X @ beta)
        """

        beta, sigma_v, sigma_u = self._unpack(self.theta)
        self.beta = beta
        self.sigma_v = sigma_v
        self.sigma_u = sigma_u

        eps = self.y - self.X @ beta
        sigma2 = sigma_v**2 + sigma_u**2

        # JLMS estimator (Jondrow et al., 1982)
        # For half-normal distribution
        mu_star = (sigma_u**2 * eps) / sigma2
        sigma_star = (sigma_v * sigma_u) / np.sqrt(sigma2)

        z = mu_star / sigma_star
        correction = norm.pdf(z) / np.maximum(norm.cdf(z), 1e-12)

        u_hat = mu_star + sigma_star * correction

        # Truncate negative values
        self.u_hat = np.maximum(u_hat, 0.0)

        # Technical efficiency: TE = exp(−û)
        self.TE = np.exp(-self.u_hat)

        # Frontier: predicted optimal consumption
        self.frontier = self.X @ beta

    # ------------------------------------------------------------------
    # Summary dictionary
    # ------------------------------------------------------------------
    def summary(self):
        return {
            "beta": self.beta,
            "sigma_v": self.sigma_v,
            "sigma_u": self.sigma_u,
            "TE_mean": float(np.mean(self.TE)),
            "TE_median": float(np.median(self.TE)),
            "log_likelihood": -self.res.fun,
            "converged": self.res.success
        }