"""
sfa_truncated.py — Truncated-Normal Stochastic Frontier Analysis (SFA)
---------------------------------------------------------------------

Model:
    y = Xβ + v + u
    v ~ N(0, σ_v²)
    u ~ N(μ, σ_u²), truncated at u ≥ 0

This version is fully stabilised for weekly energy TE.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


# ======================================================================
# Truncated-Normal SFA Class
# ======================================================================

class TruncatedNormalSFA:
    """
    Truncated-Normal Stochastic Frontier Model.
    Numerically stable version for energy benchmarking.
    """

    # ------------------------------------------------------------------
    def __init__(self, y: np.ndarray, X: np.ndarray):
        self.y = np.asarray(y, float)
        self.X = np.asarray(X, float)
        self.n, self.k = self.X.shape

        # Regression coefficients initialised later
        self.beta = None
        self.mu = None
        self.sigma_v = None
        self.sigma_u = None

    # ------------------------------------------------------------------
    def _unpack(self, theta):
        """
        theta = [β0..β(k-1), μ, log σ_v, log σ_u]
        """
        beta = theta[:self.k]
        mu = theta[self.k]
        sigma_v = np.exp(theta[self.k + 1])
        sigma_u = np.exp(theta[self.k + 2])
        return beta, mu, sigma_v, sigma_u

    # ------------------------------------------------------------------
    # Log-likelihood
    # ------------------------------------------------------------------
    def _loglik(self, theta):
        beta, mu, sv, su = self._unpack(theta)

        eps = self.y - self.X @ beta
        sigma = np.sqrt(sv**2 + su**2)
        lam = su / sv

        # Truncation term
        t = mu / su
        trunc = np.maximum(norm.cdf(t), 1e-12)

        # Composed error
        z = (eps + mu) / sigma

        # Argument for log CDF term
        arg = -(t - lam * ((eps + mu) / sigma))
        cdf_term = np.maximum(norm.cdf(arg), 1e-12)

        # Full log-likelihood
        ll = (
            -np.log(sigma)
            + norm.logpdf(z)
            + np.log(cdf_term)
            - np.log(trunc)
        )

        return -np.sum(ll)

    # ------------------------------------------------------------------
    # Gradient (numerically stabilised)
    # ------------------------------------------------------------------
    def _gradient(self, theta):
        beta, mu, sv, su = self._unpack(theta)

        eps = self.y - self.X @ beta
        sigma = np.sqrt(sv**2 + su**2)
        lam = su / sv

        # Safe denominators
        sigma2 = np.maximum(sigma**2, 1e-12)

        # Truncation
        t = mu / su
        trunc = np.maximum(norm.cdf(t), 1e-12)
        trunc_pdf = norm.pdf(t)

        # Terms
        z = (eps + mu) / sigma
        pdf_z = norm.pdf(z)

        arg = -(t - lam * ((eps + mu) / sigma))
        cdf_term = np.maximum(norm.cdf(arg), 1e-12)
        pdf_over_cdf = norm.pdf(arg) / cdf_term

        # ------------------ d/d β ------------------
        dLL_db = -(pdf_z / sigma +
                   lam * pdf_over_cdf / sigma)[:, None] * self.X
        g_beta = np.sum(dLL_db, axis=0)

        # ------------------ d/d μ ------------------
        dLL_dmu = (
            -(eps + mu) * pdf_z / sigma2
            + pdf_over_cdf * (1/su + lam*(eps + mu)/sigma2)
            - trunc_pdf/(trunc * su)
        )
        g_mu = -np.sum(dLL_dmu)

        # ------------------ d/d log σ_v ------------------
        d_sigma_dsv = sv / sigma
        d_lam_dsv = -su / (sv**2 + 1e-12)

        term_sv = (
            -(sv / sigma)
            + pdf_z * z * d_sigma_dsv
            + pdf_over_cdf *
              (-(eps + mu) * d_lam_dsv / sigma +
               lam*(eps + mu) * d_sigma_dsv / sigma2)
        )
        g_sv = -np.sum(term_sv) * sv

        # ------------------ d/d log σ_u ------------------
        d_sigma_dsu = su / sigma
        d_lam_dsu = 1 / (sv + 1e-12)

        term_su = (
            -(su / sigma)
            + pdf_z * z * d_sigma_dsu
            + pdf_over_cdf *
              (-(eps + mu) * d_lam_dsu / sigma +
               lam*(eps + mu) * d_sigma_dsu / sigma2)
            - (t * trunc_pdf / trunc)
        )
        g_su = -np.sum(term_su) * su

        return np.concatenate([g_beta, [g_mu, g_sv, g_su]])

    # ------------------------------------------------------------------
    # Fit model
    # ------------------------------------------------------------------
    def fit(self):
        # OLS start
        beta_ols = np.linalg.lstsq(self.X, self.y, rcond=None)[0]
        resid = self.y - self.X @ beta_ols
        sigma_ols = float(np.std(resid) + 1e-6)

        theta0 = np.concatenate([
            beta_ols,
            [0.0],                        # μ init
            [np.log(sigma_ols * 0.5)],    # σ_v init
            [np.log(sigma_ols * 0.7)]     # σ_u init
        ])

        # Optimisation
        res = minimize(
            self._loglik, theta0,
            jac=self._gradient,
            method="L-BFGS-B",
            options={"maxiter": 500, "disp": False}
        )

        # Fallback without gradient
        if not res.success:
            res = minimize(
                self._loglik, theta0,
                method="L-BFGS-B",
                options={"maxiter": 500, "disp": False}
            )

        self.res = res
        self.theta = res.x
        self._postprocess()
        return self

    # ------------------------------------------------------------------
    # Post-processing: JLMS inefficiency estimator
    # ------------------------------------------------------------------
    def _postprocess(self):
        beta, mu, sv, su = self._unpack(self.theta)

        self.beta = beta
        self.mu = mu
        self.sigma_v = sv
        self.sigma_u = su

        eps = self.y - self.X @ beta
        sigma2 = sv**2 + su**2

        # JLMS estimator for truncated-normal u
        mu_star = su**2 * (eps + mu) / sigma2
        sigma_star = (sv * su) / np.sqrt(sigma2)

        z = mu_star / np.maximum(sigma_star, 1e-12)
        ratio = norm.pdf(z) / np.maximum(norm.cdf(z), 1e-12)

        u_hat = mu_star + sigma_star * ratio
        self.u_hat = np.maximum(u_hat, 0.0)

        # TE
        self.TE = np.exp(-self.u_hat)

        # Frontier
        self.frontier = self.X @ beta

    # ------------------------------------------------------------------
    def summary(self):
        return {
            "beta": self.beta,
            "mu": self.mu,
            "sigma_v": self.sigma_v,
            "sigma_u": self.sigma_u,
            "TE_mean": float(np.mean(self.TE)),
            "TE_median": float(np.median(self.TE)),
            "log_likelihood": -self.res.fun,
            "converged": self.res.success
        }
