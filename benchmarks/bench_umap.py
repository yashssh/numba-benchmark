"""
UMAP benchmark, adapted from
https://github.com/lmcinnes/umap/blob/master/umap/distances.py
"""

import numpy as np
import numba
import umap

n = 1000000

arr1 = np.random.RandomState(0).rand(n).astype(np.float64)
arr2 = np.random.RandomState(1).rand(n).astype(np.float64)
x = np.random.randint
y = np.random.randint


spatial_data = np.random.randn(2, n).astype(np.float64)
v = np.abs(np.random.randn(n))


_mock_identity = np.eye(2, dtype=np.float64)
_mock_cost = 1.0 - _mock_identity
_mock_ones = np.ones(2, dtype=np.float64)

def setup():
    
    global run_umap, euclidean, euclidean_grad, standardised_euclidean, standardised_euclidean_grad, canberra_grad, yule, cosine_grad, correlation, \
    hellinger_grad, ll_dirichlet, log_beta, log_single_beta, approx_log_Gamma, ll_dirichlet_fast_math, symmetric_kl_grad, correlation_grad, spherical_gaussian_energy_grad, \
    diagonal_gaussian_energy_grad
    
    def run_umap(arr):
        fit = umap.UMAP()
        return fit.fit_transform(arr)

    @numba.njit(fastmath=True)
    def euclidean(x, y):
            r"""Standard euclidean distance.

            ..math::
                D(x, y) = \sqrt{\sum_i (x_i - y_i)^2}
            """
            result = 0.0
            for i in range(x.shape[0]):
                result += (x[i] - y[i]) ** 2
            return np.sqrt(result)


    @numba.njit(fastmath=True)
    def euclidean_grad(x, y):
        r"""Standard euclidean distance and its gradient.

        ..math::
            D(x, y) = \sqrt{\sum_i (x_i - y_i)^2}
            \frac{dD(x, y)}{dx} = (x_i - y_i)/D(x,y)
        """
        result = 0.0
        for i in range(x.shape[0]):
            result += (x[i] - y[i]) ** 2
        d = np.sqrt(result)
        grad = (x - y) / (1e-6 + d)
        return d, grad


    @numba.njit()
    def standardised_euclidean(x, y, sigma=_mock_ones):
        r"""Euclidean distance standardised against a vector of standard
        deviations per coordinate.

        ..math::
            D(x, y) = \sqrt{\sum_i \frac{(x_i - y_i)**2}{v_i}}
        """
        result = 0.0
        for i in range(x.shape[0]):
            result += ((x[i] - y[i]) ** 2) / sigma[i]
        return np.sqrt(result)


    @numba.njit(fastmath=True)
    def standardised_euclidean_grad(x, y, sigma=_mock_ones):
        r"""Euclidean distance standardised against a vector of standard
        deviations per coordinate with gradient.

        ..math::
            D(x, y) = \sqrt{\sum_i \frac{(x_i - y_i)**2}{v_i}}
        """
        result = 0.0
        for i in range(x.shape[0]):
            result += (x[i] - y[i]) ** 2 / sigma[i]
        d = np.sqrt(result)
        grad = (x - y) / (1e-6 + d * sigma)
        return d, grad

    @numba.njit()
    def canberra_grad(x, y):
        result = 0.0
        grad = np.zeros(x.shape)
        for i in range(x.shape[0]):
            denominator = np.abs(x[i]) + np.abs(y[i])
            if denominator > 0:
                result += np.abs(x[i] - y[i]) / denominator
                grad[i] = (
                    np.sign(x[i] - y[i]) / denominator
                    - np.abs(x[i] - y[i]) * np.sign(x[i]) / denominator**2
                )

        return result, grad

    @numba.njit()
    def yule(x, y):
        num_true_true = 0.0
        num_true_false = 0.0
        num_false_true = 0.0
        for i in range(x.shape[0]):
            x_true = x[i] != 0
            y_true = y[i] != 0
            num_true_true += x_true and y_true
            num_true_false += x_true and (not y_true)
            num_false_true += (not x_true) and y_true

        num_false_false = x.shape[0] - num_true_true - num_true_false - num_false_true

        if num_true_false == 0.0 or num_false_true == 0.0:
            return 0.0
        else:
            return (2.0 * num_true_false * num_false_true) / (
                num_true_true * num_false_false + num_true_false * num_false_true
            )

    @numba.njit(fastmath=True)
    def cosine_grad(x, y):
        result = 0.0
        norm_x = 0.0
        norm_y = 0.0
        for i in range(x.shape[0]):
            result += x[i] * y[i]
            norm_x += x[i] ** 2
            norm_y += y[i] ** 2

        if norm_x == 0.0 and norm_y == 0.0:
            dist = 0.0
            grad = np.zeros(x.shape)
        elif norm_x == 0.0 or norm_y == 0.0:
            dist = 1.0
            grad = np.zeros(x.shape)
        else:
            grad = -(x * result - y * norm_x) / np.sqrt(norm_x**3 * norm_y)
            dist = 1.0 - (result / np.sqrt(norm_x * norm_y))

        return dist, grad

    @numba.njit()
    def correlation(x, y):
        mu_x = 0.0
        mu_y = 0.0
        norm_x = 0.0
        norm_y = 0.0
        dot_product = 0.0

        for i in range(x.shape[0]):
            mu_x += x[i]
            mu_y += y[i]

        mu_x /= x.shape[0]
        mu_y /= x.shape[0]

        for i in range(x.shape[0]):
            shifted_x = x[i] - mu_x
            shifted_y = y[i] - mu_y
            norm_x += shifted_x**2
            norm_y += shifted_y**2
            dot_product += shifted_x * shifted_y

        if norm_x == 0.0 and norm_y == 0.0:
            return 0.0
        elif dot_product == 0.0:
            return 1.0
        else:
            return 1.0 - (dot_product / np.sqrt(norm_x * norm_y))

    @numba.njit()
    def hellinger_grad(x, y):
        result = 0.0
        l1_norm_x = 0.0
        l1_norm_y = 0.0

        grad_term = np.empty(x.shape[0])

        for i in range(x.shape[0]):
            grad_term[i] = np.sqrt(x[i] * y[i])
            result += grad_term[i]
            l1_norm_x += x[i]
            l1_norm_y += y[i]

        if l1_norm_x == 0 and l1_norm_y == 0:
            dist = 0.0
            grad = np.zeros(x.shape)
        elif l1_norm_x == 0 or l1_norm_y == 0:
            dist = 1.0
            grad = np.zeros(x.shape)
        else:
            dist_denom = np.sqrt(l1_norm_x * l1_norm_y)
            dist = np.sqrt(1 - result / dist_denom)
            grad_denom = 2 * dist
            grad_numer_const = (l1_norm_y * result) / (2 * dist_denom**3)

            grad = (grad_numer_const - (y / grad_term * dist_denom)) / grad_denom

        return dist, grad
    
    @numba.njit()
    def approx_log_Gamma(x):
        if x == 1:
            return 0
        # x2= 1/(x*x);
        return x * np.log(x) - x + 0.5 * np.log(2.0 * np.pi / x) + 1.0 / (x * 12.0)

    @numba.njit()
    def log_beta(x, y):
        a = min(x, y)
        b = max(x, y)
        if b < 5:
            value = -np.log(b)
            for i in range(1, int(a)):
                value += np.log(i) - np.log(b + i)
            return value
        else:
            return approx_log_Gamma(x) + approx_log_Gamma(y) - approx_log_Gamma(x + y)


    @numba.njit()
    def log_single_beta(x):
        return np.log(2.0) * (-2.0 * x + 0.5) + 0.5 * np.log(2.0 * np.pi / x) + 0.125 / x

    @numba.njit()
    def ll_dirichlet(data1, data2):
        """The symmetric relative log likelihood of rolling data2 vs data1
        in n trials on a die that rolled data1 in sum(data1) trials.

        ..math::
            D(data1, data2) = DirichletMultinomail(data2 | data1)
        """

        n1 = np.sum(data1)
        n2 = np.sum(data2)

        log_b = 0.0
        self_denom1 = 0.0
        self_denom2 = 0.0

        for i in range(data1.shape[0]):
            if data1[i] * data2[i] > 0.9:
                log_b += log_beta(data1[i], data2[i])
                self_denom1 += log_single_beta(data1[i])
                self_denom2 += log_single_beta(data2[i])

            else:
                if data1[i] > 0.9:
                    self_denom1 += log_single_beta(data1[i])

                if data2[i] > 0.9:
                    self_denom2 += log_single_beta(data2[i])

        return np.sqrt(
            1.0 / n2 * (log_b - log_beta(n1, n2) - (self_denom2 - log_single_beta(n2)))
            + 1.0 / n1 * (log_b - log_beta(n2, n1) - (self_denom1 - log_single_beta(n1)))
        )

    @numba.njit(fastmath=True)
    def ll_dirichlet_fast_math(data1, data2):
        """The symmetric relative log likelihood of rolling data2 vs data1
        in n trials on a die that rolled data1 in sum(data1) trials.

        ..math::
            D(data1, data2) = DirichletMultinomail(data2 | data1)
        """

        n1 = np.sum(data1)
        n2 = np.sum(data2)

        log_b = 0.0
        self_denom1 = 0.0
        self_denom2 = 0.0

        for i in range(data1.shape[0]):
            if data1[i] * data2[i] > 0.9:
                log_b += log_beta(data1[i], data2[i])
                self_denom1 += log_single_beta(data1[i])
                self_denom2 += log_single_beta(data2[i])

            else:
                if data1[i] > 0.9:
                    self_denom1 += log_single_beta(data1[i])

                if data2[i] > 0.9:
                    self_denom2 += log_single_beta(data2[i])

        return np.sqrt(
            1.0 / n2 * (log_b - log_beta(n1, n2) - (self_denom2 - log_single_beta(n2)))
            + 1.0 / n1 * (log_b - log_beta(n2, n1) - (self_denom1 - log_single_beta(n1)))
        )

    @numba.njit(fastmath=True)
    def symmetric_kl_grad(x, y, z=1e-11):  # pragma: no cover
        """
        symmetrized KL divergence and its gradient

        """
        n = x.shape[0]
        x_sum = 0.0
        y_sum = 0.0
        kl1 = 0.0
        kl2 = 0.0

        for i in range(n):
            x[i] += z
            x_sum += x[i]
            y[i] += z
            y_sum += y[i]

        for i in range(n):
            x[i] /= x_sum
            y[i] /= y_sum

        for i in range(n):
            kl1 += x[i] * np.log(x[i] / y[i])
            kl2 += y[i] * np.log(y[i] / x[i])

        dist = (kl1 + kl2) / 2
        grad = (np.log(y / x) - (x / y) + 1) / 2

        return dist, grad

    @numba.njit()
    def correlation_grad(x, y):
        mu_x = 0.0
        mu_y = 0.0
        norm_x = 0.0
        norm_y = 0.0
        dot_product = 0.0

        for i in range(x.shape[0]):
            mu_x += x[i]
            mu_y += y[i]

        mu_x /= x.shape[0]
        mu_y /= x.shape[0]

        for i in range(x.shape[0]):
            shifted_x = x[i] - mu_x
            shifted_y = y[i] - mu_y
            norm_x += shifted_x**2
            norm_y += shifted_y**2
            dot_product += shifted_x * shifted_y

        if norm_x == 0.0 and norm_y == 0.0:
            dist = 0.0
            grad = np.zeros(x.shape)
        elif dot_product == 0.0:
            dist = 1.0
            grad = np.zeros(x.shape)
        else:
            dist = 1.0 - (dot_product / np.sqrt(norm_x * norm_y))
            grad = ((x - mu_x) / norm_x - (y - mu_y) / dot_product) * dist

        return dist, grad
    
    @numba.njit(fastmath=True)
    def spherical_gaussian_energy_grad(x, y):  # pragma: no cover
        mu_1 = x[0] - y[0]
        mu_2 = x[1] - y[1]

        sigma = np.abs(x[2]) + np.abs(y[2])
        sign_sigma = np.sign(x[2])

        dist = (mu_1**2 + mu_2**2) / (2 * sigma) + np.log(sigma) + np.log(2 * np.pi)
        grad = np.empty(3, np.float32)

        grad[0] = mu_1 / sigma
        grad[1] = mu_2 / sigma
        grad[2] = sign_sigma * (1.0 / sigma - (mu_1**2 + mu_2**2) / (2 * sigma**2))

        return dist, grad

    @numba.njit(fastmath=True)
    def diagonal_gaussian_energy_grad(x, y):  # pragma: no cover
        mu_1 = x[0] - y[0]
        mu_2 = x[1] - y[1]

        sigma_11 = np.abs(x[2]) + np.abs(y[2])
        sigma_12 = 0.0
        sigma_22 = np.abs(x[3]) + np.abs(y[3])

        det = sigma_11 * sigma_22
        sign_s1 = np.sign(x[2])
        sign_s2 = np.sign(x[3])

        if det == 0.0:
            # TODO: figure out the right thing to do here
            return mu_1**2 + mu_2**2, np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)

        cross_term = 2 * sigma_12
        m_dist = (
            np.abs(sigma_22) * (mu_1**2)
            - cross_term * mu_1 * mu_2
            + np.abs(sigma_11) * (mu_2**2)
        )

        dist = (m_dist / det + np.log(np.abs(det))) / 2.0 + np.log(2 * np.pi)
        grad = np.empty(6, dtype=np.float32)

        grad[0] = (2 * sigma_22 * mu_1 - cross_term * mu_2) / (2 * det)
        grad[1] = (2 * sigma_11 * mu_2 - cross_term * mu_1) / (2 * det)
        grad[2] = sign_s1 * (sigma_22 * (det - m_dist) + det * mu_2**2) / (2 * det**2)
        grad[3] = sign_s2 * (sigma_11 * (det - m_dist) + det * mu_1**2) / (2 * det**2)

        return dist, grad

class UmapBench:
    
    def setup(self):
        np.random.seed(44)
        self.data = np.random.rand(10000, 1000)
        run_umap(self.data)
        
    def time_umap(self):
        run_umap(self.data)

class Distances:

    def setup(self):
        # Warm up
        euclidean(arr1, arr2)
        euclidean_grad(arr1, arr2)
        standardised_euclidean(spatial_data[0], spatial_data[1], v)
        standardised_euclidean_grad(spatial_data[0], spatial_data[1], v)
        canberra_grad(spatial_data[0], spatial_data[1])
        yule(arr1, arr2)
        cosine_grad(arr1, arr2)
        correlation(arr1, arr2)
        hellinger_grad(arr1, arr2)
        ll_dirichlet(arr1, arr2)
        ll_dirichlet_fast_math(arr1, arr2)
        symmetric_kl_grad(arr1, arr2)
        correlation_grad(arr1, arr2)
        spherical_gaussian_energy_grad(arr1, arr2)
        diagonal_gaussian_energy_grad(arr1, arr2)

    def time_euclidean(self):
        euclidean(arr1, arr2)

    def time_euclidean_grad(self):
        euclidean_grad(arr1, arr2)

    def time_standardised_euclidean(self):
        standardised_euclidean(spatial_data[0], spatial_data[1], v)

    def time_standardised_euclidean_grad(self):
        standardised_euclidean_grad(spatial_data[0], spatial_data[1], v)

    def time_canberra_grad(self):
        canberra_grad(spatial_data[0], spatial_data[1])

    def time_yule(self):
        yule(arr1, arr2)

    def time_cosine_grad(self):
        cosine_grad(arr1, arr2)

    def time_correlation(self):
        correlation(arr1, arr2)

    def time_hellinger_grad(self):
        hellinger_grad(arr1, arr2)
    
    def time_ll_dirichlet(self):
        ll_dirichlet(arr1, arr2)
    
    def time_ll_dirichlet_fast_math(self):
        ll_dirichlet_fast_math(arr1, arr2)
        
    def time_symmetric_kl_grad(self):
        symmetric_kl_grad(arr1, arr2)
    
    def time_correlation_grad(self):
        correlation_grad(arr1, arr2)

    def time_spherical_gaussian_energy_grad(self):
        spherical_gaussian_energy_grad(arr1, arr2)

    def time_diagonal_gaussian_energy_grad(self):
        diagonal_gaussian_energy_grad(arr1, arr2)
