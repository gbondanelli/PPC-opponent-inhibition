from numpy import *
import sys
#sys.path.insert(0,'../encoding_decoding')
#from encdec import *
#from stattools import *
from dynamicstools import *
from numpy.linalg import multi_dot, inv
from scipy.stats import norm
from scipy.linalg import solve_continuous_lyapunov

class RecurrentNet:
    def __init__(self, J, sigma_input, t=None, R0=None, f=lambda x:x):
        self.N = J.shape[0]
        self.J = J
        self.sigma_input = sigma_input
        self.pars = {'N': self.N, 't': t, 'R0': R0, 'J': J, 'f': f}
        self.propagator = inv(eye(self.N) - self.J)

    def set_input(self, I):
        if len(I.shape) == 1:
            assert I.shape[0] == self.N
            nsteps = len(self.pars['t'])
            self.input = outer(I,ones(nsteps))
        elif len(I.shape) == 2:
            nsteps = len(self.pars['t'])
            assert I.shape[0]==N and I.shape[1]==nsteps
            self.input = I
        self.pars['I'] = self.input
        self.pars['sigma'] = self.sigma_input

    def simulate_net(self):
        self.response = simulate_rate_model(self.pars)
        return self.response

    def covariance(self):
        self.covmat = solve_continuous_lyapunov(eye(self.N) - self.J, self.sigma_input ** 2 * eye(self.N))
        self.covmat_sh = diag(diag(self.covmat))
        return self.covmat

    def covariance_steadynoise(self, var = None):
        if var is None:
            self.covmat_steady = self.sigma_input ** 2 * self.propagator @ self.propagator.T
        else:
            self.covmat_steady = self.propagator @ diag(var) @ self.propagator.T
        self.covmat_steady_sh = diag(diag(self.covmat_steady))
        return self.covmat_steady

    def compute_acc_analytical(self, delta_input, idx = None):
        if idx is None:
            idx = list(range(self.N))
        dh = delta_input
        dh = dh/linalg.norm(dh)
        dm = self.propagator @ dh
        dm = dm[idx]
        C = self.covmat[idx][:,idx]
        C_sh = self.covmat_sh[idx][:,idx]
        SNR = dm.T @ inv(C) @ dm
        SNR_sh = dm.T @ inv(C_sh) @ dm
        # compute performance optimal decoder as Phi(√(SNR)/2), where Phi(x) = int_{-inf}^x Gauss(z)dz
        DecodAcc = norm.cdf(sqrt(SNR / 4))
        DecodAcc_sh = norm.cdf(sqrt(SNR_sh / 4))
        return DecodAcc, DecodAcc_sh

    def compute_acc_analytical_obsnoise(self, delta_input, cov, idx = None):
        if idx is None:
            idx = list(range(self.N))
        dh = delta_input
        dh = dh/linalg.norm(dh)
        dm = self.propagator @ dh
        dm = dm[idx]
        C = cov[idx][:,idx]
        C_sh = diag(diag(C))
        SNR = dm.T @ inv(C) @ dm
        SNR_sh = dm.T @ inv(C_sh) @ dm
        # compute performance optimal decoder as Phi(√(SNR)/2), where Phi(x) = int_{-inf}^x Gauss(z)dz
        DecodAcc = norm.cdf(sqrt(SNR / 4))
        DecodAcc_sh = norm.cdf(sqrt(SNR_sh / 4))
        return DecodAcc, DecodAcc_sh

    def compute_selectivity(self, neuron, stimulus, dh1, dh2):
        dh = dh1 - dh2
        idx = [neuron]
        decodacc, _ = self.compute_acc_analytical(dh, idx=idx)
        dm1 = self.propagator @ dh1
        dm2 = self.propagator @ dh2
        if stimulus == 0:
            selectivity = sign(dm1[neuron] - dm2[neuron]) * (decodacc-0.5)/0.5
        elif stimulus == 1:
            selectivity = sign(dm2[neuron] - dm1[neuron]) * (decodacc-0.5)/0.5
        return selectivity













##

