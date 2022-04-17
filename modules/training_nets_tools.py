import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from numpy import random
from matplotlib.pyplot import *
sys.path.insert(0, './modules/')
from modules import data_analysis_tools as dat
from scipy.stats import pearsonr


# Define network class
class RNN(nn.Module):
    def __init__(self, dim_rec, noise_std, dt, tau, g, signature, nonlinearity, psparse=1., wrec=None, use_W=False):
        """
        RNN model for training.
        dim_rec:        number of neurons
        noise_std:      neural noise standard deviation
        dt:             integration time step
        tau:            neural time scale
        g:              std dev of initial recurrent weights
        nonlinearity:   nonlinearity transforming state to rate; currently
                        only the logistic function is implemented.
        wrec:           specify the recurrent weights; this will overwrite `g`
        """
        super(RNN, self).__init__()
        self.dim_rec = dim_rec
        self.noise_std = noise_std
        self.dt = dt
        self.tau = tau
        self.signature = signature
        mask_EI = np.tile(signature, (self.dim_rec, 1))
        np.fill_diagonal(mask_EI, 0) # --- no self-coupling ---
        mask_sparse = random.choice([1, 0], size=(self.dim_rec,self.dim_rec), p=[psparse, 1-psparse])
        self.mask = torch.tensor(mask_EI*mask_sparse, dtype=torch.float32)
        self.nonlinearity = nonlinearity
        self.psparse = psparse
        self.use_W = use_W

        # Define and initilize parameters
        self.wrec = nn.Parameter(torch.Tensor(self.dim_rec, self.dim_rec))

        # Initialize parameters
        with torch.no_grad():
            # Recurrent weights
            if wrec is None:
                self.wrec.normal_(std=g / np.sqrt(self.dim_rec))
            else:
                if type(wrec) == np.ndarray:
                    wrec = torch.from_numpy(wrec)
                self.wrec.copy_(wrec)

    def effective_W(self,W):
        self.W_eff = torch.abs(W) * self.mask
        return self.W_eff

    def forward(self, input, h_init, rec_step):
        """b
        input:      tensor of shape (batch_size, #timesteps, dim_rec)
                    Important: the 3 dimensions need to be present,
                    even if they are of size 1.
        h_init:     initial state, must be of shape (batch_size, dim_rec)
        rec_step:   step size at which neural rates are recorded.
                    Note that they will be recorded whenever
                        i % rec_step == rec_step - 1
                    This means that the input and targets samples must be
                    shifted correspondingly (as done in the code below)!
        """
        batch_size = input.shape[0]
        seq_len = input.shape[1]

        noise = torch.randn(batch_size, seq_len, self.dim_rec, device=self.wrec.device)

        # Results vector
        # Only record every rec_step
        rec_seq_len = seq_len // rec_step
        rates = torch.zeros(batch_size, rec_seq_len, self.dim_rec, device=self.wrec.device)

        # Simulation loop
        h = h_init
        dt_tau = self.dt / self.tau

        if self.use_W:
            W_eff = self.wrec
        else:
            W_eff = self.effective_W(self.wrec)

        for i in range(seq_len):
            rec_input = (self.nonlinearity(h).matmul(W_eff.t()) + input[:, i, :])
            # func = nn.LeakyReLU()
            # rec_input = (torch.matmul(func(h),W.t()) + input[:, i, :])
            h = ((1 - dt_tau) * h
                 + dt_tau * rec_input
                 + np.sqrt(dt_tau) * self.noise_std * noise[:, i, :])
            # Save
            if i % rec_step == rec_step - 1:
                k = i // rec_step
                rates[:, k, :] = self.nonlinearity(h)

        return rates



class TrainNetOnTraces(RNN):

    def __init__(self, target_R_all, target_t, input_all, device):
        self.target_R_all   = target_R_all
        self.input_all  = input_all
        self.device     = device
        self.target_t   = target_t

    def set_net_params(self, dim_rec, noise_std, dt, tau, g, signature, nonlinearity, psparse=1., constraint=None, wrec=None):
        # Initialize network
        super(TrainNetOnTraces,self).__init__(dim_rec, noise_std, dt, tau, g, signature, nonlinearity, psparse, wrec)
        net = RNN(dim_rec, noise_std, dt, tau, g, signature, nonlinearity, psparse, wrec)
        net.to(device=self.device)
        self.net = net
        self.wrec_init = self.net.wrec.detach().cpu().numpy().copy()
        self.constraint = constraint

    def train_net(self, rec_step, n_epochs = 200, verbose = True, batch_size=64, n_targets_per_sample = 5, learning_rate = 0.01):

        dt_rec  = np.diff(self.target_t)[0]
        assert np.allclose(np.diff(self.target_t), dt_rec)
        dt      = dt_rec / rec_step
        t_max   = self.target_t[-1] + dt_rec
        t       = np.arange(0, t_max, dt)
        n_t     = len(t)

        loss_criterion  = torch.nn.MSELoss().to(self.device)
        optimizer       = torch.optim.Adam(self.net.parameters(), lr=learning_rate)

        n_targets       = len(self.target_t)
        n_t_per_sample  = n_targets_per_sample * rec_step
        max_idx         = n_targets - n_targets_per_sample - 1

        self.target_R_all   = np.float32(self.target_R_all)
        self.input_all      = np.float32(self.input_all)
        assert self.target_R_all.shape[0] == self.input_all.shape[0]
        n_traces            = self.target_R_all.shape[0]

        time0 = time.time()
        random_trace_for_training_epoch = random.permutation(list(range(n_traces))*n_epochs)
        self.losses = np.zeros(n_epochs*n_traces)
        self.converging = True

        for i_epoch in (tqdm(range(n_epochs*n_traces),colour='#de1da4') if verbose else range(n_epochs*n_traces)):
            which_trace = random_trace_for_training_epoch[i_epoch]
            target_R    = self.target_R_all[which_trace]
            input       = self.input_all[which_trace]

            # Inputs and targets for this batch
            t_batch         = np.zeros((batch_size, n_t_per_sample))
            h_init_batch    = np.zeros((batch_size, self.dim_rec), dtype=np.float32)
            input_batch     = np.zeros((batch_size, n_t_per_sample, self.dim_rec), dtype=np.float32)
            target_batch    = np.zeros((batch_size, n_targets_per_sample, self.dim_rec), dtype=np.float32)

            # Choose indices at random
            idxs = np.random.choice(max_idx, batch_size, replace=max_idx < batch_size)
            for i_batch in range(batch_size):
                idx = idxs[i_batch]

                # Initial condition
                # h_init_batch[i_batch] = self.inv_nonlinearity(target_R.T[idx])
                h_init_batch[i_batch] = target_R.T[idx] #  MODIFICATION

                # Times and input
                idx_t0 = idx * rec_step + 1
                idx_t1 = idx_t0 + n_t_per_sample
                t_batch[i_batch]        = t[idx_t0: idx_t1]
                input_batch[i_batch]    = input[idx_t0: idx_t1]

                # Target
                idx_0 = idx + 1
                idx_1 = idx_0 + n_targets_per_sample
                target_batch[i_batch] = target_R.T[idx_0: idx_1]

            # To torch, and allocate
            h_init_batch = torch.from_numpy(h_init_batch).to(self.device)
            input_batch  = torch.from_numpy(input_batch).to(self.device)
            target_batch = torch.from_numpy(target_batch).to(self.device)

            optimizer.zero_grad()
            output  = self.net(input_batch, h_init_batch, rec_step)

            loss    = loss_criterion(output, target_batch)



            if self.constraint is not None:
                if self.constraint['type'] == 'PearsonCorr':
                    cee, cie, cei, cii = self.constraint['values']
                    lee, lie, lei, lii = self.constraint['l']
                    S       = self.constraint['selectivity']
                    W       = abs( self.net.effective_W(self.net.wrec) )
                    S_tensor= torch.tensor(S, dtype=torch.float32)
                    weights     = dat.get_all_weights_by_type_for_tensors(W, self.signature)
                    selectivity = dat.get_all_weights_by_type_for_tensors(S_tensor, self.signature)
                    xee, xie, xei, xii = selectivity['w_ee'], selectivity['w_ie'], selectivity['w_ei'], selectivity['w_ii']
                    yee, yie, yei, yii = weights['w_ee'], weights['w_ie'], weights['w_ei'], weights['w_ii']
                    cee_model = dat.pearsonr_for_tensors(xee,yee)
                    cie_model = dat.pearsonr_for_tensors(xie,yie)
                    cei_model = dat.pearsonr_for_tensors(xei,yei)
                    cii_model = dat.pearsonr_for_tensors(xii,yii)
                    loss = loss + lee*(cee_model-cee)**2 + lie*(cie_model-cie)**2+ lei*(cei_model-cei)**2+ lii*(cii_model-cii)**2

                # if self.constraint['type'] == 'CorrEI':
                #     list_params = [p for p in self.net.parameters()]
                #     w = list_params[0].detach()
                #     print(self.net.wrec)
                #     loss = torch.tensor(0, dtype=torch.float32, requires_grad=True)+torch.tensor(abs(w[0,1]), dtype=torch.float32, requires_grad=True)

            loss.backward()
            optimizer.step()

            loss.detach_()
            output.detach_()

            self.losses[i_epoch] = loss.item()
            print(loss.item())

            # early stopping:
            if i_epoch > 200:
                last_losses = self.losses[i_epoch - 100: i_epoch]
                if (np.mean(last_losses) > 50) or (np.var(np.diff(last_losses)) < 1e-10):
                    if np.mean(last_losses > 50):
                        self.converging = False
                    print('Early stopped! Mean=', np.mean(last_losses), np.var(np.diff(last_losses)), self.converging)
                    break


        for i_trace in range(n_traces):
            rec_step = rec_step
            target_R = self.target_R_all[i_trace]
            input = self.input_all[i_trace]
            h_init_full = target_R.T[0][None]
            input_full = input[None]
            h_init_full = torch.from_numpy(np.float32(h_init_full)).to(self.device)
            input_full = torch.from_numpy(np.float32(input_full)).to(self.device)

            with torch.no_grad():
                rates_final = self.net(input_full, h_init_full, rec_step=rec_step)
                rates_final = rates_final.detach().cpu().numpy()
            if i_trace==0:
                R = rates_final
            else:
                R = np.concatenate((R,rates_final),axis=0)

        wrec_final = self.net.wrec.detach().cpu().numpy().copy()
        wrec_EI = abs(wrec_final) * self.net.mask.numpy()
        self.W_trained = wrec_EI
        self.R_trained = R
        self.state_net = dict(W_init    = self.wrec_init  if self.converging else np.nan,
                              W_EI      = wrec_EI         if self.converging else np.nan,
                              W         = self.W_trained  if self.converging else np.nan,
                              R         = self.R_trained  if self.converging else np.nan,
                              n_epochs  = n_epochs        if self.converging else np.nan,
                              lr        = learning_rate   if self.converging else np.nan,
                              losses    = self.losses     if self.converging else np.nan,
                              rec_step  = step            if self.converging else np.nan,
                              signature = self.signature  if self.converging else np.nan,
                              psparse   = self.psparse    if self.converging else np.nan)

def generate_low_pass_noise(shape, dt, tau=1., amplitude=1.):
    ampWN = np.sqrt(tau/dt)
    iWN = ampWN * np.random.randn(shape[0], shape[1])
    input = np.ones(shape)
    n_t = shape[0]
    for tt in range(1, n_t):
        input[tt] = iWN[tt] + (input[tt - 1] - iWN[tt]) * np.exp(- (dt/tau))
    input *= amplitude
    return input

def compute_fittedactivity_from_dataset(filename, which_trial, W, nonlinearity, N, noise_std, tau, rec_step, psparse=1.):

    df = pd.read_pickle(filename)
    data_training   = df.data_training
    input           = df.input
    signature       = df.signature
    g       = df.g
    theta   = df.theta
    dt      = df.deltat
    N       = len(signature)
    n_nets  = len(data_training)

    # nonlinearity = lambda x: 1 / (1 + torch.exp(-(x - theta)))
    f = lambda x: nonlinearity(x,theta)

    net         = RNN(N, noise_std, dt, tau, g, signature, f, psparse, W, use_W=True)
    h_init      = np.zeros(N)[None]
    input_this  = input[which_trial][None]
    h_init      = torch.from_numpy(np.float32(h_init))
    input_this  = torch.from_numpy(np.float32(input_this))

    rates = net(input_this, h_init, rec_step=rec_step)
    rates = rates.detach().numpy()[0].T
    return rates


def compute_fittedactivity_from_dataset_v2(df, which_trial, W, nonlinearity, N, noise_std, tau, rec_step, signature, \
                                           g, theta, dt, psparse=1.):

    data_training   = df.data_training
    input           = df.input
    N       = len(signature)
    n_nets  = len(data_training)

    # nonlinearity = lambda x: 1 / (1 + torch.exp(-(x - theta)))
    f = lambda x: nonlinearity(x,theta)

    net         = RNN(N, noise_std, dt, tau, g, signature, f, psparse, W, use_W=True)
    h_init      = np.zeros(N)[None]
    input_this  = input[which_trial][None]
    h_init      = torch.from_numpy(np.float32(h_init))
    input_this  = torch.from_numpy(np.float32(input_this))

    rates = net(input_this, h_init, rec_step=rec_step)
    rates = rates.detach().numpy()[0].T
    return rates

##

