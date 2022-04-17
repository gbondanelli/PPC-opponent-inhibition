import pandas as pd
import numpy as np
from matplotlib.pyplot import hist
from scipy.stats import entropy
import torch
import sys
#sys.path.insert(0,'../encoding_decoding')
#from encdec import *

def CenterOfMass_singletrace(DF):
    DF_norm = DF/np.sum(DF)
    t = np.arange(0,len(DF))
    t_COM = np.sum(DF_norm @ t)
    return t_COM

def CenterOfMass(activity):
    """
    :param activity: #neurons x #timesteps
    :return: array of center of masses
    """
    N = activity.shape[0]
    t_COM_array = np.empty(N)
    for i in range(N):
        t_COM_array[i] = CenterOfMass_singletrace(activity[i,:])
    return t_COM_array

def sort_by_COM(activity):
    t_COM = CenterOfMass(activity)
    t_COM_sorted = np.sort(t_COM)
    activity_sorted = activity[np.argsort(t_COM),:]
    return activity_sorted, t_COM, t_COM_sorted

def sort_EI_activity(df, mode='EthenI'):
    """
    :param df: pandas dataset
    :param mode: sort E activity and I activity separately, then stack I activity below E activity
    :return: sorted activity
    """
    if mode == 'EthenI':
        df_E = df[df.type == 'pyramidal']
        df_I = df[df.type == 'non pyramidal']

        activity_E_right = np.stack(df_E.Ca_trial_mean_bR.values, axis=0)
        activity_I_right = np.stack(df_I.Ca_trial_mean_bR.values, axis=0)
        activity_E_left = np.stack(df_E.Ca_trial_mean_wL.values, axis=0)
        activity_I_left = np.stack(df_I.Ca_trial_mean_wL.values, axis=0)

        activity_E_right_sorted, _, _ = sort_by_COM(activity_E_right)
        activity_I_right_sorted, _, _ = sort_by_COM(activity_I_right)
        activity_E_left_sorted, _, _ = sort_by_COM(activity_E_left)
        activity_I_left_sorted, _, _ = sort_by_COM(activity_I_left)

        activity_right_sorted = np.vstack((activity_E_right_sorted, activity_I_right_sorted))
        activity_left_sorted = np.vstack((activity_E_left_sorted, activity_I_left_sorted))
        return activity_left_sorted, activity_right_sorted

def normalize_data_by_peak(df, columns = ['Ca_trial_mean_bR','Ca_trial_mean_wL']):
    for column in columns:
        df[column] = df[column].apply(lambda x: x / np.amax(x))
    return df

def normalize_data_by_peak_of_preferred_trial(df, columns = ['Ca_trial_mean_bR','Ca_trial_mean_wL']):
    for index, neuron in df.iterrows():
        if neuron.selectivity_MI=='Right':
            Max = np.amax(neuron['Ca_trial_mean_bR'])
            for column in columns:
                df.at[index, column] = neuron[column]/Max
        if neuron.selectivity_MI=='Left':
            Max = np.amax(neuron['Ca_trial_mean_wL'])
            for column in columns:
                df.at[index, column] = neuron[column]/Max
        if neuron.selectivity_MI == 'Non':
            Max = np.amax([neuron['Ca_trial_mean_bR'], neuron['Ca_trial_mean_wL']])
            for column in columns:
                df.at[index, column] = neuron[column] / Max
    return df


# def normalize_data_by_peak_of_preferred_trial(df, columns = ['Ca_trial_mean_bR','Ca_trial_mean_wL']):
#     for index, neuron in df.iterrows():
#         if neuron.select_idx_MI>0:
#             Max = np.amax(neuron['Ca_trial_mean_bR'])
#             for column in columns:
#                 df.at[index, column] = neuron[column]/Max
#         if neuron.select_idx_MI<=0:
#             Max = np.amax(neuron['Ca_trial_mean_wL'])
#             for column in columns:
#                 df.at[index, column] = neuron[column]/Max
#     return df

# def normalize_data_by_peak_of_preferred_trial(df, columns = ['Ca_trial_mean_bR','Ca_trial_mean_wL']):
#     for index, neuron in df.iterrows():
#         Max = np.amax([neuron['Ca_trial_mean_bR'], neuron['Ca_trial_mean_wL']])
#         for column in columns:
#             df.at[index, column] = neuron[column] / Max
#     return df

def indices_neurons_EIchoice(signature, idx_c=None):
    """
    :param idx_c: indices of choice_1 and choice_2 (+NS)  neurons (pooled over E and I), as list containing 2 (or 3) lists
    :param signature: neuron type: 1 if E, -1 if I
    :return: indices of C1 or C2 selective neurons separated for E and I
    """

    idx_e = [i for i, type in enumerate(signature) if type == 1]
    idx_i = [i for i, type in enumerate(signature) if type == -1]
    result_dict = dict(idx_e=idx_e, idx_i=idx_i)

    if idx_c is None:
        return result_dict

    elif idx_c is not None:
        idx_1 = idx_c[0]
        idx_2 = idx_c[1]
        idx_e1 = [i for i, type in enumerate(signature) if i in idx_1 and type == 1]
        idx_e2 = [i for i, type in enumerate(signature) if i in idx_2 and type == 1]
        idx_i1 = [i for i, type in enumerate(signature) if i in idx_1 and type == -1]
        idx_i2 = [i for i, type in enumerate(signature) if i in idx_2 and type == -1]
        result_dict.update(dict(idx_e1=idx_e1, idx_e2=idx_e2, idx_i1=idx_i1, idx_i2=idx_i2))

        if len(idx_c)==2:
            return result_dict

        elif len(idx_c) == 3:
            idx_0 = idx_c[2] # non-choice selective neurons
            idx_e0 = [i for i, type in enumerate(signature) if i in idx_0 and type == 1]
            idx_i0 = [i for i, type in enumerate(signature) if i in idx_0 and type == -1]
            result_dict.update(dict(idx_e0=idx_e0, idx_i0=idx_i0))
            return result_dict

def get_all_weights_by_type(W, signature, idx_c=None):
    dict_idx = indices_neurons_EIchoice(signature, idx_c)

    idx_e = dict_idx['idx_e']
    idx_i = dict_idx['idx_i']
    w_ee = W[idx_e][:,idx_e].ravel().tolist()
    w_ie = W[idx_i][:, idx_e].ravel().tolist()
    w_ei = W[idx_e][:, idx_i].ravel().tolist()
    w_ii = W[idx_i][:, idx_i].ravel().tolist()
    result_dict = dict(w_ee=w_ee,w_ie=w_ie,w_ei=w_ei,w_ii=w_ii)

    if idx_c is None:
        return result_dict

    elif idx_c is not None:
        idx_e1 = dict_idx['idx_e1']
        idx_e2 = dict_idx['idx_e2']
        idx_i1 = dict_idx['idx_i1']
        idx_i2 = dict_idx['idx_i2']

        w_ee_in = W[idx_e1][:,idx_e1].ravel().tolist()+W[idx_e2][:,idx_e2].ravel().tolist()
        w_ee_out = W[idx_e1][:, idx_e2].ravel().tolist() + W[idx_e2][:, idx_e1].ravel().tolist()

        w_ie_in = W[idx_i1][:, idx_e1].ravel().tolist() + W[idx_i2][:, idx_e2].ravel().tolist()
        w_ie_out = W[idx_i1][:, idx_e2].ravel().tolist() + W[idx_i2][:, idx_e1].ravel().tolist()

        w_ei_in = W[idx_e1][:, idx_i1].ravel().tolist() + W[idx_e2][:, idx_i2].ravel().tolist()
        w_ei_out = W[idx_e1][:, idx_i2].ravel().tolist() + W[idx_e2][:, idx_i1].ravel().tolist()

        w_ii_in = W[idx_i1][:, idx_i1].ravel().tolist() + W[idx_i2][:, idx_i2].ravel().tolist()
        w_ii_out = W[idx_i1][:, idx_i2].ravel().tolist() + W[idx_i2][:, idx_i1].ravel().tolist()

        result_dict.update(dict(w_ee_in=w_ee_in, w_ee_out=w_ee_out,\
                        w_ie_in=w_ie_in, w_ie_out=w_ie_out,\
                        w_ei_in=w_ei_in, w_ei_out=w_ei_out,\
                        w_ii_in=w_ii_in, w_ii_out=w_ii_out))

        if len(idx_c)==2:
            return result_dict

        elif len(idx_c)==3:
            idx_e0 = dict_idx['idx_e0']
            idx_i0 = dict_idx['idx_i0']
            w_ee_ns = W[idx_e1+idx_e2][:,idx_e0].ravel().tolist() +\
                      W[idx_e0][:,idx_e1+idx_e2].ravel().tolist() +\
                      W[idx_e0][:,idx_e0].ravel().tolist()
            w_ie_ns = W[idx_i1+idx_i2][:,idx_e0].ravel().tolist() + \
                      W[idx_i0][:,idx_e1+idx_e2].ravel().tolist() + \
                      W[idx_i0][:,idx_e0].ravel().tolist()
            w_ei_ns = W[idx_e1+idx_e2][:,idx_i0].ravel().tolist() + \
                      W[idx_e0][:,idx_i1+idx_i2].ravel().tolist() + \
                      W[idx_e0][:,idx_i0].ravel().tolist()
            w_ii_ns = W[idx_i1+idx_i2][:,idx_i0].ravel().tolist() + \
                      W[idx_i0][:,idx_i1+idx_i2].ravel().tolist() + \
                      W[idx_i0][:,idx_i0].ravel().tolist()

            result_dict.update(dict(w_ee_ns = w_ee_ns, \
                                    w_ie_ns=w_ie_ns, \
                                    w_ei_ns=w_ei_ns, \
                                    w_ii_ns=w_ii_ns))
            return result_dict

def get_all_weights_by_type_for_tensors(W, signature, idx_c=None):
    dict_idx = indices_neurons_EIchoice(signature, idx_c)

    idx_e = dict_idx['idx_e']
    idx_i = dict_idx['idx_i']
    w_ee = W[idx_e][:,idx_e].view(-1)
    w_ie = W[idx_i][:, idx_e].view(-1)
    w_ei = W[idx_e][:, idx_i].view(-1)
    w_ii = W[idx_i][:, idx_i].view(-1)
    result_dict = dict(w_ee=w_ee,w_ie=w_ie,w_ei=w_ei,w_ii=w_ii)

    if idx_c is None:
        return result_dict

    elif idx_c is not None:
        idx_e1 = dict_idx['idx_e1']
        idx_e2 = dict_idx['idx_e2']
        idx_i1 = dict_idx['idx_i1']
        idx_i2 = dict_idx['idx_i2']

        w_ee_in  = torch.cat(W[idx_e1][:,idx_e1].view(-1) ,  W[idx_e2][:, idx_e2].view(-1))
        w_ee_out = torch.cat(W[idx_e1][:, idx_e2].view(-1),  W[idx_e2][:, idx_e1].view(-1))

        w_ie_in  = torch.cat(W[idx_i1][:, idx_e1].view(-1),  W[idx_i2][:, idx_e2].view(-1))
        w_ie_out = torch.cat(W[idx_i1][:, idx_e2].view(-1),  W[idx_i2][:, idx_e1].view(-1))

        w_ei_in  = torch.cat(W[idx_e1][:, idx_i1].view(-1),  W[idx_e2][:, idx_i2].view(-1))
        w_ei_out = torch.cat(W[idx_e1][:, idx_i2].view(-1),  W[idx_e2][:, idx_i1].view(-1))

        w_ii_in  = torch.cat(W[idx_i1][:, idx_i1].view(-1),  W[idx_i2][:, idx_i2].view(-1))
        w_ii_out = torch.cat(W[idx_i1][:, idx_i2].view(-1),  W[idx_i2][:, idx_i1].view(-1))

        result_dict.update(dict(w_ee_in=w_ee_in, w_ee_out=w_ee_out,\
                        w_ie_in=w_ie_in, w_ie_out=w_ie_out,\
                        w_ei_in=w_ei_in, w_ei_out=w_ei_out,\
                        w_ii_in=w_ii_in, w_ii_out=w_ii_out))

        if len(idx_c)==2:
            return result_dict

        elif len(idx_c)==3:
            idx_e0 = dict_idx['idx_e0']
            idx_i0 = dict_idx['idx_i0']
            w_ee_ns = torch.cat( (W[idx_e1+idx_e2][:,idx_e0].view(-1), W[idx_e0][:,idx_e1+idx_e2].view(-1), W[idx_e0][:,idx_e0].view(-1)) )
            w_ie_ns = torch.cat( (W[idx_i1+idx_i2][:,idx_e0].view(-1), W[idx_i0][:,idx_e1+idx_e2].view(-1), W[idx_i0][:,idx_e0].view(-1)) )
            w_ei_ns = torch.cat( (W[idx_e1+idx_e2][:,idx_i0].view(-1), W[idx_e0][:,idx_i1+idx_i2].view(-1), W[idx_e0][:,idx_i0].view(-1)) )
            w_ii_ns = torch.cat( (W[idx_i1+idx_i2][:,idx_i0].view(-1), W[idx_i0][:,idx_i1+idx_i2].view(-1), W[idx_i0][:,idx_i0].view(-1)) )

            result_dict.update(dict(w_ee_ns = w_ee_ns, \
                                    w_ie_ns=w_ie_ns, \
                                    w_ei_ns=w_ei_ns, \
                                    w_ii_ns=w_ii_ns))
            return result_dict

def modify_2d_array(W, Y, idx_pre, idx_post):
    """ the function takes the matrix W and substitutes on the sub rectangular matrix specified by
    idx_pre/post the rectangular matrix Y
    :param idx_post: list
    :param idx_pre: list
    :param W: matrix to modify
    :param Y: elements to put
    :return:
    """
    W_modif = np.copy(W)
    n_pre = len(idx_pre)
    n_post= len(idx_post)
    for i_post in range(n_post):
        for j_pre in range(n_pre):
            W_modif[idx_post[i_post], idx_pre[j_pre]] = Y[i_post, j_pre]
    return W_modif

def perturb_connectivity_weights(W, idx_pre, idx_post, value, scale=False):
    """
    :param W: connectivity matrix to perturb
    :param idx_pre: indices of pre-synaptic neurons to perturb
    :param idx_post: indices of post-synaptic neurons to perturb
    :param value: value for setting or scaling a given set of connections
    :param scale: default False. If False 'value' corresponds to the precise value to set the connections,
    if True 'value' correspond to the value to scale the connections
    :return: perturbed connectivity matrix
    """
    assert len(idx_pre)==len(idx_post)
    n = len(idx_pre)
    for i in range(n):
        idx_pre_this    = idx_pre[i]
        idx_post_this   = idx_post[i]
        value_this      = value[i]
        if not scale:
            Y = value_this * np.ones(W[idx_post_this][:, idx_pre_this].shape)
            W = modify_2d_array(W, Y, idx_pre_this, idx_post_this)
        elif scale:
            Y = value_this * W[idx_post_this][:, idx_pre_this]
            W = modify_2d_array(W, Y, idx_pre_this, idx_post_this)
    return W

def ridge_to_background(R, how_many_time_steps_around):
    """
    :param how_many_time_steps_around:
    :param R:
    :return:
    """
    N = R.shape[0]
    t = np.arange(R.shape[1])
    _, t_COM, _ = sort_by_COM(R)
    result = np.empty(N)
    C = how_many_time_steps_around
    for i in range(N):
        t_COM_idx = int(np.round(t_COM[i]))
        ridge_idx = np.arange(max(t_COM_idx-C,0), min(t_COM_idx+C,R.shape[1]))
        background_idx = np.where(np.isin(t,ridge_idx)==False)[0].tolist()
        ridge_mean = R[i,ridge_idx].mean()
        background_mean = R[i, background_idx].mean()
        result[i] = ridge_mean/background_mean
    return result

def SI(R, how_many_time_steps_around):
    T = R.shape[1]
    bins = np.linspace(0,T,20).tolist()
    RtB = ridge_to_background(R, how_many_time_steps_around).mean()
    RtB = np.log(RtB)
    _, t_COM, _ = sort_by_COM(R)
    p = hist(t_COM, bins, density=False)[0]
    p = p/np.sum(p)
    H = entropy(p)
    return H + RtB

def pearsonr_for_tensors(x,y,exclude_zeros=True):
    if exclude_zeros:
        x = x[y != 0]
        y = y[y != 0]
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    pearson_coeff = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return pearson_coeff

def perturb_weights_type(W, c, type, signature, idx_c):
    weights = get_all_weights_by_type(abs(W), signature, idx_c)
    w_in = np.array(weights['w_'+type+'_in'])
    w_in = w_in[w_in != 0]
    w_out = np.array(weights['w_'+type+'_out'])
    w_out = w_out[w_out != 0]
    m_in = np.mean(w_in)
    m_out = np.mean(w_out)

    din = c
    dout = -din * m_in / m_out

    dict_idx = indices_neurons_EIchoice(signature, idx_c)
    E = dict_idx['idx_e']
    I = dict_idx['idx_i']
    E1 = dict_idx['idx_e1']
    E2 = dict_idx['idx_e2']
    I1 = dict_idx['idx_i1']
    I2 = dict_idx['idx_i2']

    if type=='ie':
        idx_pre = [E1, E2, E1, E2]
        idx_post = [I1, I2, I2, I1]
    elif type == 'ei':
        idx_pre = [I1, I2, I1, I2]
        idx_post = [E1, E2, E2, E1]
    else:
        print('Type should be either ei or ie !')
    value = [1 + din, 1 + din, 1 + dout, 1 + dout]
    W_pert = perturb_connectivity_weights(W, idx_pre, idx_post, value, True)
    return W_pert

def r2(yfit,y):
    return 1 - np.mean((yfit - y) ** 2) / np.var(y)

def compute_R2(Xfit, X):
    # X has shape num_traces x timepoints x num_cells
    assert Xfit.shape == X.shape
    R2 = np.array([r2(Xfit[0].ravel(), X[0].ravel()),
                   r2(Xfit[1].ravel(), X[1].ravel())])
    num_traces, T, N = X.shape
    R2_singlecells = np.empty(num_traces*N)
    ii=0
    for i_trace in range(num_traces):
        for i_cell in range(N):
            R2_singlecells[ii] = r2(Xfit[i_trace,:,i_cell], X[i_trace,:,i_cell])
            ii+=1
    return R2, R2_singlecells

def compute_decoding_accuracy_perturbations(rates_pert_0, rates_pert_1, ntrials, sigma, T1, T2):
    R_pert_0 = np.tile(rates_pert_0, (ntrials, 1, 1))
    R_pert_0 = R_pert_0 + random.normal(0, sigma / np.sqrt(rates_pert_0.shape[0]), R_pert_0.shape)
    R_pert_1 = np.tile(rates_pert_1, (ntrials, 1, 1))
    R_pert_1 = R_pert_1 + random.normal(0, sigma / np.sqrt(rates_pert_1.shape[0]), R_pert_1.shape)

    accuracy_temp = np.empty(len(np.arange(T1, T2)))
    ii = 0
    for i_t in range(T1, T2):
        R = np.hstack((R_pert_0[:, :, i_t + 1].T, R_pert_1[:, :, i_t + 1].T))
        Stim = np.hstack((np.ones(ntrials), -np.ones(ntrials)))
        clf = LinearSVM(R, Stim)
        clf.set_K(5)
        accuracy_temp[ii] = np.mean(clf.get_accuracy())
        ii += 1
    return np.mean(accuracy_temp)


def compute_decoding_accuracy_perturbations_v2(rates_pert_0, rates_pert_1, ntrials, sigma, T1=None, T2=None):
    rates_pert_0 = rates_pert_0.ravel()
    rates_pert_1 = rates_pert_1.ravel()
    R_pert_0 = np.tile(rates_pert_0, (ntrials, 1))
    R_pert_0 = R_pert_0 + random.normal(0, sigma / np.sqrt(len(rates_pert_0)), R_pert_0.shape)
    R_pert_1 = np.tile(rates_pert_1, (ntrials, 1))
    R_pert_1 = R_pert_1 + random.normal(0, sigma / np.sqrt(len(rates_pert_0)), R_pert_1.shape)

    R = np.hstack((R_pert_0.T, R_pert_1.T))
    Stim = np.hstack((np.ones(ntrials), -np.ones(ntrials)))
    clf = LinearSVM(R, Stim)
    clf.set_K(5)
    return np.mean(clf.get_accuracy())













##

