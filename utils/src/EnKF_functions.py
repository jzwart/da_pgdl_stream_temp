# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 10:30:42 2020

@author: jzwart
functions for updating LSTM states using ensemble Kalman filter 
"""
import numpy as np 
import sys
sys.path.insert(1, '5_pgdl_pretrain/src')
from LSTMDA import *

def combine_lstm_states(
        preds,
        h, 
        c,
        n_segs,
        n_states_est, 
        n_en,
        hidden_units
):
    '''
    Combining lstm states for updating in EnKF. Should always be in the order of preds, h, c. 
    
    '''
    out = np.empty((n_states_est, n_en))
    
    for i in range(n_en):
        cur_idxs = np.repeat(i, n_segs) + n_en * (np.arange(0,n_segs)) # index of predictions for all segments for current ensemble 
        cur_preds = preds[cur_idxs,:].reshape(n_segs)
        cur_h = h[cur_idxs,:].reshape(n_segs*hidden_units)
        cur_c = c[cur_idxs,:].reshape(n_segs*hidden_units)
        out[:,i] = np.concatenate((cur_preds, cur_h, cur_c)).reshape((n_states_est))
        
    return out

def combine_rgcn_states(
        preds,
        h, 
        c,
        n_segs,
        n_states_est, 
        n_en,
        hidden_layers
):
    '''
    Combining rgcn states for updating in EnKF. Should always be in the order of preds, h, c. 
    
    '''
    out = np.empty((n_states_est, n_en))
    
    for i in range(n_en):
        cur_idxs = np.repeat(i, n_segs) + n_en * (np.arange(0,n_segs)) # index of predictions for all segments for current ensemble 
        cur_preds = preds[cur_idxs,:].reshape(n_segs)
        cur_h = h[cur_idxs,:,:].reshape(n_segs*hidden_layers)
        cur_c = c[cur_idxs,:,:].reshape(n_segs*hidden_layers)
        out[:,i] = np.concatenate((cur_preds, cur_h, cur_c)).reshape((n_states_est))
        
    return out

def get_forecast_preds(
        preds,
        n_segs,
        n_states_obs, 
        n_en,
        f_horizon
):
    '''
    formatting LSTM forecasts for storing 
    
    '''
    out = np.empty((n_states_obs, f_horizon, n_en))
    
    for i in range(n_segs):
        cur_idxs = np.arange(0,n_en) + (i * n_en)
        cur_preds = preds[cur_idxs,:,0]
        out[i,:,:] = np.moveaxis(cur_preds, 0, 1)
        
    return out

def get_updated_lstm_states(
        Y,
        n_segs,
        n_en, 
        hidden_units,
        cur_step
):
    '''
    

    Parameters
    ----------
    Y : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    h_out = np.empty((n_segs*n_en, hidden_units))
    c_out = np.empty((n_segs*n_en, hidden_units))
    
    for i in range(n_en): 
        # states should always be stored in the order of preds, h, c 
        h_idx = np.arange(n_segs, n_segs+n_segs*hidden_units) 
        h_out_idx = np.repeat(i, n_segs) + n_en * (np.arange(0,n_segs))
        h_out[h_out_idx,:] = Y[h_idx,cur_step,i].reshape((n_segs, hidden_units)) 
        
        c_idx = np.arange(2*n_segs + (n_segs * (hidden_units-1)), Y.shape[0]) 
        c_out_idx = np.repeat(i, n_segs) + n_en * (np.arange(0,n_segs))
        c_out[c_out_idx, :] = Y[c_idx,cur_step,i].reshape((n_segs, hidden_units)) 
    
    return h_out, c_out

def get_updated_rgcn_states(
        Y,
        n_segs,
        n_en, 
        hidden_layers,
        cur_step
):
    '''
    

    Parameters
    ----------
    Y : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    h_out = np.empty((n_segs*n_en, hidden_layers))
    c_out = np.empty((n_segs*n_en, hidden_layers))
    
    for i in range(n_en): 
        # states should always be stored in the order of preds, h, c 
        h_idx = np.arange(n_segs, n_segs+n_segs*hidden_layers) 
        h_out_idx = np.repeat(i, n_segs) + n_en * (np.arange(0,n_segs))
        h_out[h_out_idx,:] = Y[h_idx,cur_step,i].reshape((n_segs, hidden_layers)) 
        
        c_idx = np.arange(2*n_segs + (n_segs * (hidden_layers-1)), Y.shape[0]) 
        c_out_idx = np.repeat(i, n_segs) + n_en * (np.arange(0,n_segs))
        c_out[c_out_idx, :] = Y[c_idx,cur_step,i].reshape((n_segs, hidden_layers)) 
    
    return h_out, c_out


def get_Y_vector(
        n_states_est,
        n_step,
        n_en
):
    '''
    Vector for holding states (and parameters) for updating with EnKF 

    :param int n_states_est: number of states we're updating in data assimilation routine
    :param int n_step: number of model timesteps
    :param int n_en: number of ensembles
    '''
    Y = np.empty((n_states_est, n_step, n_en))
    Y[:] = np.NaN
    return Y 

def get_forecast_matrix(
        n_states_obs,
        n_step,
        n_en,
        f_horizon
):
    '''
    matrix for holding state estimates of the forecast  

    :param int n_states_obs: number of states we're observing (and making predictions for)
    :param int n_step: number of model timesteps
    :param int n_en: number of ensembles
    :param f_horizon: number of days of the forecast 
    '''
    Y = np.empty((n_states_obs, n_step, f_horizon, n_en))
    Y[:] = np.NaN
    return Y 


def get_obs_error_matrix(
        n_states_obs,
        n_step,
        state_sd
):
    '''
    Observation error matrix, should be a square matrix where col & row = the number of states (and params) for which you have observations; most likely we only have observations for temperature so n_states_obs will probably be the number of stream segments we're modeling 
    
    :param int n_states_obs: number of states we have observations for 
    :param int n_step: number of model timesteps
    :parma state_sd: vector of state observation standard deviation; assuming sd is constant through time
    '''
    R = np.zeros((n_states_obs, n_states_obs, n_step))
    
    state_var = state_sd**2 # variance of temperature observations 
    
    for i in range(n_step):
        # variance is the same for each depth and time step; could make dynamic or varying by time step if we have good reason to do so
        np.fill_diagonal(R[:,:,i], state_var)
    
    return R 

def get_obs_id_matrix(
        n_states_obs, 
        n_states_est,
        n_step, 
        obs_mat
):
    '''
    Measurement operator matrix saying 1 if there is observation data available, 0 otherwise
    
    :param int n_states_obs: number of states we have observations for 
    :param int n_states_est: number of states we're estimating - should include predicted states (e.g. water temp) and LSTM states (e.g. h & c) 
    :param int n_step: number of model timesteps 
    :param matrix obs_mat: observation matrix created with get_obs_matrix function     
    '''
    H = np.zeros((n_states_obs, n_states_est, n_step))
    
    for t in range(n_step):
        np.fill_diagonal(H[0:n_states_obs,0:n_states_obs, t], np.where(np.isnan(obs_mat[:,:,t]), 0, 1).reshape((n_states_obs)))
        
    return H


def get_obs_matrix(
        obs_array,
        model_locations,
        n_step,
        n_states_obs
):
    '''
    turn observation array into matrix
    
    :param obs_array: observation array 
    :param model_location: stream segments where we're predicting states 
    :param n_step: number of model timesteps 
    :param n_states_obs: number of states we have observations for     
    '''
    obs_mat = np.empty((n_states_obs, 1, n_step))
    obs_mat[:] = np.NaN 
    
    for i in range(model_locations.shape[0]):
        cur_site = obs_array[i,:,:]
        obs_mat[i,0,:] = np.reshape(cur_site, (1, n_step))
    
    return obs_mat


def get_covar_matrix(
        n_states_est,
        n_step
):
    '''
    covariance matrix for EnKF 
    
    '''
    P = np.empty((n_states_est, n_states_est, n_step))
    P[:] = np.NaN 
    
    return P 

def get_model_error_matrix(
        n_states_est,
        n_step,
        state_sd
):
    '''
    model error matrix, should be a square matrix where col & row are the number of states + params for which you are estimating 
    
    '''
    Q = np.zeros((n_states_est, n_states_est, n_step)) 
    
    for i in range(n_step):
        np.fill_diagonal(Q[:,:,i], state_sd)
        
    return Q 

def kalman_filter(
        Y,
        R,
        obs_mat,
        H,
        n_en,
        cur_step
): 
    '''
    calculating the kalman gain and updating all ensembles with Kalman gain and observation error 
    
    :param Y: vector for storing and updating states for EnKF 
    :param R: observation error matrix 
    :param obs_mat:  matrix of observations 
    :param H: measurement operator matrix saying 1 if there is an observation, 0 otherwise 
    :param n_en: number of ensemble members 
    :param cur_step: current model timestep 
    '''

    # obs_shape = obs_mat.shape 
    cur_obs = obs_mat[:,0, cur_step] # .reshape(obs_shape[0],obs_shape[1], 1) 
    cur_obs = np.where(np.isnan(cur_obs), 0, cur_obs) # setting NA's to zero so there is no 'error' when compared to estimated states

    ###### estimate the spread of your ensembles #####
    delta_Y = get_ens_deviate(Y = Y, 
                              n_en = n_en, 
                              cur_step = cur_step)  # difference in ensemble state/parameter and mean of all ensemble states/parameters
    
    ###### covariance matrix #####
    P_t = get_covar(deviations = delta_Y, n_en = n_en)

    # estimate Kalman gain #
    K = np.matmul(np.matmul(P_t, H[:,:,cur_step].T), np.linalg.inv(((1 / (n_en - 1)) * np.matmul(np.matmul(np.matmul(H[:,:,cur_step], delta_Y), delta_Y.T), H[:,:,cur_step].T) + R[:,:,cur_step])))

    # update Y vector #
    Y_shape = Y.shape
    for q in range(n_en):
        er = (cur_obs+np.random.normal(0,np.diag(R[:,:,cur_step]))) - np.matmul(H[:,:,cur_step], Y[:,cur_step,q])
        er = er.reshape((er.shape[0], 1))
        Y[:, cur_step, q] = np.add(Y[:,cur_step,q].reshape((Y_shape[0],1)), np.matmul(K, er)).reshape((Y_shape[0])) # adjusting each ensemble using kalman gain and observations

    return Y 

def get_ens_deviate(
        Y,
        n_en,
        cur_step
):
    '''
    calculate the ensemble deviations 
    
    :param Y: vector for storing and updating states for EnKF 
    :param n_en: number of ensemble members 
    :param cur_step: current model timestep 
    '''
    Y_shape = Y.shape
    cur_Y = Y[:,cur_step,:].reshape((Y_shape[0], Y_shape[2]))
    Y_mean = cur_Y.mean(axis = 1).reshape((Y_shape[0], 1)) # calculating mean of state / param estimates 
    Y_mean = np.repeat(Y_mean, n_en, axis = 1)
    
    delta_Y = np.subtract(cur_Y, Y_mean)
    
    return delta_Y


def get_covar(
        deviations,
        n_en
): 
    '''
    calculate the covariance matrix of the ensemble deviations 
    
    :param deviations: deviations from ensemble mean of each of the state / parameters estimated 
    :param n_en: number of ensemble members 
    '''
    covar = (1 / (n_en - 1)) * np.matmul(deviations, deviations.T) 
    
    return covar 


def get_error_dist(
        Y,
        H,
        R,
        P,
        n_en,
        cur_step,
        beta
):
    '''
    returns gamma from Rastetter et al Ecological Applications, 20(5), 2010, pp. 1285–1301
    
    :param Y: vector for storing and updating states for EnKF 
    :param H: measurement operator matrix 
    :param R: observation error matrix 
    :param P: covariance matrix 
    :param n_en: number of ensemble members 
    :param cur_step: current model timestep 
    :param beta: error weighting on observed variables from Rastetter et al 2010 
    '''
    ###### covariance #######
    P_t = P[:,:,cur_step]

    H_t = H[:,:,cur_step]
    
    I = np.identity(H_t.shape[1]) # identity matrix 
    # had to add R[:,:,cur_step] to make matrix solve not singular; is singular when there are no observations to assimilate 
    gamma = np.matmul(((1 - beta) * (np.linalg.inv(np.matmul(np.matmul(H_t, P_t), H_t.T) + R[:,:,cur_step]))),
                     np.matmul(np.matmul(H_t, P_t), (I - np.matmul(H_t.T, H_t)))) + beta * H_t 
    gamma = gamma.T
    
    return gamma 


def add_process_error(
        Y,
        Q,
        H,
        n_en,
        cur_step
):
    '''
    adding process error to estimated states from Q 
    
    '''
    # add error to unobserved values only (similar to inflation factor); 
    H_unobs_t = np.where(H[:,:,cur_step] == 0, 1, 0)
    H_unobs_t = H_unobs_t[0,:].reshape((Y.shape[0],1))
    H_obs_t = np.where(H[:,:,cur_step] == 1, 1, 0)
    H_obs_t = H_obs_t[0,:].reshape((Y.shape[0],1))
    mean_states = np.mean(Y[:,cur_step,:], axis=1)
    Q_diag = np.diag(Q[:,:,cur_step])
    mean_cv = np.matmul(Q_diag/mean_states, H_obs_t)
    
    cur_sd = np.abs(Y[:,cur_step,:] * mean_cv * H_unobs_t)
    to_add = np.random.normal(np.zeros((Y.shape[0],Y.shape[2])), cur_sd)
    
    for q in range(n_en):
        Y[0:Q.shape[1],cur_step,q] = Y[0:Q.shape[1],cur_step,q] + np.random.normal(np.repeat(0,Q.shape[1]), np.sqrt(np.abs(np.diag(Q[:,:,cur_step]))), Q.shape[1]) #+ to_add[0:Q.shape[1], q] 
    
    return Y 

def add_process_error_forecast(
        Y,
        Q,
        H,
        n_en,
        cur_step,
        cur_valid_t, 
        n_segs
):
    '''
    adding process error to estimated states from Q 
    
    '''
    for q in range(n_en):
        Y[0:n_segs,cur_step,cur_valid_t,q] = Y[0:n_segs,cur_step,cur_valid_t,q] + np.random.normal(np.repeat(0,n_segs), np.sqrt(np.abs(np.diag(Q[0:n_segs,0:n_segs,cur_step]))), n_segs) 
    
    return Y 

def get_innovations(
        obs_mat, 
        H,
        Y,
        R,
        cur_step,
        n_en,
        n_states_obs
):
    '''
    '''
    cur_obs = obs_mat[:,0,cur_step] 
    cur_obs = np.where(np.isnan(cur_obs), 0, cur_obs) # setting NA's to zero so there is no 'error' when compared to estimated states
    
    y_it = np.empty((n_states_obs, n_en))
    y_it[:] = np.NaN 
    for q in range(n_en):
        y_it[:,q] = cur_obs - np.matmul(H[:,:,cur_step], Y[:,cur_step,q]) + np.random.normal(np.repeat(0,n_states_obs), np.sqrt(np.diag(R[:,:,cur_step])), n_states_obs)
    
    return y_it
        
def update_model_error(
        Y,
        R,
        H,
        Q,
        Q_ave, # long-term average error 
        P,
        Pstar_t,
        S_t,
        n_en,
        cur_step,
        beta,
        alpha,
        psi
):
    
    '''
    updating model process error as in Rastetter et al. 2010 
    
    '''
    any_obs = H[:,:,cur_step] == 1 # are there any observations at this timestep? 
    if any_obs.any():
        gamma = get_error_dist(Y = Y,
                               H = H,
                               R = R,
                               P = P,
                               n_en = n_en,
                               cur_step = cur_step,
                               beta = beta)
        
        # add error to unobserved values only (similar to inflation factor); 
        #  adding a fraction based on the obs error 
        #H_unobs_t = np.where(H[:,:,cur_step] == 0, 1, 0)
        #ens_spread = get_ens_deviate(Y, n_en, cur_step)
        #ens_range = np.max(ens_spread, axis = 1) - np.min(ens_spread, axis = 1) 
        #to_add = ens_range * 1.02 * H_unobs_t
        #to_add = np.matmul(np.matmul(H[:,:,cur_step], gamma), H_unobs_t) * 0.2 
        #gamma = gamma + to_add.T 
        
        Q_hat = np.matmul(np.matmul(gamma, (S_t - np.matmul(np.matmul(H[:,:,cur_step], Pstar_t), H[:,:,cur_step].T) - R[:,:,cur_step])), gamma.T) 
        # np.fill_diagonal(Q_hat, np.diag(Q_hat) + to_add[:,0]) 
        Q_hat = alpha * Q[:,:,cur_step] + (1-alpha)*Q_hat
        
        Q[:,:,(cur_step+1)] = psi * Q_ave + (1-psi)*Q_hat
    else: 
        Q[:,:,(cur_step+1)] = Q[:,:,(cur_step)] # if no observations, previous Q == next step Q 
    
    return Q 


def get_EnKF_matrices(
        obs_array, 
        model_locations,
        n_step,
        n_states_obs,
        n_states_est,
        n_en,
        state_sd
):
    '''
    wrapper for getting EnKF matrices 
    '''
    obs_mat = get_obs_matrix(obs_array, 
                         model_locations,
                         n_step,
                         n_states_obs)

    # Y vector for storing state estimates and updates 
    Y = get_Y_vector(n_states_est, 
                     n_step, 
                     n_en)
    
    # model error matrix 
    Q = get_model_error_matrix(n_states_est,
                           n_step,
                           state_sd)

    # covariance matrix 
    P = get_covar_matrix(n_states_est, 
                         n_step)
    
    # observation error matrix 
    R = get_obs_error_matrix(n_states_obs,
                             n_step,
                             state_sd)
    
    # observation identity matrix 
    H = get_obs_id_matrix(n_states_obs,
                          n_states_est, 
                          n_step, 
                          obs_mat) 
    
    return obs_mat, Y, Q, P, R, H 

def forecast(
        Y_forecasts,
        weights_dir,
        h,
        c,
        hidden_units,
        n_en,
        model_locations,
        n_segs,
        x_pred,
        f_horizon,
        cur_step,
        Q,
        H,
        force_pos,
        include_ar1,
        obs_mean,
        obs_std,
        forecast_h_file,
        forecast_c_file,
):
    model_forecast = LSTMDA(hidden_units) # model that will make forecasts many days into future 
    model_forecast.load_weights(weights_dir)
    forecast_shape = (n_en * len(model_locations), 1, x_pred.shape[2]) 
    model_forecast.rnn_layer.build(input_shape=forecast_shape) # full timestep forecast 
    model_forecast.rnn_layer.reset_states(states=[h, c])
    for tt in range(f_horizon):
        cur_t = cur_step+tt
        forecast_drivers = x_pred[:,cur_t,:].reshape((x_pred.shape[0], 1, x_pred.shape[2]))
        if tt > 0: 
            if include_ar1: 
                scaled_forecast_water = (Y_forecasts[0:n_segs,cur_step,tt-1,:].reshape(n_en) - obs_mean) / (obs_std + 1e-10)
                forecast_drivers[:,0,1] = np.mean(scaled_forecast_water)
            forecast_h = np.load(forecast_h_file, allow_pickle=True)
            forecast_c = np.load(forecast_c_file, allow_pickle=True)
            model_forecast.rnn_layer.reset_states(states=[forecast_h, forecast_c])
        forecast_preds = model_forecast.predict(forecast_drivers, batch_size = n_en * n_segs)

        Y_forecasts[:,cur_step,tt,:] = forecast_preds[:,0,:].reshape((n_segs,n_en)) #cur_forecast
        Y_forecasts = add_process_error_forecast(Y = Y_forecasts, 
                                                 Q = Q,
                                                 H = H,
                                                 n_en = n_en,
                                                 cur_step = cur_step,
                                                 cur_valid_t = tt,
                                                 n_segs = n_segs)
        if force_pos: 
            Y_forecasts[0:n_segs,cur_step,tt,:] = np.where(Y_forecasts[0:n_segs,cur_step,tt,:]<0,0,Y_forecasts[0:n_segs,cur_step,tt,:])
            
    return Y_forecasts 
    
    
