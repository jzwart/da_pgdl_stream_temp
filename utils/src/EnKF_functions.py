# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 10:30:42 2020

@author: jzwart
functions for updating LSTM states using ensemble Kalman filter 
"""
import numpy as np 

def combine_lstm_states(preds, h, c):
    '''
    Combining lstm states for updating in EnKF. Should always be in the order of preds, h, c. 
    
    '''
    out = np.concatenate((preds, h, c), )
    return out


def get_Y_vector(n_states_est, n_step, n_en):
    '''
    Vector for holding states (and parameters) for updating with EnKF 

    :param int n_states_est: number of states we're updating in data assimilation routine
    :param int n_step: number of model timesteps
    :param int n_en: number of ensembles
    '''
    Y = np.empty((n_states_est, n_step, n_en))
    Y[:] = np.NaN
    return Y 

def get_obs_error_matrix(n_states_obs, n_step, state_sd):
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

def get_obs_id_matrix(n_states_obs, n_states_est, n_step, obs_mat):
    '''
    Measurement operator matrix saying 1 if there is observation data available, 0 otherwise
    
    :param int n_states_obs: number of states we have observations for 
    :param int n_states_est: number of states we're estimating - should include predicted states (e.g. water temp) and LSTM states (e.g. h & c) 
    :param int n_step: number of model timesteps 
    :param matrix obs_mat: observation matrix created with get_obs_matrix function     
    '''
    H = np.zeros((n_states_obs, n_states_est, n_step))
    
    for t in range(n_step):
        H[0:n_states_obs, 0:n_states_obs, t] = np.where(np.isnan(obs_mat[:,:,t]), 0, 1) # this needs to be a diagonal if predicting for more than 1 stream segment 
        
    return H


def get_obs_matrix(obs_array, model_locations, n_step, n_states_obs):
    '''
    turn observation array into matrix
    
    :param obs_array: observation array 
    :param model_location: stream segments where we're predicting states 
    :param n_step: number of model timesteps 
    :param n_states_obs: number of states we have observations for     
    '''
    obs_mat = np.empty((n_states_obs, 1, n_step))
    obs_mat[:] = np.NaN 
    
    for i in range(len(model_locations)):
        cur_site = obs_array[i,:,:]
        obs_mat[i,0,:] = np.reshape(cur_site, (1, n_step))
    
    return obs_mat

#  NEED TO MAKE THIS INTO PYTHON CODE ## 
def kalman_filter(Y,
                  R,
                  obs_mat,
                  H,
                  n_en,
                  cur_step): 
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
    cur_obs = obs_mat[:,:, cur_step] # .reshape(obs_shape[0],obs_shape[1], 1) 
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
        er = cur_obs - np.matmul(H[:,:,cur_step], Y[:,cur_step,q])
        Y[:, cur_step, q] = np.add(Y[:,cur_step,q].reshape((Y_shape[0],1)), np.matmul(K, er)).reshape((Y_shape[0])) # adjusting each ensemble using kalman gain and observations

    return Y 

def get_ens_deviate(Y, n_en, cur_step):
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


def get_covar(deviations, n_en): 
    '''
    calculate the covariance matrix of the ensemble deviations 
    
    :param deviations: deviations from ensemble mean of each of the state / parameters estimated 
    :param n_en: number of ensemble members 
    '''
    covar = (1 / (n_en - 1)) * np.matmul(deviations, deviations.T) 
    
    return covar 







