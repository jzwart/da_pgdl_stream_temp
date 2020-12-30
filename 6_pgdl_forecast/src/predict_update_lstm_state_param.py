import numpy as np
import tensorflow as tf
import sys
sys.path.insert(1, '5_pgdl_pretrain/src')
from LSTMDA import LSTMDA
sys.path.insert(1, 'utils/src')
from EnKF_functions import * 

train_dir = '5_pgdl_pretrain/out'
process_error = True # T/F add process error during DA step 
store_raw_states = True # T/F store the LSTM states without data assimilation 
beta = 0.5 # weighting for how much uncertainty should go to observed vs. 
               # unobserved states (lower beta attributes most of the 
               # uncertainty for unobserved states, higher beta attributes
               # most uncertainty to observed states)
alpha = 0.8  # weight for how quickly the process error is allowed to 
               # adapt (low alpha quickly changes process error 
               # based on current innovations)

# load in the data 
data = np.load('5_pgdl_pretrain/in/lstm_da_data_just_air_temp.npz')

# get model prediction parameters for setting up EnKF matrices 
obs_array = data['y_pred']
n_states_obs, n_step, tmp = obs_array.shape
model_locations = np.array(['1573']) # model index of the stream segments 
n_en = 100 # number of ensembles 
state_sd = np.repeat(1, n_states_obs) # uncertainty around observations 
dates = data['dates_pred']

# load LSTM states from trained model 
h = np.load(train_dir + '/h.npy', allow_pickle=True)
c = np.load(train_dir + '/c.npy', allow_pickle=True)

n_states_est = 3 # number of states we're estimating (predictions, h, c) 

# set up EnKF matrices 
obs_mat = get_obs_matrix(obs_array, 
                         model_locations,
                         n_step,
                         n_states_obs)

# Y vector for storing state estimates and updates 
Y = get_Y_vector(n_states_est, 
                 n_step, 
                 n_en)
if store_raw_states: 
    Y_no_da = get_Y_vector(n_states_est, 
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

# define LSTM model using previously trained model; use one model for making forecasts many days into the future and one for updating states (will only make predictions 1 timestep at a time) 
#model_forecast = LSTMDA(1) # model that will make forecasts many days into future 
model_da = LSTMDA(1) # model that will make predictions only one day into future 
#model_forecast.load_weights(train_dir + '/lstm_da_trained_wgts/')
model_da.load_weights(train_dir + '/lstm_da_trained_wgts/')
#forecast_shape = (n_en, data['x_pred'].shape[1], 1) 
#model_forecast.rnn_layer.build(input_shape=forecast_shape) # full timestep forecast 
da_drivers = data['x_pred'][:,0,:].reshape((data['x_pred'].shape[0],1,data['x_pred'].shape[2])) # only single timestep for DA model
da_shape = (n_en, 1, 1)
model_da.rnn_layer.build(input_shape=da_shape)

# initialize the states with the previously trained states 
#model_forecast.rnn_layer.reset_states(states=[h, c])
model_da.rnn_layer.reset_states(states=[h, c])
#make forecasts 
#forecast_preds = model_forecast.predict(data['x_pred'], batch_size=n_en)
#print(forecast_preds)
# make predictions and store states for updating with EnKF 
da_preds = model_da.predict(da_drivers, batch_size = n_en) # make this dynamic batch size based on n_en
cur_h, cur_c = model_da.rnn_layer.states 
#print(da_preds)

cur_states = combine_lstm_states(
        preds = da_preds[:,0,:], 
        h = cur_h.numpy(), 
        c = cur_c.numpy())
Y[:,0,:] = cur_states # storing in Y for EnKF updating 
if store_raw_states: 
    Y_no_da[:,0,:] = cur_states 
cur_deviations = get_ens_deviate(
        Y = Y, 
        n_en = n_en,
        cur_step = 0)
P[:,:,0] = get_covar(deviations = cur_deviations, n_en = n_en)

# testing out resetting weights 
cur_drivers = data['x_pred'][:,1,:].reshape((data['x_pred'].shape[0],1,data['x_pred'].shape[2]))
cur_preds = model_da.predict(cur_drivers, batch_size = n_en)
print(cur_preds)    
cur_weights = model_da.get_weights()
cur_weights[0][0,0] = 1.5
cur_weights[0][0,3] = 3
model_da.set_weights(cur_weights)
adj_preds = model_da.predict(cur_drivers, batch_size = n_en)
print(adj_preds, cur_preds)

if store_raw_states: 
    for t in range(1, n_step):
        print(dates[t])
        # update lstm with h & c states stored in Y from previous timestep 
        model_da.rnn_layer.reset_states(states=[np.array([Y_no_da[1,t-1,:]]).T, np.array([Y_no_da[2,t-1,:]]).T]) 
    
        # make predictions  and store states 
        cur_drivers = data['x_pred'][:,t,:].reshape((data['x_pred'].shape[0],1,data['x_pred'].shape[2]))
        cur_preds = model_da.predict(cur_drivers, batch_size = n_en)
    
        cur_h, cur_c = model_da.rnn_layer.states 
        
        cur_states = combine_lstm_states(
                cur_preds[:,0,:],
                cur_h.numpy(), 
                cur_c.numpy())
        Y_no_da[:,t,:] = cur_states # storing in Y for EnKF updating 


# loop through forecast time steps and make forecasts & update with EnKF 
for t in range(1, n_step):
    print(dates[t])
    # update lstm with h & c states stored in Y from previous timestep 
    model_da.rnn_layer.reset_states(states=[np.array([Y[1,t-1,:]]).T, np.array([Y[2,t-1,:]]).T]) 

    # make predictions  and store states for updating with EnKF 
    cur_drivers = data['x_pred'][:,t,:].reshape((data['x_pred'].shape[0],1,data['x_pred'].shape[2]))
    cur_preds = model_da.predict(cur_drivers, batch_size = n_en)

    cur_h, cur_c = model_da.rnn_layer.states 
    
    cur_states = combine_lstm_states(
            cur_preds[:,0,:],
            cur_h.numpy(), 
            cur_c.numpy())
    Y[:,t,:] = cur_states # storing in Y for EnKF updating 
    
    if process_error: 
        # uncorrupted ensemble deviations before adding process error 
        dstar_t = get_ens_deviate(Y = Y,
                                  n_en = n_en,
                                  cur_step = t)
        # uncorrupted covariance before adding process error 
        Pstar_t = get_covar(deviations = dstar_t, 
                            n_en = n_en) 
        
        # add process error 
        Y = add_process_error(Y = Y, 
                              Q = Q,
                              n_en = n_en,
                              cur_step = t)
        # ensemble deviations with process error 
        d_t = get_ens_deviate(Y = Y,
                              n_en = n_en, 
                              cur_step = t)
        # covariance with process error 
        P[:,:,t] = get_covar(deviations = d_t,
                             n_en = n_en)
        
        y_it = get_innovations(obs_mat = obs_mat,
                               H = H,
                               Y = Y,
                               R = R, 
                               cur_step = t,
                               n_en = n_en,
                               n_states_obs = n_states_obs)
        S_t = get_covar(deviations = y_it,
                        n_en = n_en)
        # update model process error matrix 
        if t < (n_step-1):
            Q = update_model_error(Y = Y,
                                   R = R,
                                   H = H,
                                   Q = Q, 
                                   P = P, 
                                   Pstar_t = Pstar_t,
                                   S_t = S_t,
                                   n_en = n_en,
                                   cur_step = t,
                                   beta = beta,
                                   alpha = alpha)

    any_obs = H[:,:,t] == 1 # are there any observations at this timestep? 
    if any_obs.any(): 
        #if((t < 100) or (t > 140)): 
        print('updating with Kalman filter...') 
        Y = kalman_filter(Y, R, obs_mat, H, n_en, t)

if store_raw_states: 
    out = {
    "Y": Y,
    "Y_no_da": Y_no_da,
    "obs": obs_mat,
    "R": R,
    "Q": Q,
    "P": P,
    "dates": dates,
    "model_locations": model_locations,
    #"preds_no_da": forecast_preds,
    }
else: 
    out = {
    "Y": Y,
    "obs": obs_mat,
    "R": R,
    "Q": Q,
    "P": P,
    "dates": dates,
    "model_locations": model_locations,
    #"preds_no_da": forecast_preds,
    }

np.savez('5_pgdl_pretrain/out/simple_lstm_da.npz', **out)

#print(forecast_preds[0,:,:], Y[0,:,:])
#import matplotlib.pyplot as plt
#plt.plot(Y[0,:,:], forecast_preds[:,:,0].T, 'ro')

#plt.plot(Y[0,:,:], 'ro', Y[1,:,:], 'bo', Y[2,:,:], 'go')
#plt.plot(Y[0,:,:], Y[1,:,:], 'ro')
#plt.plot(Y[0,:,:], Y[2,:,:], 'ro')
#plt.plot(Y[0,:,:], obs_mat[0,:,:].T, 'o')
#plt.plot(Y[0,:,:],'r', obs_mat[0,:,:].T, 'b')
