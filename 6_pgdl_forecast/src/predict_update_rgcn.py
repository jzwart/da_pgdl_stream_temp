import numpy as np
import tensorflow as tf
import sys
sys.path.insert(1, '5_pgdl_pretrain/src')
from RGCNDA import *
from LSTMDA import * 
sys.path.insert(1, 'utils/src')
from EnKF_functions import * 
from prep_da_rgcn_data import * 

train_dir = '5_pgdl_pretrain/out'
process_error = True # T/F add process error during DA step 
store_raw_states = False # T/F store the LSTM states without data assimilation 
store_forecasts = False # T/F store predictions that are in the future 
f_horizon = 3 # forecast horizon in days (how many days into the future to make predictions)
beta = 0.5 # weighting for how much uncertainty should go to observed vs. 
               # unobserved states (lower beta attributes most of the 
               # uncertainty for unobserved states, higher beta attributes
               # most uncertainty to observed states)
alpha = 0.8  # weight for how quickly the process error is allowed to 
               # adapt (low alpha quickly changes process error 
               # based on current innovations)
seg_ids = [1573, 1577] # needs to be a list of seg_ids (even if one segment); 
seg_ids.sort() # these should be sorted numerically!! 
n_epochs = 100 
n_en = 30 # number of ensembles 
hidden_layers = 2

prep_data_rgcn_da(
    obs_temp_file = "5_pgdl_pretrain/in/obs_temp_full",
    driver_file = "5_pgdl_pretrain/in/uncal_sntemp_input_output",
    dist_mat_file = "1_model_fabric/in/distance_matrix.npz", 
    dist_mat_direction = 'downstream', # which direction to go for distance matrix 
    seg_id = seg_ids,
    start_date_trn = "2010-06-01",
    end_date_trn = "2012-06-01",
    start_date_pred = "2012-06-02",
    end_date_pred = "2013-06-02",
    out_file="5_pgdl_pretrain/in/rgcn_da_data_just_air_temp.npz",
    n_en = n_en
)
# load in the data 
data = np.load('5_pgdl_pretrain/in/rgcn_da_data_just_air_temp.npz')

# get model prediction parameters for setting up EnKF matrices 
obs_array = data['y_pred']
n_states_obs, n_step, tmp = obs_array.shape
model_locations = data['model_locations'] # seg_id_nat of the stream segments 
n_segs = len(model_locations)
state_sd = np.repeat(1, n_states_obs) # uncertainty around observations 
dates = data['dates_pred']
dist_mat = data['distance_matrix']

# initial training of RGCN 
n_batch, seq_len, n_feat = data['x_trn'].shape
rgcn = RGCN(hidden_layers, dist_mat)
rgcn.rnn_layer.build(input_shape=data['x_trn'].shape)
rgcn.compile(loss=rmse_masked, optimizer=tf.optimizers.Adam(learning_rate=0.3), run_eagerly=True) # need the run eagerly to get out the states (might make this run slower so we should think about not doing this and instead trianing & then predicting on entire train input data to get out states from last time step)
rgcn.fit(x=data['x_trn'], y=data['y_trn'], epochs=n_epochs, batch_size=n_batch)
rgcn.save_weights('5_pgdl_pretrain/out/rgcn_da_trained_wgts/')
h = rgcn.h_gr # h corresponding to the predictions
c = rgcn.c_gr # c corresponding to the predictions

np.save('5_pgdl_pretrain/out/h_gr.npy', h.numpy())
np.save('5_pgdl_pretrain/out/c_gr.npy', c.numpy())

# load LSTM states from trained model 
h = np.load('5_pgdl_pretrain/out/h_gr.npy', allow_pickle=True)
c = np.load('5_pgdl_pretrain/out/c_gr.npy', allow_pickle=True)
# only need last timestep 
h = h[:,-1,:]
c = c[:,-1,:]

n_states_est = 1 * len(model_locations) + (hidden_layers * 2) * len(model_locations) # number of states we're estimating (predictions, h, c) for x segments

# withholding some observations for testing assimilation effectiveness 
obs_mat_orig = get_obs_matrix(obs_array, model_locations, n_step, n_states_obs)
#obs_array[0,19:34,0] = np.nan
#obs_array[0,:,:] = np.nan

# set up EnKF matrices 
obs_mat, Y, Q, P, R, H = get_EnKF_matrices(obs_array = obs_array, 
                                           model_locations = model_locations,
                                           n_step = n_step,
                                           n_states_obs = n_states_obs, 
                                           n_states_est = n_states_est,
                                           n_en = n_en,
                                           state_sd = state_sd)

if store_raw_states: 
    Y_no_da = get_Y_vector(n_states_est, 
                           n_step, 
                           n_en)
    
if store_forecasts: 
    Y_forecasts = get_forecast_matrix(n_states_obs, 
                                      n_step, 
                                      n_en, 
                                      f_horizon)
    
# define LSTM model using previously trained model; use one model for making forecasts many days into the future and one for updating states (will only make predictions 1 timestep at a time) 
if store_forecasts:
    model_forecast = LSTMDA(1) # model that will make forecasts many days into future 
    model_forecast.load_weights(train_dir + '/lstm_da_trained_wgts/')
    forecast_shape = (n_en * len(model_locations), f_horizon, data['x_pred'].shape[2]) 
    model_forecast.rnn_layer.build(input_shape=forecast_shape) # full timestep forecast 
    model_forecast.rnn_layer.reset_states(states=[h, c])
    forecast_drivers = data['x_pred'][:,0:f_horizon,:].reshape((data['x_pred'].shape[0], f_horizon, data['x_pred'].shape[2]))
    forecast_preds = model_forecast.predict(np.repeat(forecast_drivers, n_en, axis = 0), batch_size = n_en * n_segs)
    cur_forecast = get_forecast_preds(preds = forecast_preds,
                                      n_segs = n_segs,
                                      n_states_obs = n_states_obs, 
                                      n_en = n_en,
                                      f_horizon = f_horizon)
    Y_forecasts[:,0,:,:] = cur_forecast

rgcn_da = RGCN(hidden_layers, dist_mat) # model that will make predictions only one day into future 
rgcn_da.load_weights('5_pgdl_pretrain/out/rgcn_da_trained_wgts/')
da_drivers = data['x_pred'][:,0,:].reshape((data['x_pred'].shape[0],1,data['x_pred'].shape[2])) # only single timestep for DA model
da_shape = (n_en * len(model_locations), da_drivers.shape[1], da_drivers.shape[2])
rgcn_da.rnn_layer.build(input_shape=da_shape)

# initialize the states with the previously trained states 
#rgcn_da.rnn_layer.reset_states(states=[h, c])
# make predictions and store states for updating with EnKF 
#da_preds = rgcn_da.predict(np.repeat(da_drivers, n_en, axis = 0), batch_size = n_en * n_segs) # make this dynamic batch size based on n_en
da_preds = rgcn_da(np.repeat(da_drivers, n_en, axis = 0), h_init = h, c_init = c)
da_preds = da_preds.numpy() 
cur_h = rgcn_da.h_gr
cur_c = rgcn_da.c_gr
#print(da_preds)

cur_states = combine_rgcn_states(
        preds = da_preds[:,0,:], 
        h = cur_h.numpy(), 
        c = cur_c.numpy(),
        n_segs = n_segs,
        n_states_est = n_states_est,
        n_en = n_en,
        hidden_layers = hidden_layers)
Y[:,0,:] = cur_states # storing in Y for EnKF updating 
if store_raw_states: 
    Y_no_da[:,0,:] = cur_states 
cur_deviations = get_ens_deviate(
        Y = Y, 
        n_en = n_en,
        cur_step = 0)
P[:,:,0] = get_covar(deviations = cur_deviations, n_en = n_en)


if store_raw_states: 
    for t in range(1, n_step):
        print(dates[t])
        # update lstm with h & c states stored in Y from previous timestep 
        new_h, new_c = get_updated_lstm_states(
            Y = Y_no_da,
            n_segs = n_segs,
            n_en = n_en,
            cur_step = t-1)
        model_da.rnn_layer.reset_states(states=[new_h, new_c]) 
    
        # make predictions  and store states 
        cur_drivers = data['x_pred'][:,t,:].reshape((data['x_pred'].shape[0],1,data['x_pred'].shape[2]))
        cur_preds = model_da.predict(np.repeat(cur_drivers,n_en, axis =0), batch_size = n_en * n_segs)
    
        cur_h, cur_c = model_da.rnn_layer.states 
        
        cur_states = combine_lstm_states(
                cur_preds[:,0,:],
                cur_h.numpy(), 
                cur_c.numpy(),
                n_segs,
                n_states_est,
                n_en)
        Y_no_da[:,t,:] = cur_states # storing in Y for EnKF updating 


# loop through forecast time steps and make forecasts & update with EnKF 
if store_forecasts:
    n_step = n_step - f_horizon
for t in range(1, n_step):
    print(dates[t])
    
    # testing state adjustment by adjusting h & c by a lot 
    #if t == 30: 
    #    Y[1,t-1,:] = np.random.normal(20, 2, n_en)
    #    Y[2,t-1,:] = np.random.normal(40, 2, n_en)
    
    # update lstm with h & c states stored in Y from previous timestep 
    
    new_h, new_c = get_updated_rgcn_states(
            Y = Y,
            n_segs = n_segs,
            n_en = n_en,
            hidden_layers = hidden_layers, 
            cur_step = t-1)
    if store_forecasts:
        model_forecast.rnn_layer.reset_states(states=[new_h, new_c])
        start = t
        stop = start+f_horizon
        forecast_drivers = data['x_pred'][:,start:stop,:].reshape((data['x_pred'].shape[0], f_horizon, data['x_pred'].shape[2]))
        forecast_preds = model_forecast.predict(np.repeat(forecast_drivers, n_en, axis = 0), batch_size = n_en * n_segs)
        cur_forecast = get_forecast_preds(preds = forecast_preds,
                                          n_segs = n_segs,
                                          n_states_obs = n_states_obs, 
                                          n_en = n_en,
                                          f_horizon = f_horizon)
        Y_forecasts[:,t,:,:] = cur_forecast

    # make predictions  and store states for updating with EnKF 
    cur_drivers = data['x_pred'][:,t,:].reshape((data['x_pred'].shape[0],1,data['x_pred'].shape[2]))
    cur_preds = rgcn_da(np.repeat(cur_drivers, n_en, axis = 0), h_init = h, c_init = c)
    cur_preds = cur_preds.numpy() 
    cur_h = rgcn_da.h_gr
    cur_c = rgcn_da.c_gr
    
    cur_states = combine_rgcn_states(
                cur_preds[:,0,:],
                cur_h.numpy(), 
                cur_c.numpy(),
                n_segs,
                n_states_est,
                n_en,
                hidden_layers = hidden_layers)
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
                              H = H,
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
        print('updating with Kalman filter...') 
        Y = kalman_filter(Y, R, obs_mat, H, n_en, t)

if store_forecasts & store_raw_states:
    out = {
    "Y": Y,
    "Y_no_da": Y_no_da,
    "Y_forecasts": Y_forecasts,
    "obs": obs_mat,
    "R": R,
    "Q": Q,
    "P": P,
    "dates": dates,
    "model_locations": model_locations,
    "obs_orig": obs_mat_orig,
    }
#if store_raw_states: 
#    out = {
#    "Y": Y,
#    "Y_no_da": Y_no_da,
#    "obs": obs_mat,
#    "R": R,
#    "Q": Q,
#    "P": P,
#    "dates": dates,
#    "model_locations": model_locations,
#    "obs_orig": obs_mat_orig,
#    }
else: 
    out = {
    "Y": Y,
    "obs": obs_mat,
    "R": R,
    "Q": Q,
    "P": P,
    "dates": dates,
    "model_locations": model_locations,
    "obs_orig": obs_mat_orig,
    }

np.savez('5_pgdl_pretrain/out/simple_rgcn_da_50epoch.npz', **out)


