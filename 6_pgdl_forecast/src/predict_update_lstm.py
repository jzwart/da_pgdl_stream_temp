import numpy as np
import tensorflow as tf
import sys
sys.path.insert(1, '5_pgdl_pretrain/src')
from LSTMDA import *
from prep_da_lstm_data import * 
sys.path.insert(1, 'utils/src')
from EnKF_functions import * 

train_dir = '5_pgdl_pretrain/out'
pre_train = True # T/F if to pre-train with SNTemp output 
process_error = True # T/F add process error during DA step 
store_raw_states = True # T/F store the LSTM states without data assimilation 
store_forecasts = True # T/F store predictions that are in the future 
f_horizon = 3 # forecast horizon in days (how many days into the future to make predictions)
beta = 0.5 # weighting for how much uncertainty should go to observed vs. 
               # unobserved states (lower beta attributes most of the 
               # uncertainty for unobserved states, higher beta attributes
               # most uncertainty to observed states)
alpha = 0.8  # weight for how quickly the process error is allowed to 
               # adapt (low alpha quickly changes process error 
               # based on current innovations)
n_en = 100 
n_epochs_pre = 100 # number of epochs for pretraining 
n_epochs_fine = 100 # number of epochs for finetuning 
weights_dir = '5_pgdl_pretrain/out/lstm_da_trained_wgts/'
out_h_file = '5_pgdl_pretrain/out/h.npy' 
out_c_file = '5_pgdl_pretrain/out/c.npy' 

seg_ids = [1573] # needs to be a list of seg_ids (even if one segment)

prep_data_lstm_da(
    obs_temp_file = "5_pgdl_pretrain/in/obs_temp_full",
    driver_file = "5_pgdl_pretrain/in/uncal_sntemp_input_output",
    seg_id = seg_ids,
    start_date_trn = "2000-06-01",
    end_date_trn = "2012-06-01",
    start_date_pred = "2012-06-02",
    end_date_pred = "2013-06-02",
    x_vars=["seg_tave_air"],
    y_vars=["seg_tave_water"],
    obs_vars = ["temp_c"],
    out_file="5_pgdl_pretrain/in/lstm_da_data_just_air_temp.npz",
    n_en = n_en
)

# load in the data 
data = np.load('5_pgdl_pretrain/in/lstm_da_data_just_air_temp.npz')

if pre_train:
    n_batch, seq_len, n_feat = data['x_trn'].shape
    pretrain_model = LSTMDA(1)
    pretrain_model(data['x_trn'])

    pretrain_model.compile(loss=rmse_masked, optimizer=tf.optimizers.Adam(learning_rate=0.3))
    pretrain_model.fit(x=data['x_trn'], y=data['y_trn'], epochs=n_epochs_pre, batch_size=n_batch)

    pretrain_model.save_weights(weights_dir)
    
    fine_tune_model = LSTMDA(1) 
    fine_tune_model.load_weights(weights_dir) 
    fine_tune_model.rnn_layer.build(input_shape=data['x_trn'].shape)
    
    fine_tune_model.compile(loss=rmse_masked, optimizer=tf.optimizers.Adam(learning_rate=0.3))
    fine_tune_model.fit(x=data['x_trn'], y=data['obs_trn'], epochs=n_epochs_fine, batch_size=n_batch)
    
    fine_tune_model.save_weights(weights_dir)

    h, c = fine_tune_model.rnn_layer.states
    np.save(out_h_file, h.numpy())
    np.save(out_c_file, c.numpy())
else:
    fine_tune_model = LSTMDA(1) 
    fine_tune_model(data['x_trn']) 
    
    fine_tune_model.compile(loss=rmse_masked, optimizer=tf.optimizers.Adam(learning_rate=0.3))
    fine_tune_model.fit(x=data['x_trn'], y=data['obs_trn'], epochs=n_epochs_fine, batch_size=n_batch)
    
    fine_tune_model.save_weights(weights_dir)

    h, c = fine_tune_model.rnn_layer.states
    np.save(out_h_file, h.numpy())
    np.save(out_c_file, c.numpy())


# get model prediction parameters for setting up EnKF matrices 
obs_array = data['obs_pred']
n_states_obs, n_step, tmp = obs_array.shape
model_locations = data['model_locations'] # seg_id_nat of the stream segments 
n_segs = len(model_locations)
state_sd = np.repeat(1, n_states_obs) # uncertainty around observations 
dates = data['dates_pred']

# load LSTM states from trained model 
h = np.load(out_h_file, allow_pickle=True)
c = np.load(out_c_file, allow_pickle=True)

n_states_est = 3 * len(model_locations) # number of states we're estimating (predictions, h, c) for x segments

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
    model_forecast.load_weights(weights_dir)
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
    
model_da = LSTMDA(1) # model that will make predictions only one day into future 
model_da.load_weights(weights_dir)
da_drivers = data['x_pred'][:,0,:].reshape((data['x_pred'].shape[0],1,data['x_pred'].shape[2])) # only single timestep for DA model
da_shape = (n_en * len(model_locations), da_drivers.shape[1], da_drivers.shape[2])
model_da.rnn_layer.build(input_shape=da_shape)

# initialize the states with the previously trained states 
model_da.rnn_layer.reset_states(states=[h, c])
# make predictions and store states for updating with EnKF 
da_preds = model_da.predict(np.repeat(da_drivers, n_en, axis = 0), batch_size = n_en * n_segs) # make this dynamic batch size based on n_en
cur_h, cur_c = model_da.rnn_layer.states 
#print(da_preds)

cur_states = combine_lstm_states(
        preds = da_preds[:,0,:], 
        h = cur_h.numpy(), 
        c = cur_c.numpy(),
        n_segs = n_segs,
        n_states_est = n_states_est,
        n_en = n_en)
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
    new_h, new_c = get_updated_lstm_states(
            Y = Y,
            n_segs = n_segs,
            n_en = n_en,
            cur_step = t-1)
    model_da.rnn_layer.reset_states(states=[new_h, new_c]) 
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
    cur_preds = model_da.predict(np.repeat(cur_drivers,n_en, axis =0), batch_size = n_en * n_segs)

    cur_h, cur_c = model_da.rnn_layer.states 
    
    cur_states = combine_lstm_states(
                cur_preds[:,0,:],
                cur_h.numpy(), 
                cur_c.numpy(),
                n_segs,
                n_states_est,
                n_en)
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

out_file = '5_pgdl_pretrain/out/simple_lstm_da_%sepoch_%sbeta_%salpha.npz' % (n_epochs_fine, beta, alpha)
np.savez(out_file, **out)


