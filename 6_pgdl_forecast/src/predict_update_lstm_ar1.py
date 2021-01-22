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
fine_tune = True # T/F if to do fine-tune training on temeprature observations 
process_error = True # T/F add process error during DA step 
store_raw_states = True # T/F store the LSTM states without data assimilation 
store_forecasts = True # T/F store predictions that are in the future 
force_pos = True # T/F force estimates to be positive 
f_horizon = 8 # forecast horizon in days (how many days into the future to make predictions)
beta = 0.5 # weighting for how much uncertainty should go to observed vs. 
               # unobserved states (lower beta attributes most of the 
               # uncertainty for unobserved states, higher beta attributes
               # most uncertainty to observed states)
alpha = 0.9  # weight for how quickly the process error is allowed to 
               # adapt (low alpha quickly changes process error 
               # based on current innovations)
psi = 0.6 # weighting for how much uncertainty goes to long-term average vs. 
            # dynamic uncertainty (higher psi places higher weight on long-term average uncertainty)
temp_obs_sd = .8 # standard deviation of temperature observations 
doy_feat = True # T/F to add day of year 

n_en = 50 
n_epochs_pre = 100# number of epochs for pretraining 
n_epochs_fine = 100 # number of epochs for finetuning 
cycles = 1
weights_dir = '5_pgdl_pretrain/out/lstm_da_trained_wgts/'
out_h_file = '5_pgdl_pretrain/out/h.npy' 
out_c_file = '5_pgdl_pretrain/out/c.npy' 

seg_ids = [1573] # needs to be a list of seg_ids (even if one segment)

prep_data_lstm_da(
    obs_temp_file = "5_pgdl_pretrain/in/obs_temp_full",
    driver_file = "4_pb_model/out/pb_pretrain_model_output.nc",
    seg_id = seg_ids,
    start_date_trn = "2000-10-01",
    end_date_trn = "2012-06-01",
    start_date_pred = "2012-06-02",
    end_date_pred = "2013-06-02",
    x_vars=["seg_tave_air", "seg_tave_water"],
    y_vars=["seg_tave_water"],
    obs_vars = ["temp_c"],
    out_file="5_pgdl_pretrain/in/lstm_da_data.npz",
    n_en = n_en
)

# load in the data 
data = np.load('5_pgdl_pretrain/in/lstm_da_data.npz')
seg_tave_water_mean = data['seg_tave_water_mean']
seg_tave_water_std = data['seg_tave_water_std'] 

# add in yesterday's water temperature as a driver (AR1)
temp_minus1 = data['x_trn'][:,0:(data['x_trn'].shape[1]-1),1]
x_trn = data['x_trn'][:,1:data['x_trn'].shape[1],:]
#x_trn = np.append(x_trn, temp_minus1, axis = 2) # add temp_minus1 as a feature 
x_trn[:,:,1] = temp_minus1
y_trn = data['y_trn'][:,1:data['y_trn'].shape[1],:] 
# taking average of prms-sntemp preds to see if training is better 
y_trn = np.repeat(np.mean(y_trn, axis = 0), n_en, axis=1)
y_trn = np.moveaxis(y_trn, 0, -1)
y_trn = y_trn.reshape((y_trn.shape[0],y_trn.shape[1],1))
obs_trn = data['obs_trn'][:,1:data['obs_trn'].shape[1],:] 
if doy_feat:
    doy_trn = data['doy_trn'][0:(data['doy_trn'].shape[0]-1)]
    doy_pred = data['doy_pred'][0:(data['doy_pred'].shape[0]-1)]
    doy_trn = np.tile(doy_trn, n_en).reshape((n_en, doy_trn.shape[0],1))
    doy_pred = np.tile(doy_pred, n_en).reshape((n_en, doy_pred.shape[0],1))
    x_trn = np.append(x_trn, doy_trn, axis = 2)


if pre_train:
    n_batch, seq_len, n_feat = x_trn.shape
    pretrain_model = LSTMDA(1)
    pretrain_model(x_trn)

    pretrain_model.compile(loss=rmse_masked, optimizer=tf.optimizers.Adam(learning_rate=0.3))
    pretrain_model.fit(x=x_trn, y=y_trn, epochs=n_epochs_pre, batch_size=n_batch)

    pretrain_model.save_weights(weights_dir)
    h, c = pretrain_model.rnn_layer.states
    np.save(out_h_file, h.numpy())
    np.save(out_c_file, c.numpy())
    
    if fine_tune:
        dl_da_iter(cycles = cycles,
                   weights_dir = weights_dir,
                   out_h_file = out_h_file, 
                   out_c_file = out_c_file,
                   x_trn = x_trn, 
                   obs_trn = obs_trn, 
                   data = data, 
                   temp_obs_sd = temp_obs_sd,
                   n_en =n_en, 
                   process_error = process_error, 
                   n_epochs_fine = n_epochs_fine)
    
    #fine_tune_model = LSTMDA(1) 
    #fine_tune_model.load_weights(weights_dir) 
    #fine_tune_model.rnn_layer.build(input_shape=data['x_trn'].shape)
    
    #fine_tune_model.compile(loss=rmse_masked, optimizer=tf.optimizers.Adam(learning_rate=0.3))
    #fine_tune_model.fit(x=data['x_trn'], y=data['obs_trn'], epochs=n_epochs_fine, batch_size=n_batch)
    
    #fine_tune_model.save_weights(weights_dir)

    #h, c = fine_tune_model.rnn_layer.states
    #np.save(out_h_file, h.numpy())
    #np.save(out_c_file, c.numpy())
#else:
 #   fine_tune_model = LSTMDA(1) 
  #  fine_tune_model(data['x_trn']) 
    
  #  fine_tune_model.compile(loss=rmse_masked, optimizer=tf.optimizers.Adam(learning_rate=0.3))
  #  fine_tune_model.fit(x=data['x_trn'], y=data['obs_trn'], epochs=n_epochs_fine, batch_size=n_batch)
    
  #  fine_tune_model.save_weights(weights_dir)

  #  h, c = fine_tune_model.rnn_layer.states
  #  np.save(out_h_file, h.numpy())
  #  np.save(out_c_file, c.numpy())


# get model prediction parameters for setting up EnKF matrices 
obs_array = data['obs_pred'][:,1:data['obs_pred'].shape[1],:] 
temp_minus1 = data['x_pred'][:,0:(data['x_pred'].shape[1]-1),1] # this will be updated with DA 
x_pred = data['x_pred'][:,1:data['x_pred'].shape[1],:]
#x_pred = np.append(x_pred, temp_minus1, axis =2)
x_pred[:,:,1] = temp_minus1
if doy_feat:
    x_pred = np.append(x_pred, doy_pred, axis = 2)

n_states_obs, n_step, tmp = obs_array.shape
model_locations = data['model_locations'] # seg_id_nat of the stream segments 
n_segs = len(model_locations)
state_sd = np.repeat(temp_obs_sd, n_states_obs) # uncertainty around observations 
dates = data['dates_pred'][1:data['dates_pred'].shape[0]]

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

# get long term error 
Q_ave = get_model_error_matrix(n_states_est, 1, state_sd)
Q_ave[1,1,0] = 0.02
Q_ave[2,2,0] = 0.06
Q[:,:,0] = Q_ave[:,:,0]
Q[:,:,1] = Q_ave[:,:,0]

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
    forecast_shape = (n_en * len(model_locations), 1, x_pred.shape[2]) 
    model_forecast.rnn_layer.build(input_shape=forecast_shape) # full timestep forecast 
    model_forecast.rnn_layer.reset_states(states=[h, c])
    for tt in range(f_horizon):
        cur_t = 0+tt
        forecast_drivers = x_pred[:,cur_t,:].reshape((x_pred.shape[0], 1, x_pred.shape[2]))
        forecast_preds = model_forecast.predict(forecast_drivers, batch_size = n_en * n_segs)
        #cur_forecast = get_forecast_preds(preds = forecast_preds,
         #                                 n_segs = n_segs,
          #                                n_states_obs = n_states_obs, 
           #                               n_en = n_en,
            #                              f_horizon = f_horizon)
        Y_forecasts[:,0,tt,:] = forecast_preds[:,0,:].reshape((n_segs,n_en)) #cur_forecast
        Y_forecasts = add_process_error_forecast(Y = Y_forecasts, 
                                                 Q = Q,
                                                 H = H,
                                                 n_en = n_en,
                                                 cur_step = 0,
                                                 cur_valid_t = cur_t,
                                                 n_segs = n_segs)
        
model_da = LSTMDA(1) # model that will make predictions only one day into future 
model_da.load_weights(weights_dir)
da_drivers = x_pred[:,0,:].reshape((x_pred.shape[0],1,x_pred.shape[2])) # only single timestep for DA model
da_shape = (n_en * len(model_locations), da_drivers.shape[1], da_drivers.shape[2])
model_da.rnn_layer.build(input_shape=da_shape)

# initialize the states with the previously trained states 
model_da.rnn_layer.reset_states(states=[h, c])
# make predictions and store states for updating with EnKF 
da_preds = model_da.predict(da_drivers, batch_size = n_en * n_segs) # make this dynamic batch size based on n_en
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
    
Y = add_process_error(Y = Y, 
                      Q = Q,
                      H = H,
                      n_en = n_en,
                      cur_step = 0)
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
        cur_drivers = x_pred[:,t,:].reshape((x_pred.shape[0],1,x_pred.shape[2]))
        cur_preds = model_da.predict(cur_drivers, batch_size = n_en * n_segs)
    
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
    
    # update yesterday's temperature driver from Y; need to scale first though 
    scaled_seg_tave_water = (Y[0:n_segs,t-1,:].reshape(n_en) - seg_tave_water_mean) / (seg_tave_water_std + 1e-10)
    x_pred[:,t,1] = scaled_seg_tave_water
    
    # update lstm with h & c states stored in Y from previous timestep 
    new_h, new_c = get_updated_lstm_states(
            Y = Y,
            n_segs = n_segs,
            n_en = n_en,
            cur_step = t-1)
    model_da.rnn_layer.reset_states(states=[new_h, new_c]) 
    if store_forecasts:
        model_forecast.rnn_layer.reset_states(states=[new_h, new_c])
        for tt in range(f_horizon):
            cur_t = t + tt
            forecast_drivers = x_pred[:,cur_t,:].reshape((x_pred.shape[0], 1, x_pred.shape[2]))
            if tt > 0: 
                scaled_seg_tave_water = (Y_forecasts[0:n_segs,t,tt-1,:].reshape(n_en) - seg_tave_water_mean) / (seg_tave_water_std + 1e-10)
                forecast_drivers[:,0,1] = scaled_seg_tave_water
            forecast_preds = model_forecast.predict(forecast_drivers, batch_size = n_en * n_segs)
            #cur_forecast = get_forecast_preds(preds = forecast_preds,
             #                                 n_segs = n_segs,
              #                                n_states_obs = n_states_obs, 
               #                               n_en = n_en,
                #                              f_horizon = f_horizon)
            Y_forecasts[:,t,tt,:] = forecast_preds[:,0,:].reshape((n_segs,n_en)) #cur_forecast
            Y_forecasts = add_process_error_forecast(Y = Y_forecasts, 
                                                 Q = Q,
                                                 H = H,
                                                 n_en = n_en,
                                                 cur_step = t,
                                                 cur_valid_t = tt,
                                                 n_segs = n_segs)
            if force_pos: 
                Y_forecasts[0:n_segs,t,tt,:] = np.where(Y_forecasts[0:n_segs,t,tt,:]<0,0,Y_forecasts[0:n_segs,t,tt,:])
    

    # make predictions  and store states for updating with EnKF 
    cur_drivers = x_pred[:,t,:].reshape((x_pred.shape[0],1,x_pred.shape[2]))
    cur_preds = model_da.predict(cur_drivers, batch_size = n_en * n_segs)

    cur_h, cur_c = model_da.rnn_layer.states 
    
    cur_states = combine_lstm_states(
                cur_preds[:,0,:],
                cur_h.numpy(), 
                cur_c.numpy(),
                n_segs,
                n_states_est,
                n_en)
    Y[:,t,:] = cur_states # storing in Y for EnKF updating 
    if force_pos: 
        Y[0:n_segs,t,:] = np.where(Y[0:n_segs,t,:]<0,0,Y[0:n_segs,t,:])
    
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
        if force_pos: 
            Y[0:n_segs,t,:] = np.where(Y[0:n_segs,t,:]<0,0,Y[0:n_segs,t,:])
    
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
                                   Q_ave = Q_ave[:,:,0],
                                   P = P, 
                                   Pstar_t = Pstar_t,
                                   S_t = S_t,
                                   n_en = n_en,
                                   cur_step = t,
                                   beta = beta,
                                   alpha = alpha,
                                   psi = psi)

    any_obs = H[:,:,t] == 1 # are there any observations at this timestep? 
    if any_obs.any(): 
        print('updating with Kalman filter...') 
        Y = kalman_filter(Y, R, obs_mat, H, n_en, t)
        if force_pos: 
            Y[0:n_segs,t,:] = np.where(Y[0:n_segs,t,:]<0,0,Y[0:n_segs,t,:])
    
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

