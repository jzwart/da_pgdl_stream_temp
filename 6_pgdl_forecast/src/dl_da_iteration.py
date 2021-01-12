# iteration between training parameters with tensorflow and updating states with DA 

import numpy as np
import tensorflow as tf
import sys
sys.path.insert(1, '5_pgdl_pretrain/src')
from LSTMDA import *
from prep_synthetic_data import *
from train_lstm import * 
sys.path.insert(1, 'utils/src')
from EnKF_functions import * 

synthetic_obs_error = 1  # standard deviation in observation error 
seg_ids = [1573] # needs to be a list of seg_ids (even if one segment)
n_en = 100 # number of ensemble members 
cycles = 5 # number of DA-DL cycles make this dynamic based on a stopping criteria 
n_epochs = 20 # number of epochs during DA-DL cycling 
obs_freq = 7 # observation frequency in days 

# 1) create true state from SNTemp (already did this step, housed in zarr file)
# 2) sample from true state to create observations - using function below 
prep_synthetic_data(
    obs_temp_file = "5_pgdl_pretrain/in/obs_temp_full",
    synthetic_file = "5_pgdl_pretrain/in/uncal_sntemp_input_output",
    synthetic_obs_error = synthetic_obs_error,
    obs_freq = obs_freq, # frequency of samples in days 
    seg_id = seg_ids,
    start_date_trn = "2011-06-01",
    end_date_trn = "2012-06-01",
    start_date_pred = "2012-06-02",
    end_date_pred = "2013-06-02",
    out_file="5_pgdl_pretrain/in/da_dl_iter.npz",
    n_en = n_en
)
true_trn, true_pred = prep_one_var_lstm_da(
    var_file = "5_pgdl_pretrain/in/uncal_sntemp_input_output",
    vars = ["seg_tave_water"],
    seg_id = seg_ids,
    start_date_trn = "2011-06-01",
    end_date_trn = "2012-06-01",
    start_date_pred = "2012-06-02",
    end_date_pred = "2013-06-02",
    scale_data=False,
)
true_trn = true_trn.to_array().values 
true_pred = true_pred.to_array().values 

# load in the data 
data = np.load('5_pgdl_pretrain/in/da_dl_iter.npz')


train_dir = '5_pgdl_pretrain/out'
process_error = True # T/F add process error during DA step 
beta = 0.5 # weighting for how much uncertainty should go to observed vs. 
               # unobserved states (lower beta attributes most of the 
               # uncertainty for unobserved states, higher beta attributes
               # most uncertainty to observed states)
alpha = 0.8  # weight for how quickly the process error is allowed to 
               # adapt (low alpha quickly changes process error 
               # based on current innovations)

# get model prediction parameters for setting up EnKF matrices 
obs_array = data['y_trn']
n_states_obs, n_step, tmp = obs_array.shape
model_locations = data['model_locations'] # seg_id_nat of the stream segments 
n_segs = len(model_locations)
n_states_obs = n_segs
state_sd = np.repeat(synthetic_obs_error, n_states_obs) # uncertainty around observations 
dates = data['dates_trn']

# initial training of LSTM on synthetic observations; 
n_epochs_init = 50 
train_save_model(input_data = '5_pgdl_pretrain/in/da_dl_iter.npz',
                 out_model_file = '5_pgdl_pretrain/out/da_dl_iter_trained_wgts/',
                 out_h_file = '5_pgdl_pretrain/out/h.npy', 
                 out_c_file = '5_pgdl_pretrain/out/c.npy',
                 n_epochs = n_epochs_init)


n_states_est = 3 * len(model_locations) # number of states we're estimating (predictions, h, c) for x segments

# set up EnKF matrices 
obs_mat, Y, Q, P, R, H = get_EnKF_matrices(obs_array = obs_array, 
                                           model_locations = model_locations,
                                           n_step = n_step,
                                           n_states_obs = n_states_obs, 
                                           n_states_est = n_states_est,
                                           n_en = n_en,
                                           state_sd = state_sd)


for i in range(0, cycles):
    model_da = LSTMDA(1) # model that will make predictions only one day into future 
    model_da.load_weights(train_dir + '/lstm_da_trained_wgts/')
    da_drivers = data['x_trn'][:,0,:].reshape((data['x_trn'].shape[0],1,data['x_trn'].shape[2])) # only single timestep for DA model
    
    #da_shape = (n_en * len(model_locations), da_drivers.shape[1], da_drivers.shape[2])
    model_da.rnn_layer.build(input_shape=da_drivers.shape)
    
    # initialize the states with the previously trained states 
    # load LSTM states from trained model 
    h = np.load(train_dir + '/h.npy', allow_pickle=True)
    c = np.load(train_dir + '/c.npy', allow_pickle=True)
    
    model_da.rnn_layer.reset_states(states=[h, c])
    # make predictions and store states for updating with EnKF 
    da_preds = model_da.predict(da_drivers, batch_size = n_en * n_segs) 
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
    if process_error: 
            # uncorrupted ensemble deviations before adding process error 
            dstar_t = get_ens_deviate(Y = Y,
                                      n_en = n_en,
                                      cur_step = 0)
            # uncorrupted covariance before adding process error 
            Pstar_t = get_covar(deviations = dstar_t, 
                                n_en = n_en) 
            
            # add process error 
            Y = add_process_error(Y = Y, 
                                  Q = Q,
                                  H = H,
                                  n_en = n_en,
                                  cur_step = 0)
            # ensemble deviations with process error 
            d_t = get_ens_deviate(Y = Y,
                                  n_en = n_en, 
                                  cur_step = 0)
            # covariance with process error 
            P[:,:,0] = get_covar(deviations = d_t,
                                 n_en = n_en)
            
            y_it = get_innovations(obs_mat = obs_mat,
                                   H = H,
                                   Y = Y,
                                   R = R, 
                                   cur_step = 0,
                                   n_en = n_en,
                                   n_states_obs = n_states_obs)
            S_t = get_covar(deviations = y_it,
                            n_en = n_en)
            # update model process error matrix 
            Q = update_model_error(Y = Y,
                                   R = R,
                                   H = H,
                                   Q = Q, 
                                   P = P, 
                                   Pstar_t = Pstar_t,
                                   S_t = S_t,
                                   n_en = n_en,
                                   cur_step = 0,
                                   beta = beta,
                                   alpha = alpha)
    
    # loop through forecast time steps and make forecasts & update with EnKF 
    for t in range(1, n_step):
        print(dates[t])
        
        # update lstm with h & c states stored in Y from previous timestep 
        new_h, new_c = get_updated_lstm_states(
                Y = Y,
                n_segs = n_segs,
                n_en = n_en,
                cur_step = t-1)
        model_da.rnn_layer.reset_states(states=[new_h, new_c]) 
    
        # make predictions  and store states for updating with EnKF 
        cur_drivers = data['x_trn'][:,t,:].reshape((data['x_trn'].shape[0],1,data['x_trn'].shape[2]))
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
    
    # retrain LSTM with DA states - weight by uncertainty 
    n_batch, seq_len, n_feat = data['x_trn'].shape
    new_y = Y[0:n_segs,:,:] # DA states with which to train weights     
    P_diag = np.empty((n_states_obs, n_step)) # get diagonal of P for weighting in new training of LSTM 
    for step in range(0, n_step):
        cur_Y = Y[0:n_segs:,step,:].reshape((new_y.shape[0], new_y.shape[2]))
        Y_mean = cur_Y.mean(axis = 1).reshape((new_y.shape[0], 1)) # calculating mean of state / param estimates 
        Y_mean = np.repeat(Y_mean, n_en, axis = 1)
        new_y[:,step,:] = cur_Y
        P_diag[:,step] = np.diag(P[0:n_segs,0:n_segs,step]) 
    new_y = np.moveaxis(new_y, 2, 0)
    new_y = np.moveaxis(new_y, 1, 2)
    P_diag_inv = 1/P_diag  # inverse of P are the weights 
    P_diag_inv = np.repeat(P_diag_inv, n_en, axis = 0).reshape((new_y.shape))
    # need to concatonate weights onto y_true when using weighted rmse
    new_y_cat = np.concatenate([new_y, P_diag_inv], axis = 2)

    cur_lstm = LSTMDA(1)
    cur_lstm.rnn_layer.build(input_shape=data['x_trn'].shape)
    cur_lstm.compile(loss=rmse_weighted, optimizer=tf.optimizers.Adam(learning_rate=0.3))
    cur_lstm.load_weights(train_dir + '/lstm_da_trained_wgts/')

    cur_lstm.fit(x=data['x_trn'], y=new_y_cat, epochs=n_epochs, batch_size=n_batch)
    
    cur_lstm.save_weights('5_pgdl_pretrain/out/lstm_da_trained_wgts/')
    h, c = cur_lstm.rnn_layer.states
    np.save('5_pgdl_pretrain/out/h.npy', h.numpy())
    np.save('5_pgdl_pretrain/out/c.npy', c.numpy())

out = {
    "Y": Y,
    "obs": obs_mat,
    "true": true_trn,
    "R": R,
    "Q": Q,
    "P": P,
    "dates": dates,
    "model_locations": model_locations,
}

np.savez('5_pgdl_pretrain/out/simple_lstm_dl_da_iter.npz', **out)


