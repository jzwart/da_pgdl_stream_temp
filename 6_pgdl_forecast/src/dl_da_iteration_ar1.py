# iteration between training parameters with tensorflow and updating states with DA 
import numpy as np
import tensorflow as tf
import sys
sys.path.insert(1, '5_pgdl_pretrain/src')
from LSTMDA import *
from prep_da_lstm_data import * 
sys.path.insert(1, 'utils/src')
from EnKF_functions import * 

def dl_da_iter(
        cycles,
        weights_dir,
        out_h_file,
        out_c_file, 
        x_trn,
        obs_trn,
        model_locations,
        temp_obs_sd,
        n_en,
        process_error,
        n_epochs_fine,
        learn_rate_fine,
        beta,
        alpha,
        psi,
        seg_tave_water_mean,
        seg_tave_water_std,
        force_pos,
        dates):
    
    obs_array = obs_trn
    n_states_obs, n_step, tmp = obs_array.shape
    n_segs = len(model_locations)
    n_states_obs = n_segs
    state_sd = np.repeat(temp_obs_sd, n_states_obs) 

    n_states_est = 3 * len(model_locations) # number of states we're estimating (predictions, h, c) for x segments

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

    for i in range(0, cycles):
        model_da = LSTMDA(1) # model that will make predictions only one day into future 
        model_da.load_weights(weights_dir)
        da_drivers = x_trn[:,0,:].reshape((x_trn.shape[0],1,x_trn.shape[2])) # only single timestep for DA model
        
        #da_shape = (n_en * len(model_locations), da_drivers.shape[1], da_drivers.shape[2])
        model_da.rnn_layer.build(input_shape=da_drivers.shape)
        
        # initialize the states with the previously trained states 
        # load LSTM states from trained model 
        h = np.load(out_h_file, allow_pickle=True)
        c = np.load(out_c_file, allow_pickle=True)
        
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
                                       Q_ave = Q_ave[:,:,0],
                                       P = P, 
                                       Pstar_t = Pstar_t,
                                       S_t = S_t,
                                       n_en = n_en,
                                       cur_step = 0,
                                       beta = beta,
                                       alpha = alpha,
                                       psi = psi)
        
        # loop through forecast time steps and make forecasts & update with EnKF 
        for t in range(1, n_step):
            print(dates[t])
            # update yesterday's temperature driver from Y; need to scale first though 
            scaled_seg_tave_water = (Y[0:n_segs,t-1,:].reshape(n_en) - seg_tave_water_mean) / (seg_tave_water_std + 1e-10)
            x_trn[:,t,1] = scaled_seg_tave_water
    
            # update lstm with h & c states stored in Y from previous timestep 
            new_h, new_c = get_updated_lstm_states(
                    Y = Y,
                    n_segs = n_segs,
                    n_en = n_en,
                    cur_step = t-1)
            model_da.rnn_layer.reset_states(states=[new_h, new_c]) 
        
            # make predictions  and store states for updating with EnKF 
            cur_drivers = x_trn[:,t,:].reshape((x_trn.shape[0],1,x_trn.shape[2]))
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
                                       cur_step = 0,
                                       beta = beta,
                                       alpha = alpha,
                                       psi = psi)
        
            any_obs = H[:,:,t] == 1 # are there any observations at this timestep? 
            if any_obs.any(): 
                print('updating with Kalman filter...') 
                Y = kalman_filter(Y, R, obs_mat, H, n_en, t)
                if force_pos: 
                    Y[0:n_segs,t,:] = np.where(Y[0:n_segs,t,:]<0,0,Y[0:n_segs,t,:])
    
        
        # retrain LSTM with DA states - weight by uncertainty 
        n_batch, seq_len, n_feat = x_trn.shape
        new_y = Y[0:n_segs,:,:] # DA states with which to train weights     
        P_diag = np.empty((n_states_obs, n_step)) # get diagonal of P for weighting in new training of LSTM 
        for step in range(0, n_step):
            cur_Y = Y[0:n_segs:,step,:].reshape((new_y.shape[0], new_y.shape[2]))
            Y_mean = cur_Y.mean(axis = 1).reshape((new_y.shape[0], 1)) # calculating mean of state / param estimates 
            Y_mean = np.repeat(Y_mean, n_en, axis = 1)
            temp_minus1_mean = np.repeat(np.mean(x_trn[:,step,1],axis=0),n_en)
            x_trn[:,step,1] = temp_minus1_mean
            new_y[:,step,:] = Y_mean
            P_diag[:,step] = np.diag(P[0:n_segs,0:n_segs,step]) 
        new_y = np.moveaxis(new_y, 2, 0)
        new_y = np.moveaxis(new_y, 1, 2)
        P_diag_inv = 1/P_diag  # inverse of P are the weights 
        ########### multiply by H to only weight observed periods ############
        new_H = H[0:n_segs,0:n_segs,:].reshape((n_segs,n_step))
        P_diag_inv = P_diag_inv * new_H
        P_diag_inv = np.repeat(P_diag_inv, n_en, axis = 0).reshape((new_y.shape))
        # need to concatonate weights onto y_true when using weighted rmse
        new_y_cat = np.concatenate([new_y, P_diag_inv], axis = 2)
    
        cur_lstm = LSTMDA(1)
        cur_lstm.rnn_layer.build(input_shape=x_trn.shape)
        cur_lstm.compile(loss=rmse_weighted, optimizer=tf.optimizers.Adam(learning_rate=0.3))
        cur_lstm.load_weights(weights_dir)
    
        cur_lstm.fit(x=x_trn, y=new_y_cat, epochs=n_epochs_fine, batch_size=n_batch)
        
        cur_lstm.save_weights(weights_dir)
        h, c = cur_lstm.rnn_layer.states
        np.save(out_h_file, h.numpy())
        np.save(out_c_file, c.numpy())


