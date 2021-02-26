import numpy as np
import tensorflow as tf
import os 
import random 
import sys
sys.path.insert(1, '5_pgdl_pretrain/src')
from LSTMDA import *
from RGCNDA import RGCN
from prep_da_lstm_data import * 
from train_lstm import train_model
sys.path.insert(1, 'utils/src')
from EnKF_functions import * 
sys.path.insert(1, '6_pgdl_forecast/src')
from dl_da_iteration_ar1 import dl_da_iter


model_type = 'lstm' # options are rgcn, lstm 
train_dir = '5_pgdl_pretrain/out'
pre_train = True # T/F if to pre-train with SNTemp output 
fine_tune = True # T/F if to do fine-tune training on temeprature observations 
fine_tune_iter = False 
process_error = True # T/F add process error during DA step 
store_raw_states = True # T/F store the LSTM states without data assimilation 
store_forecasts = True # T/F store predictions that are in the future 
force_pos = True # T/F force estimates to be positive 
update_h_c = True # T/F update h and c states with DA 
ar1_temp = False # T/F include yesterday's water temp as driver 
ar1_up_temp = False # T/F include yesterday's upstream temperature as a driver 
mc_dropout = False # T/F to include monte carlo dropout estimates 
mc_dropout_rate = 0.5 # rate for monte carlo dropout 
f_horizon = 8 # forecast horizon in days (how many days into the future to make predictions)
beta = 0.5 # weighting for how much uncertainty should go to observed vs. 
               # unobserved states (lower beta attributes most of the 
               # uncertainty for unobserved states, higher beta attributes
               # most uncertainty to observed states)
alpha = 0.9  # weight for how quickly the process error is allowed to 
               # adapt (low alpha quickly changes process error 
               # based on current innovations)
psi = 0.95 # weighting for how much uncertainty goes to long-term average vs. 
            # dynamic uncertainty (higher psi places higher weight on long-term average uncertainty)
temp_obs_sd = .5 # standard deviation of temperature observations 
h_sd = 0.02
c_sd = 0.06
doy_feat = False # T/F to add day of year 
ave_preds = True # T/F to make batches all averages of prms-sntemp preds 

seed = 134
os.environ["PYTHONHASHSEED"] = str(seed)
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)

seg_ids = [1573] # needs to be a list of seg_ids (even if one segment)
# less-impacted segments - 2046, 2037
# lordville segment - 1573 
n_segs = len(seg_ids)

n_en = 50
learn_rate_pre = 0.05
learn_rate_fine = 0.05
n_epochs_pre = 30# number of epochs for pretraining 
n_epochs_fine = 250 # number of epochs for finetuning 
hidden_units = 6 # number of hidden units 
cycles = 10 # number of cycles for DA-DL routine 
weights_dir = '5_pgdl_pretrain/out/segid%s_%sar1_lstm_da_trained_wgts/' % (seg_ids, ar1_temp)
out_h_file = '5_pgdl_pretrain/out/segid%s_%sar1_h.npy' % (seg_ids, ar1_temp)
out_c_file = '5_pgdl_pretrain/out/segid%s_%sar1_c.npy' % (seg_ids, ar1_temp) 
da_h_file = '5_pgdl_pretrain/out/h_da.npy'
da_c_file = '5_pgdl_pretrain/out/c_da.npy'
raw_h_file = '5_pgdl_pretrain/out/h_raw.npy'
raw_c_file = '5_pgdl_pretrain/out/c_raw.npy'
forecast_h_file = '5_pgdl_pretrain/out/h_forecast.npy'
forecast_c_file = '5_pgdl_pretrain/out/c_forecast.npy'
data_file = "5_pgdl_pretrain/in/lstm_da_data.npz"
obs_temp_file = "5_pgdl_pretrain/in/obs_temp_full"
driver_file = "5_pgdl_pretrain/in/uncal_sntemp_input_output"
start_date_trn = "1985-05-01"
end_date_trn = "2014-06-01"
start_date_pred = "2014-06-02"
end_date_pred = "2015-06-02"
dist_mat_file = "1_model_fabric/in/distance_matrix.npz"
dist_mat_direction = 'downstream' # which direction to go for distance matrix 


x_vars = ["seg_tave_air", "seginc_swrad", "seg_rain", "seg_humid", "seg_slope","seg_length","seg_elev"]
y_vars=["seg_tave_water"]
obs_vars = ["temp_c"]

if ar1_temp:
    x_vars.append("seg_tave_water")
    ar1_temp_pos = x_vars.index('seg_tave_water') # position of the feature 
else: 
    ar1_temp_pos = False 
    
'''
prep the data 
'''

prep_data_lstm_da(
    obs_temp_file = obs_temp_file,
    driver_file = driver_file,
    dist_mat_file = dist_mat_file, 
    dist_mat_direction = dist_mat_direction, 
    seg_id = seg_ids,
    start_date_trn = start_date_trn,
    end_date_trn = end_date_trn,
    start_date_pred = start_date_pred,
    end_date_pred = end_date_pred,
    x_vars=x_vars,
    y_vars=y_vars,
    obs_vars = obs_vars,
    out_file=data_file,
    n_en = n_en,
    ar1_temp = ar1_temp
)

# load in the data 
x_trn, y_trn, obs_trn, obs_trn_ar1, x_pred, x_pred_da, x_pred_f, obs_array, model_locations, dist_mat, dates, dates_trn, seg_tave_water_mean, seg_tave_water_std, obs_mean, obs_std =get_data_lstm_da(data_file, 
                 ar1_temp, 
                 ar1_temp_pos,
                 doy_feat,
                 n_en)


'''
Train the model 
'''
train_model(model_type,
            x_trn, 
            y_trn,
            obs_trn,
            obs_trn_ar1, 
            hidden_units, 
            learn_rate_pre,
            learn_rate_fine, 
            n_epochs_pre, 
            n_epochs_fine,
            mc_dropout,
            mc_dropout_rate,
            weights_dir,
            out_h_file, 
            out_c_file,
            pre_train,
            fine_tune_iter,
            fine_tune,
            cycles,
            temp_obs_sd,
            process_error,
            beta,
            alpha,
            psi,
            seg_tave_water_mean, 
            seg_tave_water_std,
            obs_mean,
            obs_std,
            force_pos,
            n_en,
            ar1_temp,
            ar1_temp_pos,
            n_segs,
            model_locations,
            dist_mat,
            dates_trn,
            update_h_c,
            h_sd,
            c_sd)


'''
Set up data assimilation matrices 
'''

n_states_obs, n_step, state_sd, n_states_est, obs_mat, Y, Q, P, R, H, Q_ave, Y_no_da, Y_forecasts, forecast_model_list, forecast_pred_array = get_da_objects(model_type,
                mc_dropout,
                mc_dropout_rate,
                obs_array,
                x_pred_f,
                temp_obs_sd,
                h_sd,
                c_sd,
                update_h_c,
                hidden_units,
                model_locations,
                dist_mat,
                n_en,
                n_segs,
                store_raw_states,
                store_forecasts,
                f_horizon,
                weights_dir)

 
'''
create prediction models and load the trained weights; resest h&c states to trained model 
h & c states from last time step 
'''

Y, Y_no_da, Y_forecasts, obs_mat, R, Q, P = predict_and_forecast(model_type, 
                                                                  mc_dropout, 
                                                                  mc_dropout_rate,
                                                                  out_h_file,
                    out_c_file,
                    da_h_file,
                    da_c_file,
                    raw_h_file,
                    raw_c_file,
                    forecast_h_file,
                    forecast_c_file,
                    hidden_units,
                    weights_dir,
                    x_pred,
                    x_pred_da,
                    x_pred_f,
                    n_en,
                    n_segs,
                    n_step,
                    dates,
                    model_locations,
                    dist_mat,
                    update_h_c,
                    n_states_est,
                    n_states_obs,
                    process_error,
                    beta,
                    alpha,
                    psi,
                    Y,
                    Q,
                    Q_ave,
                    H,
                    R,
                    P,
                    obs_mat,
                    force_pos,
                    store_raw_states,
                    Y_no_da,
                    Y_forecasts,
                    store_forecasts,
                    forecast_model_list,
                    f_horizon,
                    ar1_temp,
                    ar1_temp_pos,
                    obs_mean,
                    obs_std,
                    forecast_pred_array)



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
    #"obs_orig": obs_mat_orig,
    #"obs_trn": obs_trn,
    #"trn_preds": trn_preds,
    #"trn_dates": data['dates_trn'][1:data['dates_trn'].shape[0]],
    }
elif not store_forecasts & store_raw_states: 
    out = {
    "Y": Y,
    "Y_no_da": Y_no_da,
    "obs": obs_mat,
    "R": R,
    "Q": Q,
    "P": P,
    "dates": dates,
    "model_locations": model_locations,
    #"obs_orig": obs_mat_orig,
    #"obs_trn": obs_trn,
    #"trn_preds": trn_preds,
    #"trn_dates": data['dates_trn'][1:data['dates_trn'].shape[0]],
    }
elif not store_raw_states & store_forecasts: 
    out = {
    "Y": Y,
    #"Y_no_da": Y_no_da,
    "Y_forecasts": Y_forecasts,
    "obs": obs_mat,
    "R": R,
    "Q": Q,
    "P": P,
    "dates": dates,
    "model_locations": model_locations,
    #"obs_orig": obs_mat_orig,
    #"obs_trn": obs_trn,
    #"trn_preds": trn_preds,
    #"trn_dates": data['dates_trn'][1:data['dates_trn'].shape[0]],
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
    #"obs_orig": obs_mat_orig,
    }

out_file = '5_pgdl_pretrain/out/%s_da_segid%s_%sepoch_%sbeta_%salpha_%shc_%sAR1_%sHiddenUnits_%sMCdropout.npz' % (model_type, seg_ids, n_epochs_fine, beta, alpha, update_h_c, ar1_temp, hidden_units, mc_dropout) 
np.savez(out_file, **out)

