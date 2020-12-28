import numpy as np
import tensorflow as tf
tf.enable_eager_execution() # Jake has to upgrade to tensorflow 2.x 
import sys
sys.path.insert(1, '5_pgdl_pretrain/src')
from LSTMDA import LSTMDA
sys.path.insert(1, 'utils/src')
from EnKF_functions import * 

train_dir = '5_pgdl_pretrain/out'

# load in the data 
data = np.load('5_pgdl_pretrain/in/lstm_da_data_just_air_temp.npz')

# get model prediction parameters for setting up EnKF matrices 
obs_array = data['y_pred']
n_states_obs, n_step, tmp = obs_array.shape
model_locations = ('1') # model index of the stream segments 
n_en = 1 # number of ensembles 
state_sd = np.repeat(0.5, n_states_obs) # uncertainty around observations 

# load LSTM states from trained model 
h = np.load(train_dir + '/h.npy', allow_pickle=True)
c = np.load(train_dir + '/c.npy', allow_pickle=True)

states = combine_lstm_states(h, h, c)
n_states_est = states.shape[0] # number of states we're estimating (predictions + LSTM states) 

# set up EnKF matrices 
obs_mat = get_obs_matrix(obs_array, 
                         model_locations,
                         n_step,
                         n_states_obs)

# Y vector for storing state estimates and updates 
Y = get_Y_vector(n_states_est, 
                 n_step, 
                 n_en)

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
model_forecast = LSTMDA(1) # model that will make forecasts many days into future 
model_da = LSTMDA(1) # model that will make predictions only one day into future 
model_forecast.load_weights(train_dir + '/lstm_da_trained_wgts/')
model_da.load_weights(train_dir + '/lstm_da_trained_wgts/')
model_forecast.rnn_layer.build(input_shape=data['x_pred'].shape) # full timestep forecast 
da_drivers = data['x_pred'][:,0,:].reshape((data['x_pred'].shape[0],1,data['x_pred'].shape[2])) # only single timestep for DA model
model_da.rnn_layer.build(input_shape=da_drivers.shape)

# initialize the states with the previously trained states 
model_forecast.rnn_layer.reset_states(states=[h, c])
model_da.rnn_layer.reset_states(states=[h, c])
#make forecasts 
forecast_preds = model_forecast.predict(data['x_pred'], batch_size=1)
print(forecast_preds)
# make predictions and store states for updating with EnKF 
da_preds = model_da.predict(da_drivers, batch_size = 1)
cur_h, cur_c = model_da.rnn_layer.states 
print(da_preds)

cur_states = combine_lstm_states(da_preds[:,0,:], cur_h.numpy(), cur_c.numpy())
Y[:,0,:] = cur_states # storing in Y for EnKF updating 

# loop through forecast time steps and make forecasts & update with EnKF 
for t in range(1, n_step):
    # update lstm with h & c states stored in Y from previous timestep 
    model_da.rnn_layer.reset_states(states=[np.array([Y[1,t-1,:]]), np.array([Y[2,t-1,:]])]) 

    # make predictions  and store states for updating with EnKF 
    cur_drivers = data['x_pred'][:,t,:].reshape((data['x_pred'].shape[0],1,data['x_pred'].shape[2]))
    cur_preds = model_da.predict(cur_drivers, batch_size = 1)
    cur_h, cur_c = model_da.rnn_layer.states 
    
    cur_states = combine_lstm_states(cur_preds[:,0,:], cur_h.numpy(), cur_c.numpy())
    Y[:,t,:] = cur_states # storing in Y for EnKF updating 

    any_obs = H[:,:,t] == 1 # are there any observations at this timestep? 
    if any_obs.any(): 
        print('updating with Kalman filter...') 
        #Y = kalman_filter(Y, R, obs_mat, H, n_en, t)

# adjust the states
# model.rnn_layer.reset_states(states=[np.array([[-10]]), np.array([[-100]])])
# p = model.predict(data['x_pred'], batch_size=1)
# print(p)
