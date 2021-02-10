import tensorflow as tf
import numpy as np
import sys
sys.path.insert(1, '5_pgdl_pretrain/src')
from LSTMDA import LSTMDA, rmse_masked
from RGCNDA import RGCN
sys.path.insert(1, '6_pgdl_forecast/src')
from dl_da_iteration_ar1 import dl_da_iter

# def train_save_model(
#         input_data,
#         out_model_file, 
#         out_h_file,
#         out_c_file,
#         n_epochs
# ):
    
#     data = np.load(input_data)
#     n_batch, seq_len, n_feat = data['x_trn'].shape
#     mymodel = LSTMDA(1)
#     mymodel(data['x_trn'])

#     mymodel.compile(loss=rmse_masked,
#                     optimizer=tf.optimizers.Adam(learning_rate=0.3))
#     mymodel.fit(x=data['x_trn'], y=data['y_trn'], epochs=n_epochs, batch_size=n_batch)
#     #for i in range(n_epochs):
#     #    mymodel.fit(x=data['x_trn'], y=data['y_trn'], epochs=1, batch_size=n_batch)
#     #    mymodel.reset_states()
#     mymodel.save_weights(out_model_file)
#     h, c = mymodel.rnn_layer.states
#     np.save(out_h_file, h.numpy())
#     np.save(out_c_file, c.numpy())

def train_model(model_type,
                x_trn, 
                y_trn,
                obs_trn,
                obs_trn_ar1, 
                hidden_units, 
                learn_rate_pre,
                learn_rate_fine, 
                n_epochs_pre, 
                n_epochs_fine,
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
                c_sd
):

    if pre_train:
        if model_type == 'lstm': 
            n_batch, seq_len, n_feat = x_trn.shape
            pretrain_model = LSTMDA(hidden_units)
            pretrain_model(x_trn)
        
            pretrain_model.compile(loss=rmse_masked, optimizer=tf.keras.optimizers.Adam(learning_rate=tf.Variable(learn_rate_pre)))
            pretrain_model.fit(x=x_trn, y=y_trn, epochs=n_epochs_pre, batch_size=n_batch)
        
            pretrain_model.save_weights(weights_dir)
            h, c = pretrain_model.rnn_layer.states
            np.save(out_h_file, h.numpy())
            np.save(out_c_file, c.numpy())
        elif model_type == 'rgcn':
            n_batch, seq_len, n_feat = x_trn.shape
            pretrain_model = RGCN(hidden_units, dist_mat)
            pretrain_model.rnn_layer.build(input_shape=x_trn.shape)
            pretrain_model.compile(loss=rmse_masked, optimizer=tf.keras.optimizers.Adam(learning_rate=tf.Variable(learn_rate_pre)), run_eagerly=True) # need the run eagerly to get out the states (might make this run slower so we should think about not doing this and instead trianing & then predicting on entire train input data to get out states from last time step)
            pretrain_model.fit(x=x_trn, y=y_trn, epochs=n_epochs_pre, batch_size=n_batch)
            
            pretrain_model.save_weights(weights_dir)
            h = pretrain_model.h_gr # h corresponding to the predictions
            c = pretrain_model.c_gr # c corresponding to the predictions
            
            np.save(out_h_file, h.numpy())
            np.save(out_c_file, c.numpy())
        
    if fine_tune_iter:
        dl_da_iter(cycles = cycles,
                   weights_dir = weights_dir,
                   out_h_file = out_h_file, 
                   out_c_file = out_c_file,
                   x_trn = x_trn, 
                   obs_trn = obs_trn, 
                   model_locations = model_locations,
                   temp_obs_sd = temp_obs_sd,
                   n_en =n_en, 
                   process_error = process_error, 
                   n_epochs_fine = n_epochs_fine,
                   learn_rate_fine = learn_rate_fine,
                   beta = beta,
                   alpha = alpha,
                   psi = psi,
                   seg_tave_water_mean = seg_tave_water_mean,
                   seg_tave_water_std = seg_tave_water_std,
                   obs_mean = obs_mean,
                   obs_std = obs_std, 
                   force_pos = force_pos,
                   dates = dates_trn,
                   update_h_c = update_h_c,
                   ar1_temp = ar1_temp,
                   ar1_temp_pos = ar1_temp_pos,
                   h_sd = h_sd,
                   c_sd = c_sd,
                   hidden_units = hidden_units,
                   n_segs = n_segs)
        
        #fine_tune_model = LSTMDA(1) 
        #fine_tune_model.load_weights(weights_dir) 
        #fine_tune_model.rnn_layer.build(input_shape=data['x_trn'].shape)
        
        #fine_tune_model.compile(loss=rmse_masked, optimizer=tf.optimizers.Adam(learning_rate=0.3))
        #fine_tune_model.fit(x=data['x_trn'], y=data['obs_trn'], epochs=n_epochs_fine, batch_size=n_batch)
        
        #fine_tune_model.save_weights(weights_dir)
    
        #h, c = fine_tune_model.rnn_layer.states
        #np.save(out_h_file, h.numpy())
        #np.save(out_c_file, c.numpy())
    if fine_tune:
        # add in observations as predictions; need to scale first though 
        if ar1_temp:
            scaled_obs = (obs_trn_ar1 - obs_mean) / (obs_std + 1e-10)
            x_trn[:,:,ar1_temp_pos] = np.where(np.isnan(scaled_obs[:,:,0]), x_trn[:,:,ar1_temp_pos], scaled_obs[:,:,0])
        
        if model_type == 'lstm': 
            n_batch, seq_len, n_feat = x_trn.shape
            fine_tune_model = LSTMDA(hidden_units) 
            fine_tune_model.load_weights(weights_dir).expect_partial()
            fine_tune_model(x_trn) 
            
            fine_tune_model.compile(loss=rmse_masked, optimizer=tf.keras.optimizers.Adam(learning_rate=tf.Variable(learn_rate_fine)))
            fine_tune_model.fit(x=x_trn, y=obs_trn, epochs=n_epochs_fine, batch_size=n_batch)
            
            fine_tune_model.save_weights(weights_dir)
        
            h, c = fine_tune_model.rnn_layer.states
            np.save(out_h_file, h.numpy())
            np.save(out_c_file, c.numpy())
            # trn_preds = fine_tune_model.predict(x_trn, batch_size = n_en * n_segs)
    
        elif model_type == 'rgcn':
            n_batch, seq_len, n_feat = x_trn.shape
            fine_tune_model = RGCN(hidden_units, dist_mat)
            fine_tune_model.load_weights(weights_dir).expect_partial()
            fine_tune_model(x_trn) 
            
            fine_tune_model.compile(loss=rmse_masked, optimizer=tf.keras.optimizers.Adam(learning_rate=tf.Variable(learn_rate_fine)), run_eagerly=True) # need the run eagerly to get out the states (might make this run slower so we should think about not doing this and instead trianing & then predicting on entire train input data to get out states from last time step)
            fine_tune_model.fit(x=x_trn, y=obs_trn, epochs=n_epochs_fine, batch_size=n_batch)
            
            fine_tune_model.save_weights(weights_dir)
            h = fine_tune_model.h_gr # h corresponding to the predictions
            c = fine_tune_model.c_gr # c corresponding to the predictions
            
            np.save(out_h_file, h.numpy())
            np.save(out_c_file, c.numpy())
    
    
    
    
    
