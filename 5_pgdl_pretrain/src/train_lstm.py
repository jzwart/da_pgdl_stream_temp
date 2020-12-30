import tensorflow as tf
import numpy as np
import sys
sys.path.insert(1, '5_pgdl_pretrain/src')
from LSTMDA import LSTMDA, rmse_masked


def train_save_model(
        input_data,
        out_model_file, 
        out_h_file,
        out_c_file,
        n_epochs
):
    
    data = np.load(input_data)
    n_batch, seq_len, n_feat = data['x_trn'].shape
    mymodel = LSTMDA(1)
    mymodel(data['x_trn'])

    mymodel.compile(loss=rmse_masked,
                    optimizer=tf.optimizers.Adam(learning_rate=0.3))
    mymodel.fit(x=data['x_trn'], y=data['y_trn'], epochs=n_epochs, batch_size=n_batch)
    #for i in range(n_epochs):
    #    mymodel.fit(x=data['x_trn'], y=data['y_trn'], epochs=1, batch_size=n_batch)
    #    mymodel.reset_states()
    mymodel.save_weights(out_model_file)
    h, c = mymodel.rnn_layer.states
    np.save(out_h_file, h.numpy())
    np.save(out_c_file, c.numpy())


train_save_model(input_data = '5_pgdl_pretrain/in/lstm_da_data_just_air_temp.npz',
                 out_model_file = '5_pgdl_pretrain/out/lstm_da_trained_wgts/',
                 out_h_file = '5_pgdl_pretrain/out/h.npy', 
                 out_c_file = '5_pgdl_pretrain/out/c.npy',
                 n_epochs = 50)
