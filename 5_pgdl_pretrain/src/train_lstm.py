import tensorflow as tf
import numpy as np
import sys
sys.path.insert(1, '5_pgdl_pretrain/src')
from LSTMDA import LSTMDA, rmse_masked


def train_save_model(input_data, out_model_file, out_h_file, out_c_file):
    data = np.load(input_data)
    n_batch, seq_len, n_feat = data['x_trn'].shape
    mymodel = LSTMDA(1)
    mymodel(data['x_trn'])

    mymodel.compile(loss=rmse_masked,
                    optimizer=tf.optimizers.Adam(learning_rate=0.3))
    mymodel.fit(x=data['x_trn'], y=data['y_trn'], epochs=1, batch_size=n_batch)
    mymodel.save_weights(out_model_file)
    h, c = mymodel.rnn_layer.states
    np.save(out_h_file, h.numpy())
    np.save(out_c_file, c.numpy())


train_save_model('5_pgdl_pretrain/in/lstm_da_data_just_air_temp.npz', 'lstm_da_trained_wgts/', 'h.npy', 'c.npy')
