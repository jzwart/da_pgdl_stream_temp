import numpy as np
from LSTMDA import LSTMDA


data = np.load('lstm_da_data_just_air_temp.npz')
h = np.load('c.npy', allow_pickle=True)
c = np.load('h.npy', allow_pickle=True)
model = LSTMDA(1)
model.load_weights('lstm_da_trained_wgts/')
model.rnn_layer.build(input_shape=data['x_pred'].shape)

# initialize the states with the previous ones (from training)
model.rnn_layer.reset_states(states=[h, c])
p = model.predict(data['x_pred'], batch_size=1)
print(p)


# adjust the states
model.rnn_layer.reset_states(states=[np.array([[-10]]), np.array([[-100]])])
p = model.predict(data['x_pred'], batch_size=1)
print(p)
