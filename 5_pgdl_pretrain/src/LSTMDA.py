import tensorflow as tf


class LSTMDA(tf.keras.Model):
    def __init__(self, hidden_size=1, dropout =0.):
        super().__init__()
        self.rnn_layer = tf.keras.layers.LSTM(hidden_size,
                                              return_sequences=True,
                                              stateful=True,
                                              return_state=True,
                                              dropout=dropout)
        self.out_layer = tf.keras.layers.Dense(1)

    def call(self, inputs, **kwargs):
        y, h, c = self.rnn_layer(inputs)
        out = self.out_layer(y)
        return out


def rmse_masked(y_true, y_pred):
    num_y_true = tf.cast(
        tf.math.count_nonzero(~tf.math.is_nan(y_true)), tf.float32
    )
    zero_or_error = tf.where(
        tf.math.is_nan(y_true), tf.zeros_like(y_true), y_pred - y_true
    )
    sum_squared_errors = tf.reduce_sum(tf.square(zero_or_error))
    rmse_loss = tf.sqrt(sum_squared_errors / num_y_true)
    return rmse_loss

def rmse_weighted(y_true, y_pred): # weighted by covariance matrix from DA; weights are concatonated onto y_true and need to separate out within function 
    y_true_shape = y_true.get_shape().as_list()
    y_true_end = int(y_true_shape[2] / 2)
    y_true_vals = tf.slice(y_true, [0,0,0], [y_true_shape[0],y_true_shape[1],y_true_end])
    # might need to change this slicing if moving to multiple segments 
    weights = tf.slice(y_true, [0,0,y_true_end], [y_true_shape[0],y_true_shape[1],y_true_end])
    num_y_true = tf.cast(
        tf.math.count_nonzero(~tf.math.is_nan(y_true_vals)), tf.float32
    )
    zero_or_error = tf.where(
        tf.math.is_nan(y_true_vals), tf.zeros_like(y_true_vals), y_pred - y_true_vals
    )
    zero_or_error = zero_or_error * weights
    sum_squared_errors = tf.reduce_sum(tf.square(zero_or_error))
    rmse_loss = tf.sqrt(sum_squared_errors / num_y_true)
    
    return rmse_loss