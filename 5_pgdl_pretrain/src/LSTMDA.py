import tensorflow as tf


class LSTMDA(tf.keras.Model):
    def __init__(self, hidden_size=1):
        super().__init__()
        self.rnn_layer = tf.keras.layers.LSTM(hidden_size,
                                              return_sequences=True,
                                              stateful=True,
                                              return_state=True)
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