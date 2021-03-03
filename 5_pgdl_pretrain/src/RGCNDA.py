import tensorflow as tf
import numpy as np

class GraphConvLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_size, A, rand_seed=None):
        """

        :param hidden_size: [int] the number of hidden units
        :param A: [numpy array] adjacency matrix
        """
        super().__init__()
        self.A = tf.cast(A, tf.float32)
        # set up the layer
        self.lstm = tf.keras.layers.LSTMCell(hidden_size)
        self.hidden_size = hidden_size

        w_initializer = tf.random_normal_initializer(
            stddev=0.02, seed=rand_seed
        )

        # was Wg1
        self.W_graph_h = self.add_weight(
            shape=[hidden_size, hidden_size],
            initializer=w_initializer,
            name="W_graph_h",
        )
        # was bg1
        self.b_graph_h = self.add_weight(
            shape=[hidden_size], initializer="zeros", name="b_graph_h"
        )
        # was Wg2
        self.W_graph_c = self.add_weight(
            shape=[hidden_size, hidden_size],
            initializer=w_initializer,
            name="W_graph_c",
        )
        # was bg2
        self.b_graph_c = self.add_weight(
            shape=[hidden_size], initializer="zeros", name="b_graph_c"
        )

        # was Wa1
        self.W_h_cur = self.add_weight(
            shape=[hidden_size, hidden_size],
            initializer=w_initializer,
            name="W_h_cur",
        )
        # was Wa2
        self.W_h_prev = self.add_weight(
            shape=[hidden_size, hidden_size],
            initializer=w_initializer,
            name="W_h_prev",
        )
        # was ba
        self.b_h = self.add_weight(
            shape=[hidden_size], initializer="zeros", name="b_h"
        )

        # was Wc1
        self.W_c_cur = self.add_weight(
            shape=[hidden_size, hidden_size],
            initializer=w_initializer,
            name="W_c_cur",
        )
        # was Wc2
        self.W_c_prev = self.add_weight(
            shape=[hidden_size, hidden_size],
            initializer=w_initializer,
            name="W_c_prev",
        )
        # was bc
        self.b_c = self.add_weight(
            shape=[hidden_size], initializer="zeros", name="b_c"
        )

    def call(self, inputs, **kwargs):
        h_list = []
        c_list = []
        n_steps = inputs.shape[1]
        inputs = tf.cast(inputs, tf.float32)
        h_update = tf.cast(kwargs['h_init'], tf.float32)
        c_update = tf.cast(kwargs['c_init'], tf.float32)
        for t in range(n_steps):
            seq, state = self.lstm(inputs[:, t, :], states=[h_update, c_update])
            h, c = state
            h_graph = tf.nn.tanh(
                tf.matmul(
                    self.A,
                    tf.matmul(h, self.W_graph_h)
                    + self.b_graph_h,
                    )
            )
            c_graph = tf.nn.tanh(
                tf.matmul(
                    self.A,
                    tf.matmul(c, self.W_graph_c) + self.b_graph_c,
                    )
            )

            h_update = tf.nn.sigmoid(
                tf.matmul(h, self.W_h_cur)
                + tf.matmul(h_graph, self.W_h_prev)
                + self.b_h
            )
            c_update = tf.nn.sigmoid(
                tf.matmul(c, self.W_c_cur)
                + tf.matmul(c_graph, self.W_c_prev)
                + self.b_c
            )

            h_list.append(h_update)
            c_list.append(c_update)
        h_list = tf.stack(h_list)
        c_list = tf.stack(c_list)
        h_list = tf.transpose(h_list, [1, 0, 2])
        c_list = tf.transpose(c_list, [1, 0, 2])
        return h_list, c_list


class RGCN(tf.keras.Model):
    def __init__(self, hidden_size, A, dropout =0.):
        """

        :param hidden_size: [int] the number of hidden units
        :param A: [numpy array] adjacency matrix
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn_layer = tf.keras.layers.LSTM(
            hidden_size, 
            return_sequences=True, 
            stateful=True,
            return_state=True,
            dropout=dropout
        )

        self.graph_conv_layer = GraphConvLayer(A=A.astype("float32"),
                                               hidden_size=hidden_size)
        self.out_layer = tf.keras.layers.Dense(1)

        self.h_gr = None
        self.c_gr = None

    def call(self, inputs, **kwargs):
        batch_size = inputs.shape[0]
        h_init = kwargs.get('h_init', tf.zeros([batch_size, self.hidden_size]))
        c_init = kwargs.get('c_init', tf.zeros([batch_size, self.hidden_size]))
        h_gr, c_gr = self.graph_conv_layer(inputs, h_init=h_init, c_init=c_init)
        self.h_gr = h_gr
        self.c_gr = c_gr
        y_out = self.out_layer(h_gr)
        return y_out
    
    


#inputs = np.random.randn(2, 4, 4)
#y_obs = np.random.randn(2,4)
#adj_matrix = np.random.randn(2, 2)
#h_init = tf.convert_to_tensor(np.random.randn(2, 2))
#c_init = tf.convert_to_tensor(np.random.randn(2, 2))

#model = RGCN(2, adj_matrix)
#model.compile(loss = rmse_masked, optimizer=tf.optimizers.Adam(learning_rate=0.3))
#model.fit(x = inputs, y = y_obs, epochs = 2, batch_size = 2)
#predictions = model(inputs, h_init=h_init, c_init=c_init)
#h = model.h_gr # h corresponding to the predictions
#c = model.c_gr # c corresponding to the predictions
#h,c = model.states 

#updated_h = h[:, -1, :] * 500 # update just the states of just the last time step
#updated_c = c[:, -1, :] * 500
#next_inputs = np.random.randn(2, 4, 4)
#next_predictions = model(inputs, h_init=updated_h, c_init=updated_c)
#new_h = model.h_gr
#new_c = model.c_gr
