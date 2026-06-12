import tensorflow as tf
import keras 

# --------- FeedForward - Dense Section

def G_Dense(inputs: tf.Tensor, nodes: list, DP: int, n_final: int,
            act_func: str = 'leaky_relu', final_act_func: str = 'softmax',
            WI: str = 'he_normal', L1: float = 0.0, L2: float = 0.0, 
            use_bias: bool = False) -> tf.Tensor:
    '''
    Description:
        This functions does forward propagation throught a series of dense layers, returning the desired number
        of variables to characterize.


    Args:
        inputs (tf.Tensor): Layer to be processed
        nodes (list): Number of nodes in each dense layer
        DP (int): Dropout probability [0-100]
        n_final (int): Desired number of variables to be target
        act_func (str): Activation function to be used in dense layers, by default is leaky_relu
        final_act_func (str): Activation function to be used in the final layer, by default is softmax

    Output:
        tf.Tensor corresponding to the output of the last dense layer
    '''
    WR = keras.regularizers.L1L2(l1 = L1, l2=L2)
    D = keras.layers.Dense(
        nodes[0],
        activation = act_func,
        kernel_initializer = WI,
        kernel_regularizer =  WR,
        use_bias = use_bias
        )(inputs) # Pass the information through the first dense layer
    Drop = keras.layers.Dropout(DP/100)(D)
    BN = keras.layers.BatchNormalization()(Drop)

    for node in nodes[1:]:
        D = keras.layers.Dense(node,
                               activation = act_func,
                               kernel_initializer = WI,
                               kernel_regularizer =  WR,
                               use_bias = use_bias
                               )(BN) # Pass the information through several dense layers
        Drop = keras.layers.Dropout(DP/100)(D)
        BN = keras.layers.BatchNormalization()(Drop)

    return keras.layers.Dense(n_final, 
                              activation = final_act_func,
                              kernel_initializer = WI, 
                              use_bias = True
                              )(BN) # Return the information with the shape of the target variable