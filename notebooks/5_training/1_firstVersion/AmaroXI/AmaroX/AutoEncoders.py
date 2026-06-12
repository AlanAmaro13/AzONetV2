import tensorflow as tf
import keras 
# --------- AutoEncoders
def G_AE_Dense(inputs: tf.Tensor, nodes: list, DP: int, act_func = 'sigmoid', WI: str = 'glorot_normal', WR: float = 1e-4, 
                     final_act_func: str = 'sigmoid'): 
    '''
    Descriptions: 
        This functions generate a Deep Dense AutoEncoder architecture (Symmetric). Each block has the following flux: 

        Dense -> DP -> BN 

    Args:
        inputs (tf.Tensor): Refers to the input shape tensor. 
        nodes (list): Refers to the number of nodes for each dense layer
        DP (int): Dropout probability
        WI (list): Weight Initializer, the first element is the metric used and the second element is the value 
        WR (list): Weight Regulazier, by default we used L1L2, the values of the list are the values for each lambda. 
        act_func (str): Activation function used 
    '''
    
    reg = keras.regularizers.L2(l2=WR)

    # --- Encoder ------------- # [100, 50, 25, 10, 3]
    D = keras.layers.Dense(
        nodes[0], 
        activation = act_func,
        kernel_initializer = WI,
    )(inputs)
    Drop = keras.layers.Dropout(DP/100)(D)
    BN = keras.layers.BatchNormalization()(Drop) 

    # --- Main Cycle of Compresion ---------------
    for node in nodes[1:-1]:
        D = keras.layers.Dense(
            node, 
            activation = act_func,
            kernel_initializer = WI,
        )(BN)
        Drop = keras.layers.Dropout(DP/100)(D)
        BN = keras.layers.BatchNormalization()(Drop)

    # --- Bottleneck ------------------------------
    LS = keras.layers.Dense(
        nodes[-1], 
        activation = 'linear',
        kernel_initializer = WI,
        kernel_regularizer = reg, 
        name = 'latent_space'
    )(BN)
    Drop = keras.layers.Dropout(DP/100)(LS)
    BN = keras.layers.BatchNormalization()(Drop)
    
    # --- Main Cycle of Decompresion ------------- 
    _nodes = nodes[::-1][1:] # [10, 25, 50, 100]
    for node in _nodes: 
        D = keras.layers.Dense(
            node, 
            activation = act_func,
            kernel_initializer = WI,
        )(BN)
        Drop = keras.layers.Dropout(DP/100)(D)
        BN = keras.layers.BatchNormalization()(Drop)

    # --- Decoder -----------------------------
    DF = keras.layers.Dense(
        inputs.shape[1], 
        activation = final_act_func,
        kernel_initializer = WI,
        name = 'reconstruction'
    )(BN)
    
    return DF


def G_AE_Conv1D(inputs: tf.Tensor, filters: int, kernel:list, act_func: str, pad_type:str, 
               pool:int, stride:int, WIC:str, WRC:str, stride_conv:int = 1):

    # ------------------- Encoder Input 
    _stage = G_ConvBlock(
        inputs,
        filters,
        kernel[0], 
        act_func, 
        pad_type,
        pool,
        stride,
        WIC,
        WRC,
        stride_conv
    )
    # ------------------ Encoder iterations: 
    _i = 0
    for subkernel in kernel[1:-1]:
        _i+=1
        _stage = G_ConvBlock(
            _stage,
            (filters * 2**_i),
            subkernel, 
            act_func, 
            pad_type,
            pool,
            stride,
            WIC,
            WRC,
            stride_conv
        )
    # --------------- Latent Space: 
    ls = G_ConvBlock(
            _stage,
            (filters * 2** (len(kernel)-1) ),
            kernel[-1], 
            act_func, 
            pad_type,
            pool,
            stride,
            WIC,
            WRC,
            stride_conv,
            None
        )
    # -------------- Decoder Input: [30, 20, 10, 5, 5] -> [5, 5, 10, 20, 30] -> [5, 10, 20, 30]
    _i -+1
    _stage = G_DeConvBlock(
                    ls,
                    (filters * 2**_i),
                    kernel[::-1][1], 
                    act_func, 
                    pad_type,
                    stride,
                    WIC,
                    WRC,
                    None
                )
    
    # ------------ Decoder Flux
    _kernel = kernel[::-1][2:]
    _i -=1
    for subkernel in _kernel:
        _stage = G_DeConvBlock(
                    _stage,
                    (filters * 2**_i),
                    subkernel, 
                    act_func, 
                    pad_type,
                    stride,
                    WIC,
                    WRC,
                    None
                )
        _i -=1

    return keras.layers.Convolution1D(1, 1, padding = 'same')(_stage)

