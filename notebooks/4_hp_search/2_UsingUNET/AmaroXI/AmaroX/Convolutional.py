import tensorflow as tf
import keras
from AmaroXI.AmaroX.DNN import *

# --------- Convolutional Section
def G_ConvBlock_1D(inputs: tf.Tensor, filters: int, kernel:int, act_func: str, pad_type:str, 
                pool:int, stride:int, WIC:str, WRC, stride_conv: int = 1, pool_op = 'AP'):
    '''
    Args: 
        inputs (Tensor): The input information for the Conv1D
        filters (int): The number of filters in the Convolutional
        kernel (int): kernel size in the Conv
        stride_conv (int): strides in the Conv
        WIC (str): Kernel Initializer in the Conv
        WRC (str): Kernel Regulatier in the Conv 
        act_func (str): Activation Function 
        pool (int): pool size in the pooling layer
        stride (int): stride size in the pooling layer

    Flux: 
        Conv->BN->Activation->AP

    Observations: 
        1. The Convolutionals preserves dimension. When padding="same" and strides=1, the output has the same size as the input
        2. Before is the BatchNormalization and after the Activation. 
        3. The dimension after the Conv is
            (input_shape - pool_size + 1) / strides) 

    Test:
        G_ConvBlock_AP(keras.layers.Input((200, 1)), 10, 10, 'relu', 4, 4, 'he_normal', keras.regularizers.L2(0.1))


    Pooling Operations: 
    Downsamples the input representation by taking the maximum value over a spatial window of size pool_size. 
    The window is shifted by strides.

    The resulting output when using the "valid" padding option has a shape of: output_shape = (input_shape - pool_size + 1) / strides).

    The resulting output shape when using the "same" padding option is: output_shape = input_shape / strides
    '''

    Conv = keras.layers.Conv1D(
        filters = filters,
        kernel_size = kernel,
        strides = stride_conv, 
        padding = 'same',
        kernel_initializer = WIC, 
        kernel_regularizer = WRC
    )(inputs)

    BN = keras.layers.BatchNormalization()(Conv)
    Act = keras.layers.Activation(activation = act_func)(BN)

    if pool_op == 'AP':
        AP = keras.layers.AveragePooling1D(
            pool_size = pool,
            strides = stride, 
            padding = pad_type
            )(Act)
    elif pool_op == 'MP':
        AP = keras.layers.MaxPooling1D(
        pool_size = pool,
        strides = stride, 
        padding = pad_type
        )(Act)

    elif pool_op == None:
        AP = Act

    return AP

def G_DeConvBlock_1D_Transpose(inputs: tf.Tensor, filters: int, kernel:int, act_func: str, pad_type: str, 
                  stride_deconv:int, WIC:str, WRC:str, out_pad = None):
    '''
    Args: 
        inputs (Tensor): The input information for the Conv1D
        filters (int): The number of filters in the Convolutional
        kernel (int): kernel size in the Conv
        stride_conv (int): strides in the Conv
        WIC (str): Kernel Initializer in the Conv
        WRC (str): Kernel Regulatier in the Conv 
        act_func (str): Activation Function 
        stride (int): stride size in the pooling layer

    Flux: 
        Conv->BN->Activation

    Observations: 
        1. The Convolutionals preserves dimension. 
        2. Before is the BatchNormalization and after the Activation. 
        3. The dimension after the DeConv is
            (input_shape - 1) * S + K

    Test:
        G_DeConvBlock(keras.layers.Input((50,1)), 10, 4, 'relu', 4, 'he_normal', None)
    '''

    Conv = keras.layers.Conv1DTranspose(
            filters = filters,
            kernel_size = kernel,
            strides = stride_deconv, 
            output_padding = out_pad,
            padding = pad_type,
            kernel_initializer = WIC, 
            kernel_regularizer = WRC
        )(inputs)

    BN = keras.layers.BatchNormalization()(Conv)
    Act = keras.layers.Activation(activation = act_func)(BN)
    
    return Act

# ---------- Convolutional Neural Network 

def CNN(inputs: tf.Tensor, filters:list, kernel:list, pad_type:str, pool:int, stride:int, nodes: list, DP: int, n_final: int,
       WIC:str, WRC, stride_conv:int = 1, pool_op:str= 'AP', act_func: str = 'leaky_relu', 
        final_act_func: str = 'softmax', WI: str = 'he_normal', L1: float = 0.0, L2: float = 0.0, use_bias: bool = False, 
       act_func_conv:str = 'leaky_relu'): 
    '''
    Desc: 
        This function is used to create a Convolutional Neural Network. It provides a easy interface to perform Hyper-parameters research. 

        The initial component is the input, this is usually a keras.layers.Input((n,1)), where n is the dimension of the signal. To define
        the number of ConvBlock we used a list for the filters, each element in this list represents the filter for each ConvBlock. We also
        provide the use the ability to change the kernel size depending of the ConvBlock (if you want to set a fix kernel size just use 
        the operation Deep*[kernel_size]). 

        The activation function for the convolutional is set to LeakyReLU (better results might be obtained with a SReLU). An important note
        is: 
        
        * The Convolutional section preverves the dimensionality. This means that all the reduction is performed by the Pooling Operations.
        This also implies that the padding in the Conv is set to 'same' and stride_conv=1. 

        The Pooling Operations are configured to be: 
        * AP: Average Pooling
        * MP: Max Pooling 
        It's easy to observe that a new pooling operation can be added without major problem. The parameters to reduce dimension are: pad_type,
        pool and stride, the final dimension is calculated as follows: 

        - The resulting output when using the "valid" padding option has a shape of: output_shape = (input_shape - pool_size + 1) / strides).

        - The resulting output shape when using the "same" padding option is: output_shape = input_shape / strides

        WIC means Weight Initializer for the Convolutional. The best WIC is He Normal (or He Uniform, depending if the architecture has BachNorm),
        WRC means Weight Regularizer, I recommend the use of L1L2. 

        This flux is repeated for each filter in filters. After the convolutional operations, a flatten operation is applied. Then we employ
        the G_Dense section. Here the input is the output flatten, the nodes are defined as a list, where each elements are the nodes in each
        hidden layer. WI means Weight initializer (again, I recommend use He Normal). For regularization I added Dropout (integer number 
        represeting a porcentage), L1L2 (ElasticNet). The activation function inside the architecture might be different from the final, so I 
        give that option (I use LeakyReLU for hiddden layers and ReLU for output). Finally, you can decide whether you want to use bias in the 
        Dense Layers. 
    
    Args: 
        inputs (tf.Tensor): This is the Tensor input, like keras.layers.Input((64,1))
        filters (list): This list contains each filter for each convolutional block. 
        kernel (list): This list contains the kernel size for each convolutional block. 
        act_func (str): This is the activation function, I really recommed use a rectified operation, such as LeakyReLU, ReLU, PReLU, SRELU. 

        -> These parameters are for the Pooling Operation. 
        pad_type (str): 
        pool (int): 
        stride (int): 

        WIC (str): This is the Weight Initializer for the Convolutional, I really recommend using He Normal (because the BatchNorm). 
        WRC (KerasObject): This is the kernel regularizer, I really recommend using L1L2, varying the values. 
        stride_conv (int): This is the stride for the Convolutional

        pool_op (str): This is the Pooling Operation. 

    Returns: 
        CNN Model

    Flux: 
        ConvBlock -> ConvBlock -> ... -> Flatten -> G_Dense 

    Example: 
        modelo = CNN(
            inputs = keras.layers.Input((911,1)),
            filters = [256, 128, 64], 
            kernel = [25, 15, 10], 
            pad_type = 'valid', 
            pool = 4, 
            stride = 4, 
            nodes = [50, 25, 5], 
            DP = 5, 
            n_final = 1, 
            WIC = 'he_normal', 
            WRC = keras.regularizers.L1L2(l1 = 1e-6, l2 = 1e-6), 
            final_act_func= 'relu', 
            L1 = 1e-6, 
            L2 = 1e-6, 
            use_bias = True
        )
    '''

    _stage = G_ConvBlock_1D(
        inputs = inputs, 
        filters = filters[0], 
        kernel = kernel[0], 
        act_func = act_func_conv, 
        pad_type = pad_type, 
        pool = pool, 
        stride = stride, 
        WIC = WIC, 
        WRC = WRC, 
        stride_conv=stride_conv, 
        pool_op=pool_op 
    )

    for i in range(len(filters[0:])): 
        _stage = G_ConvBlock_1D(
            inputs = _stage, 
            filters = filters[i], 
            kernel = kernel[i], 
            act_func = act_func_conv, 
            pad_type = pad_type, 
            pool = pool, 
            stride = stride, 
            WIC = WIC, 
            WRC = WRC, 
            stride_conv=stride_conv, 
            pool_op=pool_op 
        )

    _flatten = keras.layers.Flatten()(_stage)

    _dense = G_Dense(
        inputs = _flatten, 
        nodes = nodes, 
        DP = DP, 
        n_final = n_final, 
        act_func = act_func, 
        final_act_func= final_act_func, 
        WI = WI, 
        L1 = L1, 
        L2 = L2, 
        use_bias = use_bias  
    )

    return keras.models.Model(inputs = inputs, outputs = _dense)