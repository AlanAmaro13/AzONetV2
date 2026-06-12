import tensorflow as tf
import keras 

# ---
def UNET_ConvDown(inputs: tf.Tensor, filters: int, power_of_two: int, kernel:int, act_func: str, pad_type:str, 
                pool:int, stride:int, WIC:str, WRC, stride_conv: int = 1, pool_op = 'AP', pool_bool: bool = True) -> tf.Tensor:
    '''
        Applies a downsampling block consisting of two convolutional layers, 
        batch normalization, activation functions, and optional pooling.
    
        Args:
            inputs (tf.Tensor): Input tensor of shape (batch_size, sequence_length, channels).
            filters (int): Base number of filters (multiplied by `2^power_of_two`).
            power_of_two (int): Exponent to scale filters as `filters * (2^power_of_two)`.
            kernel (int): Kernel size for convolutional layers.
            act_func (str): Activation function name (e.g., 'relu', 'sigmoid').
            pad_type (str): Padding type for pooling ('same' or 'valid').
            pool (int): Pooling window size.
            stride (int): Stride for pooling operations.
            WIC (str): Weight initializer for convolutional layers.
            WRC: Weight regularizer for convolutional layers.
            stride_conv (int, optional): Stride for convolutional layers. Defaults to 1.
            pool_op (str, optional): Pooling operation: 'AP' (AveragePooling), 'MP' (MaxPooling), or None. Defaults to 'AP'.
            pool_bool (bool, optional): If True, apply pooling and return both pooled and pre-pooling tensors. Defaults to True.
    
        Returns:
            tf.Tensor: If `pool_bool = False`, returns the activated output after convolutions.
            tuple[tf.Tensor, tf.Tensor]: If `pool_bool = True`, returns:
                - Pooled output tensor
                - Pre-pooling activation tensor (skip connection)
    
        Architecture:
            1. Conv1D → BatchNorm → Activation
            2. Conv1D → BatchNorm → Activation
            3. (Optional) Pooling (Average/Max) or identity pass
            '''

    n, f = power_of_two, filters

    Conv1 = keras.layers.Conv1D(
        filters = int( (2**n) *f),
        kernel_size = kernel,
        strides = stride_conv, 
        padding = 'same',
        kernel_initializer = WIC, 
        kernel_regularizer = WRC
    )(inputs)
    
    BN = keras.layers.BatchNormalization()(Conv1)
    Act = keras.layers.Activation(activation = act_func)(BN)

    Conv2 = keras.layers.Conv1D(
        filters = int( (2**n) *f),
        kernel_size = kernel,
        strides = stride_conv, 
        padding = 'same',
        kernel_initializer = WIC, 
        kernel_regularizer = WRC
    )(Act)

    BN = keras.layers.BatchNormalization()(Conv2)
    Act = keras.layers.Activation(activation = act_func)(BN)

    if pool_bool:

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
            
        return AP, Act 

    else:
        
        return Act 

# -----------------------------------------
def UNET_ConvUp_US(inputs: tf.Tensor, block: tf.Tensor, filters: int, power_of_two: int, kernel:int, act_func: str, WIC:str, WRC:str, 
                   stride_conv: int = 1) -> tf.Tensor:
    '''
    Performs upsampling and concatenation with skip connection followed by two convolutional layers.
    Handles dimension matching between encoder and decoder paths with automatic padding.

    Args:
        inputs (tf.Tensor): Input tensor from previous decoder layer.
        block (tf.Tensor): Skip connection tensor from corresponding encoder layer.
        filters (int): Base number of filters (multiplied by `2^power_of_two`).
        power_of_two (int): Exponent to scale filters as `filters * (2^power_of_two)`.
        kernel (int): Kernel size for convolutional layers.
        act_func (str): Activation function name (e.g., 'relu', 'sigmoid').
        WIC (str): Weight initializer for convolutional layers.
        WRC (str): Weight regularizer for convolutional layers.
        stride_conv (int, optional): Stride for convolutional layers. Defaults to 1.

    Returns:
        tf.Tensor: Processed tensor after upsampling, concatenation, and convolutions.

    Architecture:
        1. Calculate required upsampling size and padding
        2. Upsample input tensor
        3. Apply zero-padding if necessary for dimension matching
        4. Concatenate with skip connection
        5. Apply two Conv1D → BatchNorm → Activation blocks

    Dimension Handling:
        - Automatically calculates upsampling factor (n) and padding (m)
        - Handles both divisible and non-divisible dimension relationships
        - Uses zero-padding for fractional dimension adjustments
    '''

    _f, _n = filters, power_of_two
    
    # ---- UpSampling Section 
    dim_in, dim_out = inputs.shape[1], block.shape[1] # Get the corresponding dimensions 
    
    n = dim_out//dim_in
    r = dim_out%dim_in
    m = dim_out - (n*dim_in)

    if r == 0 :
        _Up = keras.layers.UpSampling1D(size = n)(inputs)
        
    elif r!=0: 
        _Up = keras.layers.UpSampling1D(size = n)(inputs)
        _Up = keras.layers.ZeroPadding1D(padding = (0,m))(_Up)
    
    # ---- Concatenate Section
    _Up =keras.layers.Concatenate()([_Up, block]) # Match the UpSampled and the corresponding block

    # ---- Process the block 

    Conv1 = keras.layers.Conv1D(
        filters = int( (2**_n) *_f),
        kernel_size = kernel,
        strides = stride_conv, 
        padding = 'same',
        kernel_initializer = WIC, 
        kernel_regularizer = WRC
    )(_Up)
    
    BN = keras.layers.BatchNormalization()(Conv1)
    Act = keras.layers.Activation(activation = act_func)(BN)

    Conv2 = keras.layers.Conv1D(
        filters = int( (2**_n) *_f),
        kernel_size = kernel,
        strides = stride_conv, 
        padding = 'same',
        kernel_initializer = WIC, 
        kernel_regularizer = WRC
    )(Act)

    BN = keras.layers.BatchNormalization()(Conv2)
    Act = keras.layers.Activation(activation = act_func)(BN)

    return Act

# -----------------------------
def Up_Match_USConv(inputs: tf.Tensor, block: tf.Tensor):
    '''
    Description:
        This function takes into a inputs and scale it using UpSampling1D to the desire dimension (the input layer). 
        As the UpSampling conserve the number of filters, a posterior convolutional layer is required.

    Args:
        inputs (tf.Tensor): Tensor to be UpSampled 
        block (tf.Tensor): Tensor to match dimensions with
    '''
    # ---- UpSampling Section 
    dim_in, dim_out = inputs.shape[1], block.shape[1] # Get the corresponding dimensions 
    n = dim_out//dim_in
    r = dim_out%dim_in
    m = dim_out - (n*dim_in)

    if r == 0 :
        _Up = keras.layers.UpSampling1D(size = n)(inputs)
        
    elif r!=0: 
        _Up = keras.layers.UpSampling1D(size = n)(inputs)
        _Up = keras.layers.ZeroPadding1D(padding = (0,m))(_Up)


    return _Up

# -------------------------------

def G_UNET(inputs: tf.Tensor, layers:int, unet_kernel: list, WIC:str, WRC, pad_type:str, unet_act_func = 'leaky_relu', 
           pool: int = 12, stride: int = 3, final_func_act: str = 'relu', stride_conv: int = 1, pool_op:str = 'AP', 
           pool_bool: bool = True):
    '''
    Constructs a complete 1D U-Net generator with symmetric encoder-decoder architecture
    and skip connections between corresponding encoder and decoder blocks.

    Args:
        inputs (tf.Tensor): Input tensor of shape (batch_size, sequence_length, channels).
        layers (int): Base number of filters (scaled exponentially across layers).
        unet_kernel (list): List of kernel sizes for each encoder/decoder block.
        WIC (str): Weight initializer for all convolutional layers.
        WRC: Weight regularizer for all convolutional layers.
        pad_type (str): Padding type for pooling operations ('same' or 'valid').
        unet_act_func (str, optional): Activation function for intermediate layers. Defaults to 'leaky_relu'.
        pool (int, optional): Pooling window size. Defaults to 12.
        stride (int, optional): Stride for pooling operations. Defaults to 3.
        final_func_act (str, optional): Activation function for final output layer. Defaults to 'relu'.
        stride_conv (int, optional): Stride for convolutional layers. Defaults to 1.
        pool_op (str, optional): Pooling operation type: 'AP' (AveragePooling) or 'MP' (MaxPooling). Defaults to 'AP'.
        pool_bool (bool, optional): Whether to apply pooling in encoder blocks. Defaults to True.

    Returns:
        tf.Tensor: Output tensor with the same temporal dimensions as input, processed through U-Net.

    Architecture:
        Encoder Path:
          - Series of downsampling blocks (UNET_ConvDown)
          - Each block doubles the number of filters (2^power_of_two scaling)
          - Skip connections are stored for decoder path

        Bottleneck:
          - Highest abstraction level without pooling
          - Maximum filter count (2^(len(unet_kernel)-1 * layers)

        Decoder Path:
          - Series of upsampling blocks (UNET_ConvUp_US)
          - Each block halves the number of filters
          - Skip connections from encoder are concatenated
          - Uses learned upsampling with dimension matching

        Final Layer:
          - 1x1 convolution to produce output
          - Dimension matching with original input

    Notes:
        - The number of encoder/decoder blocks is determined by the length of unet_kernel
        - Uses exponential filter scaling (2^power_of_two) throughout the network
        - Automatically handles dimension matching between encoder and decoder paths
    '''

    _blocks = [] 

    # Enconder Section ----------------- ConvDown
    #UNET_ConvDown(inputs: tf.Tensor, filters: int, power_of_two: int, kernel:int, act_func: str, pad_type:str, 
    #            pool:int, stride:int, WIC:str, WRC, stride_conv: int = 1, pool_op = 'AP', pool_bool: bool = True) -> tf.Tensor:

    
    Encoder, Concat = UNET_ConvDown(
        inputs= inputs, 
        filters = layers, 
        power_of_two = 0, 
        kernel = unet_kernel[0], 
        act_func = unet_act_func, 
        pad_type = pad_type, 
        pool = pool, 
        stride = stride, 
        WIC = WIC, 
        WRC = WRC, 
        stride_conv = stride_conv, 
        pool_op = pool_op, 
        pool_bool = pool_bool) 
    
    _blocks.append(Concat) # Save the first block

    for i in range(1, len(unet_kernel) - 1 ):
        Encoder, Concat = UNET_ConvDown(
            inputs= Encoder, 
            filters = layers, 
            power_of_two = i, 
            kernel = unet_kernel[i], 
            act_func = unet_act_func, 
            pad_type = pad_type, 
            pool = pool, 
            stride = stride, 
            WIC = WIC, 
            WRC = WRC, 
            stride_conv = stride_conv, 
            pool_op = pool_op, 
            pool_bool = pool_bool) 
        
        _blocks.append(Concat) # Save each block
    
    # Bottleneck -------------------------------------
    Bottleneck = UNET_ConvDown(
            inputs= Encoder, 
            filters = layers, 
            power_of_two = len(unet_kernel)-1, 
            kernel = unet_kernel[-1], 
            act_func = unet_act_func, 
            pad_type = pad_type, 
            pool = pool, 
            stride = stride, 
            WIC = WIC, 
            WRC = WRC, 
            stride_conv = stride_conv, 
            pool_op = pool_op, 
            pool_bool = False) 
    
    
    # Decoder Section ---------------- ConvUp_US
    # UNET_ConvUp_US(inputs: tf.Tensor, block: tf.Tensor, filters: int, power_of_two: int, kernel:int, act_func: str, WIC:str, WRC:str, 
    #               stride_conv: int = 1) -> tf.Tensor:
    Decoder = UNET_ConvUp_US(
        inputs = Bottleneck, 
        block = _blocks[-1], 
        filters= layers, 
        power_of_two= len(unet_kernel)-2, 
        kernel= unet_kernel[-2], 
        act_func = unet_act_func, 
        WIC=WIC, 
        WRC=WRC, 
        stride_conv= stride_conv)
    
    
    i = len(unet_kernel) -3
    for _block in _blocks[ -2: : -1]:        
        Decoder = UNET_ConvUp_US(
            inputs = Decoder, 
            block = _block, 
            filters= layers, 
            power_of_two= i, 
            kernel= unet_kernel[i], 
            act_func = unet_act_func, 
            WIC=WIC, 
            WRC=WRC, 
            stride_conv = stride_conv)
        
        i -= 1

    Final_Stage = keras.layers.Conv1D( 
        filters = 1, 
        kernel_size = 1,
        kernel_initializer= WIC,
        activation = final_func_act)( 
            Up_Match_USConv(Decoder, inputs) 
        )

    return Final_Stage

# ------------------------------------------------