from models.keras_vision_transformer import swin_layers
from models.keras_vision_transformer import transformer_layers
from tensorflow.keras.layers import Input, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, Concatenate, Activation, Add, multiply, add, concatenate, LeakyReLU, ZeroPadding2D, UpSampling2D, BatchNormalization, Dense, AveragePooling2D
from tensorflow.keras.models import Model
import tensorflow.keras.layers as layers
import tensorflow as tf
from keras import backend as K

######################################################################################################################################
############### SWIN-Unet ############################################################################################################
######################################################################################################################################

def swin_transformer_stack(X, stack_num, embed_dim, num_patch, num_heads, window_size, num_mlp, shift_window=True, name='', mlp_drop_rate = 0, attn_drop_rate = 0, proj_drop_rate = 0, drop_path_rate = 0):
    '''
    Stacked Swin Transformers that share the same token size.
    
    Alternated Window-MSA and Swin-MSA will be configured if `shift_window=True`, Window-MSA only otherwise.
    *Dropout is turned off.
    '''
    # Turn-off dropouts
     # Droupout after each MLP layer
     # Dropout after Swin-Attention
     # Dropout at the end of each Swin-Attention block, i.e., after linear projections
     # Drop-path within skip-connections
    
    qkv_bias = True # Convert embedded patches to query, key, and values with a learnable additive value
    qk_scale = None # None: Re-scale query based on embed dimensions per attention head # Float for user specified scaling factor
    
    if shift_window:
        shift_size = window_size // 2
    else:
        shift_size = 0
    
    for i in range(stack_num):
    
        if i % 2 == 0:
            shift_size_temp = 0
        else:
            shift_size_temp = shift_size

        X = swin_layers.SwinTransformerBlock(dim=embed_dim, 
                                             num_patch=num_patch, 
                                             num_heads=num_heads, 
                                             window_size=window_size, 
                                             shift_size=shift_size_temp, 
                                             num_mlp=num_mlp, 
                                             qkv_bias=qkv_bias, 
                                             qk_scale=qk_scale,
                                             mlp_drop=mlp_drop_rate, 
                                             attn_drop=attn_drop_rate, 
                                             proj_drop=proj_drop_rate, 
                                             drop_path_prob=drop_path_rate, 
                                             name='name{}'.format(i))(X)
    return X


def swin_unet_2d_base(input_tensor, filter_num_begin, depth, stack_num_down, stack_num_up, 
                      patch_size, num_heads, window_size, num_mlp, shift_window=True, name='swin_unet', mlp_drop_rate = 0, attn_drop_rate = 0, proj_drop_rate = 0, drop_path_rate = 0):
    '''
    The base of Swin-UNET.
    
    The general structure:
    
    1. Input image --> a sequence of patches --> tokenize these patches
    2. Downsampling: swin-transformer --> patch merging (pooling)
    3. Upsampling: concatenate --> swin-transfprmer --> patch expanding (unpooling)
    4. Model head
    
    '''
    # Compute number be patches to be embeded
    input_size = input_tensor.shape.as_list()[1:]
    num_patch_x = input_size[0]//patch_size[0]
    num_patch_y = input_size[1]//patch_size[1]
    
    # Number of Embedded dimensions
    embed_dim = filter_num_begin
    
    depth_ = depth
    
    X_skip = []

    X = input_tensor
    
    # Patch extraction
    X = transformer_layers.patch_extract(patch_size)(X)

    # Embed patches to tokens
    X = transformer_layers.patch_embedding(num_patch_x*num_patch_y, embed_dim)(X)
    
    # The first Swin Transformer stack
    X = swin_transformer_stack(X, 
                               stack_num=stack_num_down, 
                               embed_dim=embed_dim, 
                               num_patch=(num_patch_x, num_patch_y), 
                               num_heads=num_heads[0], 
                               window_size=window_size[0], 
                               num_mlp=num_mlp, 
                               shift_window=shift_window, 
                               name='{}_swin_down0'.format(name), 
                               mlp_drop_rate = mlp_drop_rate,
                               attn_drop_rate = attn_drop_rate,
                               proj_drop_rate = proj_drop_rate,
                               drop_path_rate = drop_path_rate)
    X_skip.append(X)
    
    # Downsampling blocks
    for i in range(depth_-1):
        
        # Patch merging
        X = transformer_layers.patch_merging((num_patch_x, num_patch_y), embed_dim=embed_dim, name='down{}'.format(i))(X)
        
        # update token shape info
        embed_dim = embed_dim*2
        num_patch_x = num_patch_x//2
        num_patch_y = num_patch_y//2
        
        # Swin Transformer stacks
        X = swin_transformer_stack(X, 
                                    stack_num=stack_num_down, 
                                    embed_dim=embed_dim, 
                                    num_patch=(num_patch_x, num_patch_y), 
                                    num_heads=num_heads[i+1], 
                                    window_size=window_size[i+1], 
                                    num_mlp=num_mlp, 
                                    shift_window=shift_window, 
                                    name='{}_swin_down{}'.format(name, i+1),
                                    mlp_drop_rate = mlp_drop_rate,
                                    attn_drop_rate = attn_drop_rate,
                                    proj_drop_rate = proj_drop_rate,
                                    drop_path_rate = drop_path_rate)
        
        # Store tensors for concat
        X_skip.append(X)
        
    # reverse indexing encoded tensors and hyperparams
    X_skip = X_skip[::-1]
    num_heads = num_heads[::-1]
    window_size = window_size[::-1]
    
    # upsampling begins at the deepest available tensor
    X = X_skip[0]
    
    # other tensors are preserved for concatenation
    X_decode = X_skip[1:]
    
    depth_decode = len(X_decode)
    
    for i in range(depth_decode):
        
        # Patch expanding
        X = transformer_layers.patch_expanding(num_patch=(num_patch_x, num_patch_y), 
                                               embed_dim=embed_dim, 
                                               upsample_rate=2, 
                                               return_vector=True, name='{}_patch_expanding_{}'.format(name, i))(X)
        

        # update token shape info
        embed_dim = embed_dim//2
        num_patch_x = num_patch_x*2
        num_patch_y = num_patch_y*2
        
        # Concatenation and linear projection
        X = concatenate([X, X_decode[i]], axis=-1, name='{}_concat_{}'.format(name, i))
        X = Dense(embed_dim, use_bias=False, name='{}_concat_linear_proj_{}'.format(name, i))(X)
        
        # Swin Transformer stacks
        X = swin_transformer_stack(X, 
                                    stack_num=stack_num_up, 
                                    embed_dim=embed_dim, 
                                    num_patch=(num_patch_x, num_patch_y), 
                                    num_heads=num_heads[i], 
                                    window_size=window_size[i], 
                                    num_mlp=num_mlp, 
                                    shift_window=shift_window, 
                                    name='{}_swin_up{}'.format(name, i),
                                    mlp_drop_rate = mlp_drop_rate,
                                    attn_drop_rate = attn_drop_rate,
                                    proj_drop_rate = proj_drop_rate,
                                    drop_path_rate = drop_path_rate)
        
    # The last expanding layer; it produces full-size feature maps based on the patch size
    # !!! <--- "patch_size[0]" is used; it assumes patch_size = (size, size)
    
    X = transformer_layers.patch_expanding(num_patch=(num_patch_x, num_patch_y), 
                                           embed_dim=embed_dim, 
                                           upsample_rate=patch_size[0], 
                                           return_vector=False, name='patch_expanding_last')(X)
    
    return X



def Swin_unet(
    input_size = (1024, 1024, 1),
    filter_num_begin = 128,     # number of channels in the first downsampling block; it is also the number of embedded dimensions
    depth = 4,                 # the depth of SwinUNET; depth=4 means three down/upsampling levels and a bottom level 
    stack_num_down = 2,         # number of Swin Transformers per downsampling level
    stack_num_up = 2,           # number of Swin Transformers per upsampling level
    patch_size = 8,        # Extract 4-by-4 patches from the input image. Height and width of the patch must be equal.
    att_heads = 4,   # number of attention heads per down/upsampling level
    w_size = 4, # the size of attention window per down/upsampling level
    mlp_ratio = 4, # Ratio of mlp hidden dim to embedding dim. Default: 4
    shift_window=True,          # Apply window shifting, i.e., Swin-MSA,
    input_centering=True,
    dropout = 0.1
    ):
    
    num_mlp = mlp_ratio*filter_num_begin              # number of MLP nodes within the Transformer
    
    num_heads =   [2**i*att_heads for i in range(depth)] # number of attention heads per down/upsampling level
    window_size = [w_size for _ in range(depth)]
    
    
    mlp_drop_rate = dropout
    attn_drop_rate = dropout
    proj_drop_rate = dropout
    drop_path_rate = dropout

    IN = Input(input_size)
    
    if input_centering:
        # Create a Lambda layer to replace 0.5 with -1
        replace_half_with_negative_one = tf.keras.layers.Lambda(
            lambda x: tf.where(tf.equal(x, 0.5), tf.constant(-1.0, dtype=x.dtype), x)
        )

        IN = replace_half_with_negative_one(IN)
    else:
        pass
    

    # Base architecture
    X = swin_unet_2d_base(IN, filter_num_begin, depth, stack_num_down, stack_num_up, 
                          (patch_size,patch_size), num_heads, window_size, num_mlp, 
                          shift_window=shift_window, name='swin_unet',                                     
                          mlp_drop_rate = mlp_drop_rate,
                        attn_drop_rate = attn_drop_rate,
                        proj_drop_rate = proj_drop_rate,
                        drop_path_rate = drop_path_rate)
    # Output section
    OUT = Conv2D(1, kernel_size=1, use_bias=False, activation='relu')(X)

    # Model configuration
    model = Model(inputs=[IN,], outputs=[OUT,])
    
    return model

######################################################################################################################################
############### SWIN-Unet END   ######################################################################################################
######################################################################################################################################



######################################################################################################################################
############### CASPIAN ##############################################################################################################
######################################################################################################################################

def resnetxt_block(x, filters, cardinality=32, bottleneck_width=4, strides=1, activation='tanh', init="glorot_normal", kernel=3):
    # Shortcut connection
    shortcut = x
    # Compute the width of each group (path)
    group_width = bottleneck_width * cardinality
    # 1x1 Convolution
    x = layers.Conv2D(group_width, (1, 1), padding='same', strides=strides, activation=activation, kernel_initializer=init)(x)
    # Grouped 3x3 Convolution
    x = layers.Conv2D(group_width, (kernel, kernel), padding='same', strides=1, groups=cardinality, activation=activation, kernel_initializer=init)(x)
    #x = SpatialDropout2D(0.2)(x)
    # 1x1 Convolution
    x = layers.Conv2D(filters, (1, 1), padding='same', strides=1, kernel_initializer=init)(x)
    # Adjust the shortcut path if needed
    if strides != 1 or x.shape[-1] != shortcut.shape[-1]:
        shortcut = layers.Conv2D(filters, (1, 1), strides=strides)(shortcut)
    # Element-wise addition
    x = layers.add([shortcut, x])
    x = layers.Activation(activation)(x)
    return x


def CASPIAN(input_shape=(1024, 1024, 1), filters=81, cardinality=36, activation='tanh', init="glorot_normal",
                    input_centering=True, depth=4, bottleneck_depth=8, compression_factor=0.9, gr_level=4, sup_level=1):
    
    input_grid = layers.Input(shape=input_shape)
    
    if input_centering:
        # Create a Lambda layer to replace 0.5 with -1
        replace_half_with_negative_one = layers.Lambda(
            lambda x: tf.where(tf.equal(x, 0.5), tf.constant(-1.0, dtype=x.dtype), x)
        )
        x = replace_half_with_negative_one(input_grid)
    else:
        x = input_grid

    # Encoder
    pools = []
    for i in range(depth):
        if i == 0:
            #Entry block
            pool_size=2
            s = layers.Conv2D(filters=filters, kernel_size=(2, 2), strides=(2, 2), padding='same', activation=activation, kernel_initializer=init)(x)
            #Stratified Pooling
            if input_centering:
                mask =  layers.Lambda(
                    lambda x: tf.where(tf.equal(x, -1), x, tf.zeros_like(x))
                )
            else:
                mask =  layers.Lambda(
                    lambda x: tf.where(tf.equal(x, 0.5), x, tf.zeros_like(x))
                )
            pool_1 = mask(x)
            mask =  layers.Lambda(
                    lambda x: tf.where(tf.equal(x, 1), x, tf.zeros_like(x))
            )
            pool_2 = mask(x)
            p = layers.concatenate([pool_1, pool_2], axis=-1)
            p = layers.AveragePooling2D(pool_size=(pool_size, pool_size), strides=(2, 2), padding='same')(p)
            pools.append(p)
            x = layers.concatenate([s, p], axis=-1)
            x = layers.Conv2D(filters, (1, 1), padding="same", activation=None, kernel_initializer=init)(x)
            x = layers.add([s, x])
            x = layers.Activation(activation)(x)
        else:
            # Downsampling block with Depthwise convs.
            s = x
            d = layers.DepthwiseConv2D(depth_multiplier=1, kernel_size=(2, 2), strides=(2, 2), padding="same", activation=activation, kernel_initializer=init)(x)
            p = layers.AveragePooling2D(pool_size=(pool_size, pool_size), strides=(2, 2), padding='same')(pools[-1])
            pools.append(p)
            x = layers.concatenate([d, p], axis=-1)
            x = layers.Conv2D(filters, (1, 1), padding="same", activation=None, kernel_initializer=init)(x)
            b = layers.Conv2D(filters, (1, 1), strides=(2, 2), padding="same", activation=None, kernel_initializer=init)(s)
            x = layers.add([b, x])
            x = layers.Activation(activation)(x)
        
        # Information Bottleneck layer
    for _ in range(bottleneck_depth):
        x = resnetxt_block(x, filters=filters, cardinality=cardinality, init=init, activation=activation)
        
    # Decoder
    the_weights = None
    for i in range(depth-1, -1, -1):
        # Pooling Supervised Upsampling block
        x = layers.Concatenate(axis=-1)([x, pools[i]])
        x = layers.Conv2DTranspose(filters=filters, kernel_size=(2, 2), strides=(2, 2), padding='same', activation=activation, kernel_initializer=init)(x)
        # Channel weights
        if sup_level != 0:
            p = tf.reduce_sum(pools[i], axis=-1, keepdims=True)
            p = layers.AveragePooling2D(pool_size=(gr_level, gr_level), strides=(gr_level, gr_level), padding='same')(p)
            #p = tf.reduce_sum(p, axis=-1, keepdims=True)
            spatial_encoding_flattened = layers.Flatten()(p)
            weights = layers.Dense(int(compression_factor*filters), activation=activation, kernel_initializer=init)(spatial_encoding_flattened)
            weights = layers.Dense(filters, activation='sigmoid', kernel_initializer=init)(weights)
            weights_reshaped = layers.Reshape((1, 1, filters))(weights)
            the_weights = weights_reshaped
            sup_level = sup_level -1
        else:
            weights_reshaped = the_weights
        unifying_convs = layers.Conv2D(filters, (1, 1), padding='same', kernel_initializer=init, activation=activation)(x)
        x = tf.keras.layers.Multiply()([unifying_convs, weights_reshaped])
    
    # Output head 
    out = layers.Conv2D(1, kernel_size=(1, 1), strides=(1, 1), activation=None, padding='same', kernel_initializer=init)(x)
    
    summed_features = tf.reduce_sum(x, axis=-1, keepdims=True)

    out = layers.add([summed_features, out])
    out = layers.Activation('relu')(out)

    model = tf.keras.models.Model(inputs=input_grid, outputs=out)

    return model
  
######################################################################################################################################
############### CASPIAN END  #########################################################################################################
######################################################################################################################################



######################################################################################################################################
############### FOR ABLATION STUDIES #################################################################################################
######################################################################################################################################

def CASPIAN_beta(input_shape=(1024, 1024, 1), filters=81, cardinality=36, activation='tanh', init="glorot_normal",
                    input_centering=True, depth=4, bottleneck_depth=8, compression_factor=0.9, gr_level=4, sup_level=1, version=1):
    
    if version == 4:
      bottleneck_depth = 2
    
    input_grid = layers.Input(shape=input_shape)
    
    if input_centering:
        # Create a Lambda layer to replace 0.5 with -1
        replace_half_with_negative_one = layers.Lambda(
            lambda x: tf.where(tf.equal(x, 0.5), tf.constant(-1.0, dtype=x.dtype), x)
        )
        x = replace_half_with_negative_one(input_grid)
    else:
        x = input_grid

    # Encoder
    pools = []
    for i in range(depth):
        if i == 0:
            #Entry block
            pool_size=2
            s = layers.Conv2D(filters=filters, kernel_size=(2, 2), strides=(2, 2), padding='same', activation=activation, kernel_initializer=init)(x)
            if version != 3:
              #Stratified Pooling
              if input_centering:
                  mask =  layers.Lambda(
                      lambda x: tf.where(tf.equal(x, -1), x, tf.zeros_like(x))
                  )
              else:
                  mask =  layers.Lambda(
                      lambda x: tf.where(tf.equal(x, 0.5), x, tf.zeros_like(x))
                  )
              pool_1 = mask(x)
              mask =  layers.Lambda(
                      lambda x: tf.where(tf.equal(x, 1), x, tf.zeros_like(x))
              )
              pool_2 = mask(x)
              p = layers.concatenate([pool_1, pool_2], axis=-1)
              p = layers.AveragePooling2D(pool_size=(pool_size, pool_size), strides=(2, 2), padding='same')(p)
              pools.append(p)
              x = layers.concatenate([s, p], axis=-1)
            else:
              x = s
            x = layers.Conv2D(filters, (1, 1), padding="same", activation=None, kernel_initializer=init)(x)
            x = layers.add([s, x])
            x = layers.Activation(activation)(x)
        else:
            # Downsampling block with Depthwise convs.
            s = x
            d = layers.DepthwiseConv2D(depth_multiplier=1, kernel_size=(2, 2), strides=(2, 2), padding="same", activation=activation, kernel_initializer=init)(x)
            if version != 3:
              p = layers.AveragePooling2D(pool_size=(pool_size, pool_size), strides=(2, 2), padding='same')(pools[-1])
              pools.append(p)
              x = layers.concatenate([d, p], axis=-1)
            else:
              x = d
            x = layers.Conv2D(filters, (1, 1), padding="same", activation=None, kernel_initializer=init)(x)
            b = layers.Conv2D(filters, (1, 1), strides=(2, 2), padding="same", activation=None, kernel_initializer=init)(s)
            x = layers.add([b, x])
            x = layers.Activation(activation)(x)
        
        # Information Bottleneck layer
    for _ in range(bottleneck_depth):
        x = resnetxt_block(x, filters=filters, cardinality=cardinality, init=init, activation=activation)
        
    # Decoder
    the_weights = None
    for i in range(depth-1, -1, -1):
        # Pooling Supervised Upsampling block
        if version != 3:
          x = layers.Concatenate(axis=-1)([x, pools[i]])
        x = layers.Conv2DTranspose(filters=filters, kernel_size=(2, 2), strides=(2, 2), padding='same', activation=activation, kernel_initializer=init)(x)
        # Channel weights
        if version != 2 and version != 3:
          if sup_level != 0:
              p = tf.reduce_sum(pools[i], axis=-1, keepdims=True)
              p = layers.AveragePooling2D(pool_size=(gr_level, gr_level), strides=(gr_level, gr_level), padding='same')(p)
              #p = tf.reduce_sum(p, axis=-1, keepdims=True)
              spatial_encoding_flattened = layers.Flatten()(p)
              weights = layers.Dense(int(compression_factor*filters), activation=activation, kernel_initializer=init)(spatial_encoding_flattened)
              weights = layers.Dense(filters, activation='sigmoid', kernel_initializer=init)(weights)
              weights_reshaped = layers.Reshape((1, 1, filters))(weights)
              the_weights = weights_reshaped
              sup_level = sup_level -1
          else:
              weights_reshaped = the_weights
          unifying_convs = layers.Conv2D(filters, (1, 1), padding='same', kernel_initializer=init, activation=activation)(x)
          x = tf.keras.layers.Multiply()([unifying_convs, weights_reshaped])
        else:
          x = layers.Conv2D(filters, (1, 1), padding='same', kernel_initializer=init, activation=activation)(x)
    
    # Output head 
    out = layers.Conv2D(1, kernel_size=(1, 1), strides=(1, 1), activation=None, padding='same', kernel_initializer=init)(x)
    
    if version != 1:
      summed_features = tf.reduce_sum(x, axis=-1, keepdims=True)
      out = layers.add([summed_features, out])

    out = layers.Activation('relu')(out)

    model = tf.keras.models.Model(inputs=input_grid, outputs=out)

    return model

######################################################################################################################################
############### ABLATION STUDIES END##################################################################################################
######################################################################################################################################