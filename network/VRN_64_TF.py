import tensorflow as tf


cfg = {
    'batch_size' : 16,       
    'decay_rate' : 0,
    'reg' : 0.001,
    'momentum' : 0.9,
    'dims' : (64, 64, 64),    
    'n_classes' : 2,
    'batches_per_chunk': 1,
    'max_epochs' : 250,
    'n_rotations' : 1,
}

def printLayer(layer):
    print("Output Shape : ", layer.shape)
    print("Number of Params : ", len(tf.trainable_variables()) )


def InceptionLayer(inputs,param_dict,block_name):
    branch = [0]*len(param_dict)

    # Loop across branches
    for i,dict in enumerate(param_dict):
        for j,style in enumerate(dict['style']):

            if j == 0:
                input_val = inputs
            else:
                input_val = branch[i]

            if style.__name__=='conv3d':
                branch[i] = style(
                    inputs = input_val,
                    filters = dict['num_filters'][j],
                    kernel_size = dict['filter_size'][j],
                    padding = dict['padding'][j],
                    strides = dict['strides'][j],                    
                    activation = dict['activation'][j],
                    name = block_name+'_'+str(i)+'_'+str(j)
                )  
            else:
                branch[i] = style(
                    inputs=input_val,
                    pool_size = dict['filter_size'][j],                    
                    strides = dict['strides'][j],
                    padding = dict['padding'][j],
                    name = block_name+'_'+str(i)+'_'+str(j)
                )                    
                
                #Apply Activation Function
                if not dict['activation'][j] == None:
                    branch[i] = dict['activation'][j](branch[i])
                    # branch[i] = NL(inputs=branch[i], activation=dict['activation'][j])
                
            # Apply Batchnorm
            if dict['bnorm'][j]:
                branch[i] = tf.layers.batch_normalization(branch[i],name = block_name+'_bnorm_'+str(i)+'_'+str(j))
        
    # Concatenate Sublayers
    return tf.concat(branch, axis=4)

def ResLayer(inputs, IB):
    return tf.nn.elu(tf.add(IB, inputs))

def ResDrop(inputs, IB, p):
    return tf.add(tf.layers.dropout(inputs=IB, rate=p), inputs)

def ResDropNoPre(inputs, IB, p):
    return tf.nn.elu(tf.add(tf.layers.dropout(inputs=IB, rate=p), inputs))
    

dims, n_classes = tuple(cfg['dims']), cfg['n_classes']
shape = (None,) + dims + (1,)


def get_model():
    keep_prob = tf.placeholder(dtype=tf.float32)
    l_in = tf.placeholder(dtype=tf.float32, shape=shape)
    l_conv0 = tf.layers.conv3d(
        inputs = l_in,
        filters = 32,
        kernel_size = (3,3,3),
        strides=(1, 1, 1),
        padding='same',
        name='l_conv0'
    )
    l_conv1 = ResDrop(
            inputs = l_conv0,
            IB = InceptionLayer(
                inputs = tf.nn.elu(tf.layers.batch_normalization(inputs=l_conv0, name='bn_conv0')),
                param_dict = [
                    {
                        'style': [tf.layers.conv3d]*3,
                        'num_filters':[8,8,16],
                        'filter_size':[(1,1,1),(3,3,3),(1,1,1)],
                        'padding':['same']*3,
                        'strides':[(1,1,1)]*3,
                        'activation': [tf.nn.elu,tf.nn.elu,None],                    
                        'bnorm':[1,1,0]
                    },
                    {   
                        'style': [tf.layers.conv3d]*2,
                        'num_filters':[8,16],
                        'filter_size':[(3,3,3)]*2,
                        'padding':['same']*2,
                        'strides':[(1,1,1)]*2,
                        'activation': [tf.nn.elu,None],                    
                        'bnorm':[1,0]
                    }],
                block_name = 'conv1'
            ),
            p=1.0-(1.0-keep_prob)*0.95)
    l_conv2 = ResDrop(
            inputs = l_conv1,
            IB = InceptionLayer(
                inputs = tf.nn.elu(tf.layers.batch_normalization(inputs=l_conv1, name='bn_conv1')),
                param_dict = [
                    {
                        'style': [tf.layers.conv3d]*3,
                        'num_filters':[8,8,16],
                        'filter_size':[(1,1,1),(3,3,3),(1,1,1)],
                        'padding':['same']*3,
                        'strides':[(1,1,1)]*3,
                        'activation': [tf.nn.elu,tf.nn.elu,None],                    
                        'bnorm':[1,1,0]
                    },
                    {   
                        'style': [tf.layers.conv3d]*2,
                        'num_filters':[8,16],
                        'filter_size':[(3,3,3)]*2,
                        'padding':['same']*2,
                        'strides':[(1,1,1)]*2,
                        'activation': [tf.nn.elu,None],                    
                        'bnorm':[1,0]
                    }],
                block_name = 'conv2'
            ),
            p=1.0-(1.0-keep_prob)*0.90)

    l_conv3 = ResDrop(
            inputs = l_conv2,
            IB = InceptionLayer(
                inputs = tf.nn.elu(tf.layers.batch_normalization(inputs=l_conv2, name='bn_conv2')),
                param_dict = [
                    {
                        'style': [tf.layers.conv3d]*3,
                        'num_filters':[8,8,16],
                        'filter_size':[(1,1,1),(3,3,3),(1,1,1)],
                        'padding':['same']*3,
                        'strides':[(1,1,1)]*3,
                        'activation': [tf.nn.elu,tf.nn.elu,None],                    
                        'bnorm':[1,1,0]
                    },
                    {   
                        'style': [tf.layers.conv3d]*2,
                        'num_filters':[8,16],
                        'filter_size':[(3,3,3)]*2,
                        'padding':['same']*2,
                        'strides':[(1,1,1)]*2,
                        'activation': [tf.nn.elu,None],                    
                        'bnorm':[1,0]
                    }],
                block_name = 'conv3'
            ),
            p=1.0-(1.0-keep_prob)*0.8)



    l_conv4 = InceptionLayer(
        inputs = tf.nn.elu(tf.layers.batch_normalization(inputs=l_conv3, name='bn_conv3')),
        param_dict = [
            {
                'style': [tf.layers.conv3d],
                'num_filters':[16],
                'filter_size':[(3,3,3)],
                'padding':['same'],
                'strides':[(2,2,2)],
                'activation': [None],      
                'bnorm':[1]
            },
            {
                'style': [tf.layers.conv3d],
                'num_filters':[16],
                'filter_size':[(1,1,1)],
                'padding':['valid'],
                'strides':[(2,2,2)],
                'activation': [None],                
                'bnorm':[1]
            },
            {
                'style': [tf.layers.conv3d,tf.layers.max_pooling3d],                
                'num_filters':[16,1],                
                'filter_size':[(3,3,3),(3,3,3)],
                'padding':['same','same'],
                'strides':[(1,1,1),(2,2,2)],
                'activation': [None,None],                
                'bnorm':[0,1]
            },
            {
                'style': [tf.layers.conv3d,tf.layers.average_pooling3d],                
                'num_filters':[16,1],                
                'filter_size':[(3,3,3),(3,3,3)],
                'padding':['same','same'],
                'strides':[(1,1,1),(2,2,2)],
                'activation': [None,None],                
                'bnorm':[0,1]
            }],
        block_name = 'conv4'
    )
    l_conv5 = ResDrop(
        inputs = l_conv4,
        IB = InceptionLayer(
            inputs = tf.nn.elu(tf.layers.batch_normalization(inputs=l_conv4, name='bn_conv4')),
            param_dict = [
                {
                    'style': [tf.layers.conv3d]*3,
                    'num_filters':[16,16,32],
                    'filter_size':[(1,1,1),(3,3,3),(1,1,1)],
                    'padding':['same']*3,
                    'strides':[(1,1,1)]*3,
                    'activation': [tf.nn.elu,tf.nn.elu,None],                    
                    'bnorm':[1,1,0]
                },
                {   
                    'style': [tf.layers.conv3d]*2,
                    'num_filters':[16,32],
                    'filter_size':[(3,3,3)]*2,
                    'padding':['same']*2,
                    'strides':[(1,1,1)]*2,
                    'activation': [tf.nn.elu,None],                    
                    'bnorm':[1,0]
                }],
            block_name = 'conv5'
        ),
        p=1.0-(1.0-keep_prob)*0.7
    )
    l_conv6 = ResDrop(
        inputs = l_conv5,
        IB = InceptionLayer(
            inputs = tf.nn.elu(tf.layers.batch_normalization(inputs=l_conv5, name='bn_conv5')),
            param_dict = [
                {
                    'style': [tf.layers.conv3d]*3,
                    'num_filters':[16,16,32],
                    'filter_size':[(1,1,1),(3,3,3),(1,1,1)],
                    'padding':['same']*3,
                    'strides':[(1,1,1)]*3,
                    'activation': [tf.nn.elu,tf.nn.elu,None],                    
                    'bnorm':[1,1,0]
                },
                {   
                    'style': [tf.layers.conv3d]*2,
                    'num_filters':[16,32],
                    'filter_size':[(3,3,3)]*2,
                    'padding':['same']*2,
                    'strides':[(1,1,1)]*2,
                    'activation': [tf.nn.elu,None],                    
                    'bnorm':[1,0]
                }],
            block_name = 'conv6'
        ),
        p=1.0-(1.0-keep_prob)*0.6
    )
    l_conv7 = ResDrop(
        inputs = l_conv6,
        IB = InceptionLayer(
            inputs = tf.nn.elu(tf.layers.batch_normalization(l_conv6,name='bn_conv6')),
            param_dict = [
                {
                    'style': [tf.layers.conv3d]*3,
                    'num_filters':[16,16,32],
                    'filter_size':[(1,1,1),(3,3,3),(1,1,1)],
                    'padding':['same']*3,
                    'strides':[(1,1,1)]*3,
                    'activation': [tf.nn.elu,tf.nn.elu,None],                    
                    'bnorm':[1,1,0]
                },
                {   
                    'style': [tf.layers.conv3d]*2,
                    'num_filters':[16,32],
                    'filter_size':[(3,3,3)]*2,
                    'padding':['same']*2,
                    'strides':[(1,1,1)]*2,
                    'activation': [tf.nn.elu,None],                    
                    'bnorm':[1,0]
                }],
            block_name = 'conv7'
        ),
        p=1.0-(1.0-keep_prob)*0.5
    )
    l_conv8 = InceptionLayer(
        inputs = tf.nn.elu(tf.layers.batch_normalization(l_conv7,name='bn_conv7')),
        param_dict = [
            {
                'style': [tf.layers.conv3d],
                'num_filters':[32],
                'filter_size':[(3,3,3)],
                'padding':['same'],
                'strides':[(2,2,2)],
                'activation': [None],      
                'bnorm':[1]
            },
            {
                'style': [tf.layers.conv3d],
                'num_filters':[32],
                'filter_size':[(1,1,1)],
                'padding':['valid'],
                'strides':[(2,2,2)],
                'activation': [None],                
                'bnorm':[1]
            },
            {
                'style': [tf.layers.conv3d,tf.layers.max_pooling3d],                
                'num_filters':[32,1],                
                'filter_size':[(3,3,3),(3,3,3)],
                'padding':['same','same'],
                'strides':[(1,1,1),(2,2,2)],
                'activation': [None,None],                
                'bnorm':[0,1]
            },
            {
                'style': [tf.layers.conv3d,tf.layers.average_pooling3d],                
                'num_filters':[32,1],                
                'filter_size':[(3,3,3),(3,3,3)],
                'padding':['same','same'],
                'strides':[(1,1,1),(2,2,2)],
                'activation': [None,None],                
                'bnorm':[0,1]
            }],
        block_name = 'conv8'
    )
    l_conv9 = ResDrop(
        inputs = l_conv8,
        IB = InceptionLayer(
            inputs = tf.nn.elu(tf.layers.batch_normalization(l_conv8,name='bn_conv8')),
            param_dict = [
                {
                    'style': [tf.layers.conv3d]*3,
                    'num_filters':[32,32,64],
                    'filter_size':[(1,1,1),(3,3,3),(1,1,1)],
                    'padding':['same']*3,
                    'strides':[(1,1,1)]*3,
                    'activation': [tf.nn.elu,tf.nn.elu,None],                    
                    'bnorm':[1,1,0]
                },
                {   
                    'style': [tf.layers.conv3d]*2,
                    'num_filters':[32,64],
                    'filter_size':[(3,3,3)]*2,
                    'padding':['same']*2,
                    'strides':[(1,1,1)]*2,
                    'activation': [tf.nn.elu,None],                    
                    'bnorm':[1,0]
                }],
            block_name = 'conv9'
        ),
        p=1.0-(1.0-keep_prob)*0.5
    )
    l_conv10 = ResDrop(
        inputs = l_conv9,
        IB = InceptionLayer(
            inputs = tf.nn.elu(tf.layers.batch_normalization(l_conv9,name='bn_conv9')),
            param_dict = [
                {
                    'style': [tf.layers.conv3d]*3,
                    'num_filters':[32,32,64],
                    'filter_size':[(1,1,1),(3,3,3),(1,1,1)],
                    'padding':['same']*3,
                    'strides':[(1,1,1)]*3,
                    'activation': [tf.nn.elu,tf.nn.elu,None],                    
                    'bnorm':[1,1,0]
                },
                {   
                    'style': [tf.layers.conv3d]*2,
                    'num_filters':[32,64],
                    'filter_size':[(3,3,3)]*2,
                    'padding':['same']*2,
                    'strides':[(1,1,1)]*2,
                    'activation': [tf.nn.elu,None],                    
                    'bnorm':[1,0]
                }],
            block_name = 'conv10'
        ),
        p=1.0-(1.0-keep_prob)*0.45
    )
    l_conv11 = ResDrop(
        inputs = l_conv10,
        IB = InceptionLayer(
            inputs = tf.nn.elu(tf.layers.batch_normalization(l_conv10,name='bn_conv10')),
            param_dict = [
                {
                    'style': [tf.layers.conv3d]*3,
                    'num_filters':[32,32,64],
                    'filter_size':[(1,1,1),(3,3,3),(1,1,1)],
                    'padding':['same']*3,
                    'strides':[(1,1,1)]*3,
                    'activation': [tf.nn.elu,tf.nn.elu,None],                    
                    'bnorm':[1,1,0]
                },
                {   
                    'style': [tf.layers.conv3d]*2,
                    'num_filters':[32,64],
                    'filter_size':[(3,3,3)]*2,
                    'padding':['same']*2,
                    'strides':[(1,1,1)]*2,
                    'activation': [tf.nn.elu,None],                    
                    'bnorm':[1,0]
                }],
            block_name = 'conv11'
        ),
        p=1.0-(1.0-keep_prob)*0.40
    )
    l_conv12 = InceptionLayer(
        inputs = tf.nn.elu(tf.layers.batch_normalization(l_conv11,name='bn_conv11')),
        param_dict = [
            {
                'style': [tf.layers.conv3d],
                'num_filters':[64],
                'filter_size':[(3,3,3)],
                'padding':['same'],
                'strides':[(2,2,2)],
                'activation': [None],      
                'bnorm':[1]
            },
            {
                'style': [tf.layers.conv3d],
                'num_filters':[64],
                'filter_size':[(1,1,1)],
                'padding':['valid'],
                'strides':[(2,2,2)],
                'activation': [None],                
                'bnorm':[1]
            },
            {
                'style': [tf.layers.conv3d,tf.layers.max_pooling3d],                
                'num_filters':[64,1],                
                'filter_size':[(3,3,3),(3,3,3)],
                'padding':['same','same'],
                'strides':[(1,1,1),(2,2,2)],
                'activation': [None,None],                
                'bnorm':[0,1]
            },
            {
                'style': [tf.layers.conv3d,tf.layers.average_pooling3d],                
                'num_filters':[64,1],                
                'filter_size':[(3,3,3),(3,3,3)],
                'padding':['same','same'],
                'strides':[(1,1,1),(2,2,2)],
                'activation': [None,None],                
                'bnorm':[0,1]
            }],
        block_name = 'conv12'
    )
    l_conv13 = ResDrop(
        inputs = l_conv12,
        IB = InceptionLayer(
            inputs = tf.nn.elu(tf.layers.batch_normalization(l_conv12,name='bn_conv12')),
            param_dict = [
                {
                    'style': [tf.layers.conv3d]*3,
                    'num_filters':[64,64,128],
                    'filter_size':[(1,1,1),(3,3,3),(1,1,1)],
                    'padding':['same']*3,
                    'strides':[(1,1,1)]*3,
                    'activation': [tf.nn.elu,tf.nn.elu,None],                    
                    'bnorm':[1,1,0]
                },
                {   
                    'style': [tf.layers.conv3d]*2,
                    'num_filters':[64,128],
                    'filter_size':[(3,3,3)]*2,
                    'padding':['same']*2,
                    'strides':[(1,1,1)]*2,
                    'activation': [tf.nn.elu,None],                    
                    'bnorm':[1,0]
                }],
            block_name = 'conv13'
        ),
        p=1.0-(1.0-keep_prob)*0.35
    )
    l_conv14 = ResDrop(
        inputs = l_conv13,
        IB = InceptionLayer(
            inputs = tf.nn.elu(tf.layers.batch_normalization(l_conv13,name='bn_conv13')),
            param_dict = [
                {
                    'style': [tf.layers.conv3d]*3,
                    'num_filters':[64,64,128],
                    'filter_size':[(1,1,1),(3,3,3),(1,1,1)],
                    'padding':['same']*3,
                    'strides':[(1,1,1)]*3,
                    'activation': [tf.nn.elu,tf.nn.elu,None],                    
                    'bnorm':[1,1,0]
                },
                {   
                    'style': [tf.layers.conv3d]*2,
                    'num_filters':[64,128],
                    'filter_size':[(3,3,3)]*2,
                    'padding':['same']*2,
                    'strides':[(1,1,1)]*2,
                    'activation': [tf.nn.elu,None],                    
                    'bnorm':[1,0]
                }],
            block_name = 'conv14'
        ),
        p=1.0-(1.0-keep_prob)*0.30
    )
    l_conv15 = ResDrop(
        inputs = l_conv14,
        IB = InceptionLayer(
            inputs = tf.nn.elu(tf.layers.batch_normalization(l_conv14,name='bn_conv14')),
            param_dict = [
                {
                    'style': [tf.layers.conv3d]*3,
                    'num_filters':[64,64,128],
                    'filter_size':[(1,1,1),(3,3,3),(1,1,1)],
                    'padding':['same']*3,
                    'strides':[(1,1,1)]*3,
                    'activation': [tf.nn.elu,tf.nn.elu,None],                    
                    'bnorm':[1,1,0]
                },
                {   
                    'style': [tf.layers.conv3d]*2,
                    'num_filters':[64,128],
                    'filter_size':[(3,3,3)]*2,
                    'padding':['same']*2,
                    'strides':[(1,1,1)]*2,
                    'activation': [tf.nn.elu,None],                    
                    'bnorm':[1,0]
                }],
            block_name = 'conv15'
        ),
        p=1.0-(1.0-keep_prob)*0.25
    )
    l_conv16 = InceptionLayer(
        inputs = tf.nn.elu(tf.layers.batch_normalization(l_conv15,name='bn_conv15')),
        param_dict = [
            {
                'style': [tf.layers.conv3d],
                'num_filters':[128],
                'filter_size':[(3,3,3)],
                'padding':['same'],
                'strides':[(2,2,2)],
                'activation': [None],      
                'bnorm':[1]
            },
            {
                'style': [tf.layers.conv3d],
                'num_filters':[128],
                'filter_size':[(1,1,1)],
                'padding':['valid'],
                'strides':[(2,2,2)],
                'activation': [None],                
                'bnorm':[1]
            },
            {
                'style': [tf.layers.conv3d,tf.layers.max_pooling3d],                
                'num_filters':[128,1],                
                'filter_size':[(3,3,3),(3,3,3)],
                'padding':['same','same'],
                'strides':[(1,1,1),(2,2,2)],
                'activation': [None,None],                
                'bnorm':[0,1]
            },
            {
                'style': [tf.layers.conv3d,tf.layers.average_pooling3d],                
                'num_filters':[128,1],                
                'filter_size':[(3,3,3),(3,3,3)],
                'padding':['same','same'],
                'strides':[(1,1,1),(2,2,2)],
                'activation': [None,None],                
                'bnorm':[0,1]
            }],
        block_name = 'conv16'
    )
    l_conv17 = ResDropNoPre(l_conv16,tf.layers.batch_normalization(tf.layers.conv3d(
        inputs = l_conv16,
        filters = 512,
        kernel_size = (3,3,3),
        padding = 'same',
        # kernel_initializer=tf.nn.relu,
        activation = None,
        name = 'l_conv17'),name='bn_conv17'),0.5)

    # print("conv17 shape : ", l_conv17.shape)


    #Global Average Pooling : Pool_size = Input Volume Dimensions    
    l_pool = tf.layers.batch_normalization(tf.layers.average_pooling3d(
        inputs = l_conv17, pool_size=(4,4,4), strides=(1,1,1)),name='l_pool')

    l_fc = tf.layers.conv3d(
        inputs=l_pool,
        filters=n_classes,0.90
        kernel_size=(1,1,1),
        name='fc'
    )

    
    return l_in, l_fc, keep_prob

