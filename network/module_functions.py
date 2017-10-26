import theano
import theano.tensor as T
import lasagne

def make_functions(model):
        #Input Array
        X = T.TensorType('float32', [False]*5)('X')

        #Class Vector
        y = T.TensorType('int32', [False]*1)('y')

        #Output Layer
        l_out = model['l_out']
        y_hat_deterministic = lasagne.layers.get_output(l_out, X, deterministic=True)        
        softmax = T.nnet.softmax(y_hat_deterministic)
        pred_list_fn = theano.function([X], softmax)
        

        #Get ColorMap
        l_color = model['l_color']        
        
        color_map = lasagne.layers.get_output(l_color, X, deterministic=True)
        colorMap_fn = theano.function([X], color_map)

        return  pred_list_fn, colorMap_fn


def make_score_functions(model):

        #Input Array
        X = T.TensorType('float32', [False]*5)('X')
        l_out = model['l_out']

        y = lasagne.layers.get_output(l_out, X, deterministic=True)        
        score_volume_function = theano.function([X], y)

        return score_volume_function
