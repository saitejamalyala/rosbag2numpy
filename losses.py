from tensorflow.keras import backend as K

def euclidean_distance_loss(y_true, y_pred):
    """
    Euclidean distance loss
    https://en.wikipedia.org/wiki/Euclidean_distance
    :param y_true: TensorFlow tensor
    :param y_p red: TensorFlow tensor of the same shape as y_true
    :return: float
    """
    #original euclidean distance loss =  K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))
    #loss = K.mean(K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1)))
    loss =  K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

    return loss

def endpoint_loss(y_true, y_pred):
    loss = K.sqrt(K.sum(K.square(y_pred[-1,:] - y_true[-1,:])))
    
    return loss