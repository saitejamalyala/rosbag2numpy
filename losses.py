from tensorflow.keras import backend as K
import tensorflow as tf

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

def endpoint_loss(y_true, y_pred): # or final displacement error
    loss = K.sqrt(K.sum(K.square(y_pred[-1,:] - y_true[-1,:])))
    
    return loss

def costmap_loss_wrapper(costmap):

    costmap = K.squeeze(costmap,axis=len(costmap.shape)-1)

    def costmap_loss(y_true, y_pred):
        """
        Costmap loss
        https://en.wikipedia.org/wiki/Cost_map
        :param costmap: TensorFlow tensor
        :param y_true: TensorFlow tensor
        :param y_pred: TensorFlow tensor
        :return: float

        init_diff_sqr = K.square(gridmap_idx - tf_init_path[5])
        init_dist = K.sqrt(K.sum(init_diff_sqr, axis=1))
        init_inv_dist = tf.math.reciprocal_no_nan(tf.math.pow(init_dist,0.1))
        """
        valid_indices = tf.where(costmap > 0.35)
        valid_costs = tf.gather_nd(costmap, valid_indices)
        valid_costs = tf.cast(valid_costs, dtype=tf.float32)

        allcosts = tf.constant(0.0)
        valid_indices = tf.cast(valid_indices, dtype=tf.float32)

        for i in range(y_pred.shape[0]):
            #distance to all valid indices
            pred_dist  = K.sqrt(K.sum(K.square(valid_indices - y_pred[i]),axis=1))
            inv_pred_dist = tf.math.reciprocal_no_nan(tf.math.pow(pred_dist,0.1))

            # cost for one point in the path
            cost_for_point = tf.reduce_sum(tf.multiply(inv_pred_dist,valid_costs))
            pred_allcosts = tf.add(allcosts,cost_for_point) if i == 0 else tf.add(pred_allcosts,cost_for_point)

        for j in range(y_true.shape[0]):
            #distance to all valid indices
            true_dist  = K.sqrt(K.sum(K.square(valid_indices - y_true[i]),axis=1))
            inv_true_dist = tf.math.reciprocal_no_nan(tf.math.pow(true_dist,0.1))

            # cost for one point in the path
            cost_for_point = tf.reduce_sum(tf.multiply(inv_true_dist,valid_costs))
            true_allcosts = tf.add(allcosts,cost_for_point) if j == 0 else tf.add(true_allcosts,cost_for_point)
            
        costmap_loss = tf.abs(tf.subtract(pred_allcosts,true_allcosts))
        return costmap_loss

    return costmap_loss
#code to train a neural network to predict the next point in a trajectory in tensorflow

