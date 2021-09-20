from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.losses import mean_squared_error

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
    
    loss =   K.mean(K.sqrt(K.sum(K.square(y_pred - y_true), axis=1)))
    loss = loss + K.mean(K.sqrt(K.sum(K.square(y_pred[-1,:] - y_true[-1,:]),axis=1)))
    loss += K.mean(K.sqrt(K.sum(K.square(y_pred[0,:] - y_true[0,:]),axis=1)))

    return loss

def enc_costmap_loss(y_true, y_pred):
    #prediction = y_pred[:,:,0:2]
    print(y_pred.shape)
    print(y_pred[:,25:,:].shape)
    print(y_true.shape)

    #enc_costmap = y_pred[:,2:]
    #print(tf.shape(enc_costmap))
    d_loss = 1.0#y_pred[:,:,0:2]-y_true
    #d_loss =  K.mean(K.sqrt(K.sum(K.square(prediction - y_true), axis=1)))
    #d_loss += K.mean(K.sqrt(K.sum(K.square(prediction[-1,:] - y_true[-1,:]), axis=1)))
    #d_loss += K.mean(K.sqrt(K.sum(K.square(prediction[0,:] - y_true[0,:]),axis=1)))

    #costmap_loss = K.sum(K.flatten(K.abs(enc_costmap*K.abs(prediction - y_true))))

    return d_loss #+ costmap_loss


def endpoint_loss(y_true, y_pred): # or final displacement error
    loss = K.mean(K.sqrt(K.sum(K.square(y_pred[-1,:] - y_true[-1,:]))))
    
    return loss

def costmap_loss_wrapper(costmap):

    #costmap = K.squeeze(costmap,axis=len(costmap.shape)-1)

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
        print(f"shape of y pred :{y_pred}")
        valid_indices = tf.where(costmap > 0.35) # get indices
        valid_costs = tf.gather_nd(costmap, valid_indices) # get respective values
        valid_costs = tf.cast(valid_costs, dtype=tf.float32) #cast them to float32

        allcosts = tf.constant(0.0)
        valid_indices = tf.cast(valid_indices, dtype=tf.float32)
        print(f"valid_costs:{valid_costs}")

        for i in range(25):
            #distance to all valid indices
            #print(f"valid indices:{valid_indices[:,1:]}")
            pred_dist  = K.sqrt(K.sum(K.square(valid_indices[:,1:] - y_pred[i]),axis=1))
            inv_pred_dist = tf.math.reciprocal_no_nan(tf.math.pow(pred_dist,0.1))

            # cost for one point in the path
            cost_for_point = tf.reduce_sum(tf.multiply(inv_pred_dist,valid_costs))
            pred_allcosts = tf.add(allcosts,cost_for_point) if i == 0 else tf.add(pred_allcosts,cost_for_point)

        for j in range(25):
            #distance to all valid indices
            true_dist  = K.sqrt(K.sum(K.square(valid_indices[:,1:] - y_true[i]),axis=1))
            inv_true_dist = tf.math.reciprocal_no_nan(tf.math.pow(true_dist,0.1))

            # cost for one point in the path
            cost_for_point = tf.reduce_sum(tf.multiply(inv_true_dist,valid_costs))
            true_allcosts = tf.add(allcosts,cost_for_point) if j == 0 else tf.add(true_allcosts,cost_for_point)
            
        relative_cost_loss = tf.abs(tf.subtract(pred_allcosts,true_allcosts))
        
        return relative_cost_loss
    
    return costmap_loss
#code to train a neural network to predict the next point in a trajectory in tensorflow

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
    print(f"shape of y true :{y_true}")
    costmap = y_true # costmap from groundtruth
    valid_indices = tf.where(costmap > 0.35) # get indices
    valid_costs = tf.gather_nd(costmap, valid_indices) # get respective values
    valid_costs = tf.cast(valid_costs, dtype=tf.float32) #cast them to float32

    allcosts = tf.constant(0.0)
    valid_indices = tf.cast(valid_indices, dtype=tf.float32)
    print(f"valid_costs:{valid_costs}")

    for i in range(25):
        #distance to all valid indices
        #print(f"valid indices:{valid_indices[:,1:]}")
        pred_dist  = K.sqrt(K.sum(K.square(valid_indices[:,1:] - y_pred[i]),axis=1))
        inv_pred_dist = tf.math.reciprocal_no_nan(tf.math.pow(pred_dist,0.1))

        # cost for one point in the path
        cost_for_point = tf.reduce_sum(tf.multiply(inv_pred_dist,valid_costs))
        pred_allcosts = tf.add(allcosts,cost_for_point) if i == 0 else tf.add(pred_allcosts,cost_for_point)

    for j in range(25):
        #distance to all valid indices
        path_gt=y_true[1] # path from groundttruth
        true_dist  = K.sqrt(K.sum(K.square(valid_indices[:,1:] - path_gt[i]),axis=1))
        inv_true_dist = tf.math.reciprocal_no_nan(tf.math.pow(true_dist,0.1))

        # cost for one point in the path
        cost_for_point = tf.reduce_sum(tf.multiply(inv_true_dist,valid_costs))
        true_allcosts = tf.add(allcosts,cost_for_point) if j == 0 else tf.add(true_allcosts,cost_for_point)
        
    relative_cost_loss = tf.abs(tf.subtract(pred_allcosts,true_allcosts))
    
    return relative_cost_loss


def loss_wrapper(cost_gt_opt):

    def pred_cost(y_true, y_pred):
        
        return K.sum(K.flatten(cost_gt_opt * K.abs(y_true - y_pred)))
    return pred_cost
        

def distance_loss(y_true, y_pred):
    print(f"y_true:{y_true}")
    print(f"y_pred:{y_pred}")

    return K.mean(K.abs(y_true-y_pred))