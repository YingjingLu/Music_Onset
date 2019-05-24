import numpy as np 
import math 

def eval_log_prob_syn_one( true_f, pred_f, true_thresh = 0.6, num_window_accuracy = 5, is_log = True, regu = 1.0 ):
    """
    takes in the true predict logn prob and true log prob judge whether the prediction is accurate enough.
    Two criteria:
        1. there is an onset if at least one value of the pred log is above true thresh within prediction window
        2. The onset is labeled as the window with peak log prob, if the pred peak is less than num_window_accuracy away
            from true peak, then the onset prediction is considered correct.
        Args:
            true_f - str.npy: the path where true labels are located
            pred_f - str.npy: the path where pred labels are located
            true_thresh - float: the threshold where a window log prob can be considered as having onset 
            num_window_accuracy - int: number of windows max for a 
    """

    true = np.load( true_f )* regu
    if is_log:
        true =  np.exp( true ) 
    pred = np.load( pred_f )* regu
    if is_log:
        pred =  np.exp(pred )
    assert true.shape == pred.shape
    num_sample, window_size = true.shape

    # filter out if there is an onset
    true_raw = ( true >= true_thresh  ) # n * window_size
    true_is_onset = np.any( true_raw, axis = 1 ).flatten() # ( n, )
    print( "true_is_onset", true_is_onset[:20] )

    pred_raw = ( pred >= true_thresh  ) # n * window_size
    pred_is_onset = np.any( pred_raw, axis = 1 ).flatten() # ( n, )
    print( "pred_is_onset", pred_is_onset[:20] )

    true_true, true_false, false_true, false_false = 0,0,0,0
    get_all_true, get_part_true, get_all_false, get_part_false = 0, 0, 0, 0
    for i in range( num_sample ):
        if true_is_onset[ i ]:
            if pred_is_onset[ i ]:
                true_true += 1
            else:
                true_false += 1
            
            if np.array_equal( true_raw[i], pred_raw[ i ] ):
                get_all_true += 1
            else:
                get_part_true += 1
            
            
        else:
            if pred_is_onset[ i ]:
                false_true += 1
            else:
                false_false += 1
            
            if np.array_equal( true_raw[ i ], pred_raw[ i ] ):
                get_all_false += 1
            else:
                get_part_false += 1
    print( "Results --------------------" )
    print( true_true, true_false, false_true, false_false )
    print( get_all_true, get_part_true, get_all_false, get_part_false )
    print("End Results")

    # figure out where the onset is
    true_loc = np.argmax( true, axis = 1 ).flatten()
    pred_loc = np.argmax( pred, axis = 1 ).flatten()

    total_right = 0
    within_4 = 0
    total_false = 0
    distance = np.abs( true_loc - pred_loc )
    all_dist = []
    for i in range( num_sample ):
        # check if it is a sample with onset:
        if true_is_onset[ i ] == 1:
            all_dist.append( distance[ i ] )
            if true_loc[ i ] == pred_loc[ i ]:
                total_right += 1
            if distance[ i ] <= 4:
                within_4 += 1
    print( total_right, within_4 )
    all_dist = np.array( all_dist )
    print( np.mean( all_dist ), np.std( all_dist ) )

