# zero crossing function for onset detection
# The zero-crossing rate is the rate of sign-changes along a signal, i.e., the rate at which the signal changes 
# from positive to zero to negative or 
# from negative to zero to positive.
# reference https://en.wikipedia.org/wiki/Zero-crossing_rate

import numpy as np 

def zero_crossing( signal, window_size, noverlap, thresh = None ):
    """
    Compute the zero crossing rate of each window in the signal matrix
    discard tailing frames

    Args:
        signal: matrix( num_sample, )

    """
    res = []
    for i in range( 0, signal.shape[ 0 ], window_size - noverlap ):
        res.append( np.sum( np.diff( np.sign( signal[ i: i + window_size ] ) ) ) )[ 0 ]
    return np.array( res )

def zero_crossing_rate( signal, window_size, noverlap ):
    return zero_crossing( signal, window_size, noverlap ) / window_size