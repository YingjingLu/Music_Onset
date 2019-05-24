import numpy as np 

# (t, bin)
def hfc( Sxx ):
    mgn = np.square( Sxx.real ) + np.square( Sxx.imag )
    mgn = np.sqrt( mgn )

    num_bin = mgn.shape[ 1 ]
    z = np.sum( np.arange( 1, num_bin+1, 1, dtype = np.float32 ) )
    for i in range( 1, num_bin + 1 ):
        mgn[ :, i - 1 ] = mgn[ :, i-1 ] * i / z 

    return mgn


