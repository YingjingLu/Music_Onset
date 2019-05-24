import numpy as np 

m = np.load( "piano_N_1024_20_256_256_single_left_label.npy" )

print( np.exp(m[:10,:]) )