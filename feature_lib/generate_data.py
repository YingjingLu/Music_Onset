import numpy as np 
import matplotlib.pyplot as plt 
import math 

def generate_sine( sample_rate, window_size, _bin ):
    freq = sample_rate/window_size*_bin
    return freq, np.sin( np.linspace(0, freq * np.pi, sample_rate) )

def generate_cosine( sample_rate, window_size, _bin ):
    freq = sample_rate/window_size*_bin
    return freq, np.cos( np.linspace(0, freq * np.pi, sample_rate) )

# freq, y = generate_sine( 44100, 1024, 2 )
# plt.plot( list(range(len(y))), y )
# plt.show()