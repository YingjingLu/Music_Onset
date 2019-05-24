
import numpy as np 
import matplotlib.pyplot as plt 
import cmath
from scipy import signal
from scipy.io import wavfile
from generate_data import *

case = 7
def princarg( phase ):
    return phase % (2 * np.pi)
# phase change
def DFT_slow(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

def displace( Sxx, hop, freq, sample_rate ):

    theta = hop/ sample_rate * freq/2 * np.pi
    Sxx[ 1: ].real = Sxx[:-1].real - Sxx[ 1: ].real - np.sin( theta )
    Sxx[ 1: ].imag = Sxx[:-1].imag - Sxx[ 1: ].imag- np.cos( theta )

    return Sxx

def displace2( Sxx, hop, freq, sample_rate ):

    theta = hop/ sample_rate * freq/2 * np.pi
    Sxx = np.angle( Sxx )
    Sxx[1: ] = Sxx[ :-1 ] - Sxx[ 1: ] - theta
    return Sxx

# t * n
def d_gamma( Sxx ):
    r = amplitude( Sxx )
    res = np.copy( Sxx )
    angle = np.arctan2( Sxx.real, Sxx.imag )
    res[:2, :] = 0
    res[ 2:, : ] = np.square( r[ 1:-1, : ] ) + np.square( r[ 2:, : ] ) - 2 * r[ 1:-1, : ] * r[ 2:, : ] * np.cos( princarg( angle[ 2:, : ] - 2 * angle[ 1:-1, : ] + angle[ :-2, : ] ) )
    return np.sqrt( res )

def amplitude( Sxx ):
    res = np.square( Sxx.real ) + np.square( Sxx.imag )
    res = np.sqrt( res )
    return res

def easy( Sxx ):
    res = np.square( Sxx.real ) + np.square( Sxx.imag )
    res = np.sqrt( res )
    res[1:, :] = res[ :-1, : ] - res[ 1:, : ]
    return res

def orig( Sxx ):
    real = Sxx.real[1:,:] - Sxx.real[ :-1, : ]
    imag = Sxx.imag[ 1:, : ] - Sxx.imag[ :-1, : ]
    res = np.zeros( ( real.shape[ 0 ] + 1, real.shape[1] ) )
    res[ 1:, : ] = np.sqrt( np.square( real ) + np.square( imag ) )
    return res

# t * freq
def delta_phase( Sxx ):
    Sxx = np.angle( Sxx )
    delta = np.copy( Sxx )
    delta[ 1:, : ] = delta[ 1:, : ] - delta[ :-1, : ]
    target = np.copy( Sxx )
    target[ 2:, : ] = Sxx[ 1:-1, : ] + delta[ 1:-1, : ]
    return np.square( target - Sxx )

def delta_delta_phase( Sxx ):
    Sxx = delta_phase( Sxx )
    Sxx[ 1:, : ] = Sxx[ 1:, : ] - Sxx[ :-1, : ]
    return Sxx
    
def delta_delta_phase_with_magn( Sxx ):
    res = delta_delta_phase( Sxx )
    magn = np.sqrt( np.square( Sxx.real ) + np.square( Sxx.imag ) )
    return res * magn

def cartesian( Sxx ):
    angle = np.angle( Sxx )
    delta = np.copy( angle )
    delta[ 1:, : ] = delta[ 1:, : ] - delta[ :-1, : ]
    target = np.copy( angle )
    target[ 2:, : ] = angle[ 1:-1, : ] + delta[ 1:-1, : ]

    magnitude = np.sqrt( np.square( Sxx.real ) + np.square( Sxx.imag ) )
    target_real = np.cos( target ) * magnitude
    target_imag = np.sin( target ) * magnitude 

    diff = np.square( Sxx.real - target_real ) + np.square( Sxx.imag - target_imag )
    diff = np.sqrt( diff )

    diff[0, :] = 0
    return diff

def subplot( x, y, z ):
    num_sub = len( y )
    assert len( x ) == len( y ) or len( x ) == 1
    if len( x ) == 1:
        for i in range( num_sub ):
            plt.subplot( num_sub ,1, i+1 )
            plt.plot( x[ 0 ], y[ i ] )
            plt.ylabel( z[ i ] )
    else:
        for i in range( num_sub ):
            plt.subplot( num_sub,1, i+1 )
            plt.plot( x[ i ], y[ i ] )
            plt.ylabel( z[ i ] )

    plt.show()
# base case
if( case == 0 ):
    x = np.linspace( 0, 6, num = 1000 )
    y = x * 2*np.pi
    # y = x%( 2*np.pi ) + np.pi
    y = princarg( y )
    z = np.cos( x * 2 * np.pi )
    t = np.sin( x * 2 * np.pi  )
    plt.subplot( 3,1,1 )
    plt.plot( x, y )
    plt.ylabel( "phase" )
    plt.subplot( 3,1,2 )
    plt.plot( x, z ) 
    plt.ylabel( "cos wave" )
    plt.subplot( 3,1,3 )
    plt.plot( x, t ) 
    plt.ylabel( "sin wave" )
    plt.show()
elif( case == 1 ):
    x = np.linspace( 0, 6, num = 100 )

    y1 = np.cos( x * 2 * np.pi )
    y2 = np.cos( x * 0.8 * np.pi )
    y3 = np.cos( x * 3.3 * np.pi )
    y4 = y1+y2+y3
    subplot( [ x ], [ y1, y2, y3, y4 ] )

elif ( case == 2 ):
    x = np.linspace( 0, 6, num = 100 )
    y1 = np.sin( np.hstack( (np.zeros( 100 ), x) ) * 2 * np.pi )
    y4 = y1

    res = DFT_slow( y4 )
    print( res.shape )
    f = np.copy( res.imag ) 
    f[ 1: ] = f[:-1] - f[1:]

    subplot( [ np.linspace(0,6,200) ], [ y1, princarg( np.angle( res ) ), np.angle( res ) ] )

elif ( case == 3 ):
    x = np.linspace( 0, 6, num = 100 )
    y1 = np.sin(  x  * np.pi )
    z = 0.6*np.cos( x * 2 * np.pi )
    y4 = y1

    res = DFT_slow( y4 )
    print( res.shape )
    f = np.copy( res.imag ) 
    f[ 1: ] = f[:-1] - f[1:]

    subplot( [ x ], [ y4, princarg(np.angle( res )) ] )

elif( case == 4 ):
    fs, Audiodata = wavfile.read("../data/train/violin/single/mono_sound/violin_mono_0.wav")

    Audiodata = Audiodata[:,1]

    f, t, Sxx = signal.spectrogram(Audiodata, fs, window = signal.blackman(1024), return_onesided = True, nfft = 1024, noverlap = 768, mode = "complex" )
    print( f.shape )
    print( t.shape )
    print( Sxx.shape )
   
    # t * f
    Sxx = np.transpose( Sxx )
    print(  f[10] )
    res = Sxx[ :, 10 ]
    subplot( [ t ], [ res.real, res.imag, np.sum( amplitude( Sxx ), axis = 1 ), np.sum( orig( Sxx ), axis = 1 ), np.sum( easy( Sxx ), axis = 1), np.sum( d_gamma( Sxx ), axis = 1 )  ], [ "real", "imag","amp","orig", "amp-diff", "displace"] )

elif( case == 5 ):
    fs, Audiodata = wavfile.read("../data/train/piano/single/mono_sound/piano_mono_0.wav")

    Audiodata = Audiodata[:,1]

    f, t, Sxx = signal.spectrogram(Audiodata, fs, window = signal.blackman(1024), return_onesided = True, nfft = 1024, noverlap = 768, mode = "complex" )
    print( f.shape )
    print( t.shape )
    print( Sxx.shape )
   
    # t * f
    Sxx = np.transpose( Sxx )
    print(  f[10] )
    res = Sxx[ :, 10 ]
    subplot( [ t ], [ res.real, res.imag, np.sum( delta_phase( Sxx ), axis = 1 ), np.sum( delta_delta_phase( Sxx ), axis = 1 )  ], [ "real", "imag","delta", "delta delta phase"] )
    
elif( case == 6 ):
    fs, Audiodata = wavfile.read("../data/train/piano/single/mono_sound/piano_mono_0.wav")
    Audiodata = Audiodata[:,1]
    for i in range(1, 7 ):
        f, t, Sxx = signal.spectrogram(Audiodata, fs, window = signal.blackman(256*i), return_onesided = True, nfft = 256*i, noverlap = 64*i, mode = "complex" )
        # t * f
        Sxx = np.transpose( Sxx )
        subplot( [ t ], [ res.real, res.imag, np.sum( delta_phase( Sxx ), axis = 1 ), np.sum( delta_delta_phase( Sxx ), axis = 1 )  ], [ "real", "imag","delta", "delta delta phase"] )

elif( case == 7 ):
    fs, Audiodata = generate_sine( 44100, 1024, 4 )
    # Audiodata = np.hstack( ( np.zeros(44100), Audiodata ) )
    # fs, Audiodata = wavfile.read("../data/train/piano/single/mono_sound/piano_mono_0.wav")
    # Audiodata = Audiodata[:,1]
    f, t, Sxx = signal.spectrogram(Audiodata, fs, window = signal.blackman(1024), return_onesided = True, nfft = 1024, noverlap = 768, mode = "complex" )
    
    # t * f
    Sxx = np.transpose( Sxx )
    res = Sxx[ :, 3 ]
    # Sxx = Sxx[ :, :10 ]
    subplot( [ t ], [ res.real, res.imag, np.sum( cartesian( Sxx ), axis = 1 ), np.sum(delta_delta_phase(Sxx), axis = 1), np.sum( delta_delta_phase_with_magn( Sxx ), axis = 1 )  ], [ "real", "imag","cartesian", "delta delta phase" , "delta delta w/ Magnitude"] )
    






