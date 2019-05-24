from scipy.io import wavfile # scipy library to read wav files
import numpy as np
import math 
from scipy import signal as signal
import matplotlib.pyplot as plt
import os
import argparse
import scipy
import soundfile as psf
import matplotlib.pyplot as plt

from config.tri import *

SAMPLE_RATE = 48000
N = 1024
FFT_WINDOW_SIZE = 20
HOP_SIZE = 128
TRUNCATED_FREQ = 300
IS_LOG = True
TMP = 800.

INSTRUMENT_MAPPING = {
    "vn": 0,
    "va": 1,
    "vc": 2,
    "db": 3,
    "fl": 4,
    "ob": 5,
    "cl": 6,
    "sax": 7,
    "bn": 8,
    "tpt": 9,
    "hn": 10,
    "tbn": 11,
    "tba": 12
}



##########################  helper Functions #################################
def plot( s ):
    plt.plot( np.arange(len(s)), s )
    plt.show()

def imfig( s ):
    plt.imshow( s )
    plt.show()
# take in onset time ( in sec ) convert to the frame position, indexed by 0
def note_onset_time_to_sample_pos( onset ):
    global SAMPLE_RATE, N, HOP_SIZE
    sample_pos = onset * SAMPLE_RATE
    if sample_pos < N:
        return 0.0
    else:
        return ( sample_pos - N ) / ( HOP_SIZE )
    
def label_pos_float_to_int( pos ):
    return math.ceil( pos )

def label_text_to_np( path ):
    with open( path, "r" ) as f:
        l = f.readline()
        if l == "":
            EOFError()
        else:
            res = []
            while( l != "" ):
                onset = float( l.strip().split()[ 0 ] )
                res.append( note_onset_time_to_sample_pos( onset ) )
                l = f.readline()
            return res 

def log_gaussian_label( raw_label, num_sample ):
    global HOP_SIZE, TMP
    onset_frame = raw_label * HOP_SIZE + float(  N - N // 2 )
    # print("onset frame", onset_frame)
    exp_prob = np.arange( num_sample, dtype = np.float )

    mid_sample = exp_prob * float( HOP_SIZE ) + float(  N - N // 2 )
    # plot( -1.0 * np.abs( mid_sample - onset_frame ) / TMP )
    exp_prob = -1.0*np.abs( mid_sample - onset_frame ) / TMP
    # plot( exp_prob )
    return exp_prob

def wav_to_fft( path ):
    global TRUNCATED_FREQ, N, HOP_SIZE
    data, samplerate = psf.read( path )
    new_path = path.split(".")[0] + "_16" + ".wav"
    psf.write( new_path, data, samplerate, subtype = "PCM_16" )
    window = signal.blackman( N )
    noverlap = N - HOP_SIZE
    fs, audio_data = wavfile.read( new_path )
    f, t, Sxx = signal.spectrogram( audio_data, fs, window = window,
                                    noverlap = noverlap, nfft = N, nperseg = N )
    Sxx = Sxx[ :TRUNCATED_FREQ, : ]
    Sxx = np.transpose( 10*np.log10(Sxx + 1e-8 ) )
    return f, t, Sxx

########################### End Helper Functions #############################

def raw_label_to_actual_label( raw_label, num_FFT ):

    num_label = len( raw_label )
    num_sample = num_FFT
    label_table = []
    # print( num_label, num_sample )
    for onset_frame in raw_label:
        label_table.append( log_gaussian_label( onset_frame, num_FFT ).reshape( 1, -1 ) )
    
    label_table = np.concatenate( label_table, axis = 0 )
    # print(label_table.shape)
    res = np.amax( label_table, axis = 0 )
    # plot( res )
    return res

def make_data( sample, real_label, raw_label, spread = 30 ):

    res_sample = []
    res_label = []
    max_sample = sample.shape[ 0 ]
    for i in range( len( raw_label ) ):
        if math.ceil( raw_label[ i ] ) - raw_label[ i ] < 0.5:
            raw_label[ i ] = math.ceil( raw_label[ i ] )
        else:
            raw_label[ i ] = math.floor( raw_label[ i ] )
    # print( "raw label", raw_label )
    start_index = 0
    end = max_sample
    for raw_onset in raw_label:
        
        int_sample_index = label_pos_float_to_int( raw_onset )
        # print( "label index", int_sample_index )
        start_index = max( start_index, int_sample_index - spread )
        end = min( max_sample, int_sample_index + spread )
        # print( "start", start_index )
        # print( "end", end )
        if( ( end - start_index ) < FFT_WINDOW_SIZE ):
            continue
        for i in range( start_index, end - FFT_WINDOW_SIZE ):
            res_sample.append( sample[ i: i + FFT_WINDOW_SIZE ].reshape( 1, TRUNCATED_FREQ, FFT_WINDOW_SIZE, 1 ) )
            res_label.append( real_label[ i: i + FFT_WINDOW_SIZE ] )
            # imfig( sample[ i: i + FFT_WINDOW_SIZE ] )
            # plot( real_label[ i: i + FFT_WINDOW_SIZE ] )
        start_index = max( 0, end - FFT_WINDOW_SIZE )
        end = min( max_sample, start_index + 2 * spread )
    return np.vstack( res_sample ), np.vstack( res_label )

def wrapper():
    assert len( SAMPLE_LIST ) == len( LABEL_LIST )
    # ( *,  window_size, freq bin)
    sample = []
    #( *, window_size )
    label = []
    for i in range( len( SAMPLE_LIST ) ):
        sample_path = SAMPLE_LIST[ i ]
        
        # label_path = sample_path.split(".")[ 0 ] + ".txt"
        f, t, Sxx = wav_to_fft( sample_path )
        label_path = LABEL_LIST[ i ]
        raw_label = []
        for path in label_path:
            raw_label += label_text_to_np( path )
        raw_label.sort()
        real_label = raw_label_to_actual_label( raw_label, Sxx.shape[ 0 ] )
        print(real_label.shape)
        s, l = make_data( Sxx, real_label, raw_label )

        sample.append( s )
        label.append( l )
        print( "Finished", sample_path )
    sample = np.concatenate( sample, axis = 0 )
    np.save( BASE_PATH + "_sample.npy", sample )
    np.save( BASE_PATH + "_label.npy", np.concatenate( label, axis = 0 ) )
    print("Sample size", sample.shape)



################################## Main Args #################################

if __name__ == "__main__":
    wrapper()