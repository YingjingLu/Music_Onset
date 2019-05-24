from scipy.io import wavfile # scipy library to read wav files
import numpy as np
import math 
from scipy import signal
import matplotlib.pyplot as plt
import os
import argparse
import scipy

def create_sample_label( audio_data, fs, window, noverlap, N, truncate_freq, label ):
    # print( "noverlap", noverlap, "N", N )
    f, t, Sxx = signal.spectrogram( audio_data, fs, window = window,
                                    noverlap = noverlap, nfft = N, nperseg = N )
    Sxx = Sxx[ :truncate_freq, : ]
    Sxx = np.transpose( 10*np.log10(Sxx + 1e-8 ) )

    label_pos = np.where( label == 1 )[ 0 ]
    return f, t, Sxx, label_pos


def make_sample_label_gaussian( Sxx, label_pos, N, noverlap, truncate_freq, window_size, std = 400., is_log = 1, regularize_label = 0 ):
    # print( "Sxx", Sxx.shape )
    single_sample, single_label = [], []

    label_interval_index = math.ceil( ( ( label_pos + 1 ) - N ) / ( N - noverlap ) )

    for i in range( window_size ):
        ahead = window_size - i
        start_idx = label_interval_index - ahead + 1
        end_idx = label_interval_index - ahead + 1 + window_size
        sample = Sxx[ start_idx : end_idx , :].reshape((truncate_freq, window_size, 1))
        single_sample.append(sample)

        label = np.zeros(window_size, dtype = np.float32)
        for k in range( window_size ):
            label[ k ] = log_gaussian_label( start_idx + k, label_pos, N, noverlap, std, is_log = is_log )
        single_label.append(label.reshape((window_size,)))

    base_index = label_interval_index + window_size
    for i in range( window_size ):
        start_idx, end_idx = base_index + i, base_index + i + window_size
        sample = Sxx[ start_idx : end_idx , : ].reshape( (truncate_freq, window_size, 1) )
        

        label = np.zeros(window_size , dtype = np.float32)
        for k in range( window_size ):
            label[ k ] = log_gaussian_label( start_idx + k, label_pos, N, noverlap, std, is_log = is_log )
        single_sample.append(sample)
        single_label.append(label.reshape((window_size,)))

    sample, label = np.stack( single_sample, axis = 0 ), np.stack( single_label, axis = 0 )
    if regularize_label:
        max_regu = window_size * 2 * ( N - noverlap ) / std
        if not is_log: 
            max_regu = np.exp( max_regu )
        label = label / max_regu
        print( "Applied regularization on label value, please change the post.py regu to {} when running".format( max_regu ) )
    return sample, label


def log_gaussian_label( idx, label_sample_idx, N, noverlap, std, is_log = 1 ):
    
    hop_size = N - noverlap
    mid_sample_idx = N + idx * hop_size - N // 2
    # print(mid_sample_idx, label_sample_idx)

    # return scipy.stats.lognorm.logpdf( mid_sample_idx, std, loc = label_sample_idx ) + np.log(2)
    exp_prob = -1.0*np.abs( mid_sample_idx - label_sample_idx ) / std
    # print( exp_prob )
    if is_log:
        return exp_prob
    else:
        return np.exp( exp_prob )

def asym_log_gaussian_label( idx, label_sample_idx, N, noverlap, std ):
    
    hop_size = N - noverlap
    mid_sample_idx = N + idx * hop_size

    return ( -0.5 * np.log( 2 * math.pi * std* std )  - \
           ( mid_sample_idx - label_sample_idx )**2 / 2 / std / std  + np.log(2))\
            * np.sign( mid_sample_idx - label_sample_idx )


def preprocess_gaussian( sound_file, label_file, 
                         instrument, window_size, 
                         N, hop_size, truncate_freq,
                         window_func, std, is_log ):

    fs, audio_data = wavfile.read( sound_file )
    raw_label = np.load( label_file )
    left_data, right_data = audio_data[ :, 0 ], audio_data[ :, 1 ]
    # print( left_data.shape, N, hop_size )
    noverlap = N - hop_size


    """ Process left data """
    f, t, Sxx, label_pos = create_sample_label( left_data, fs, window_func, 
                                                noverlap, N, truncate_freq, raw_label )
    # print( "t", t.shape )
    # calculate std

    left_sample, left_label = make_sample_label_gaussian( Sxx, label_pos,
                                                          N, noverlap, truncate_freq,
                                                          window_size, std = std, is_log = is_log )
    # print( "t", t.shape )
    """ Process right data """
    f, t, Sxx, label_pos = create_sample_label( right_data, fs, window_func, 
                                                noverlap, N, truncate_freq, raw_label )
    right_sample, right_label = make_sample_label_gaussian( Sxx, label_pos,
                                                          N, noverlap, truncate_freq,
                                                          window_size, std = std, is_log = is_log )
    return left_sample, left_label, right_sample, right_label

def preprocess_wrapper(  base_path, num_sample,
                         instrument = "piano", window_size = 10, 
                         N = 2048, hop_size = 512, truncate_freq = 256,
                         window_func = signal.blackman, window_text = "blackman",std = 256,  is_log = 1 ):
    
    left_sample, left_label, right_sample, right_label = [], [], [], []

    for k in range( num_sample ):
        audio_path = "{}/{}/single/mono_sound/{}_mono_{}.wav".format( base_path, instrument, instrument, k )
        label_path = "{}/{}/single/mono_label/{}_mono_{}.npy".format( base_path, instrument, instrument, k )

        ls, ll, rs, rl = preprocess_gaussian( audio_path, label_path, instrument, 
                                              window_size, N, hop_size, truncate_freq,
                                              window_func, std = std, is_log = is_log )
        left_sample.append( ls )
        left_label.append( ll )
        right_sample.append( rs )
        right_label.append( rl )

    np.save( "{}_N_{}_{}_{}_{}_single_left_sample".format( instrument, N, window_size, hop_size, truncate_freq ), np.concatenate( left_sample, axis = 0 ) )
    np.save( "{}_N_{}_{}_{}_{}_single_left_label".format( instrument, N, window_size, hop_size, truncate_freq ), np.concatenate( left_label, axis = 0 ) )

    np.save( "{}_N_{}_{}_{}_{}_single_right_sample".format( instrument, N, window_size, hop_size, truncate_freq ), np.concatenate( right_sample, axis = 0 ) )
    np.save( "{}_N_{}_{}_{}_{}_single__right_label".format( instrument, N, window_size, hop_size, truncate_freq ), np.concatenate( right_label, axis = 0 ) )

def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()

    parser.add_argument('--base-path', dest='base_path',
                        type=str, default="quantized",
                        help="folder that symbols are in")
    parser.add_argument('--num-sample', dest='num_sample',
                        type=int, default=1000,
                        help="number of sample")
    parser.add_argument('--instrument', dest='instrument', type=str,
                        default='piano', help="instrument")
    parser.add_argument('--window-size', dest='window_size',
                        type=int, default=20,
                        help="number of samples grouped as one picture")
    parser.add_argument('--N', dest='N',
                        type=int, default=1024,
                        help="FFT size for spectrogram")
    parser.add_argument('--hop-size', dest='hop_size',
                        type=int, default=128,
                        help="resolution not overlap between window")
    parser.add_argument('--truncated-freq', dest='truncated_freq',
                        type=int, default=256,
                        help="resolution not overlap between window")
    parser.add_argument('--window-text', dest='window_text',
                        type=str, default="blackman",
                        help="widnow type used as fft")
    parser.add_argument('--std', dest='std',
                        type=int, default=256,
                        help="the temperature of the exp decay")
    parser.add_argument('--is-log', dest='is_log',
                        type=int, default=1,
                        help="if the exp of prob of label is in log form")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    if args.window_text == "blackman":
        window_func = signal.blackman( args.N )


    preprocess_wrapper( args.base_path, args.num_sample,
                        args.instrument, args.window_size,
                        args.N, args.hop_size, args.truncated_freq,
                        window_func, args.window_text, args.std, args.is_log )