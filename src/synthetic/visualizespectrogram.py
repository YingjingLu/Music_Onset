from scipy.io import wavfile # scipy library to read wav files
import numpy as np
import math 
from scipy import signal
import matplotlib.pyplot as plt
import os

base_path = "train"
def preprocess( instrument = "piano", file_index = 1, window_size = 10, N = 128, hop_size = 128, window = signal.blackman, window_text = "blackman" ):
	# N = 512 #Number of point in the fft
	window_size = 10
	single_sample, single_label, single_total = [], [], []
	double_sample, double_label, double_total = [], [], []
	truncate_freq = 500

	# single voice dataset
	AudioName = ( "quantized/{}/single/mono_sound/{}_mono_{}.wav".format( instrument, instrument, file_index ) ) 
	fs, Audiodata = wavfile.read(AudioName)
	raw_label = np.load( "quantized/{}/single/mono_label/{}_mono_{}.npy".format( instrument, instrument, file_index ) )
	
	print("Audio data shape: ", Audiodata.shape )
	print( "Expected seg num", (Audiodata.shape[0] - N) // ( hop_size ) )
	print( "expected label len", raw_label.shape )
	Audiodata = Audiodata[:,1]
	print( fs, Audiodata.shape )
	f, t, Sxx = signal.spectrogram(Audiodata, fs, window = signal.blackman(N), nfft = N, noverlap = N-hop_size )
	print( "f", f, f.shape )
	print( "t shape", t.shape )
	print( "SXX shape", Sxx.shape )
	plt.pcolormesh(t, f, Sxx)
	Sxx = Sxx[:truncate_freq, :]
	plt.plot( 1.002, 1000, "or" )
	plt.title( "FFT window size: {} samples, Resolution: {} samples".format( N, hop_size ) )
	plt.show()
	Sxx = np.transpose( 10*np.log10(Sxx + 1e-8) )
	# single_total.append( Sxx.reshape( ( 1, -1 ) ) )

	num_sample = Audiodata.shape[0]
	sample_per_interval = math.floor( num_sample / t.shape[0] )
	label_pos = np.where( raw_label == 1 )[0]
	label_interval_index = ( label_pos // sample_per_interval )[0]

	

	# print(label_interval_index)

	# for i in range( window_size ):
	# 	ahead = window_size - 1 - i
	# 	sample = Sxx[ label_interval_index - ahead: label_interval_index - ahead + window_size, :].reshape((1,truncate_freq, window_size, 1))
	# 	single_sample.append(sample)

	# 	label = np.zeros(window_size + 1, dtype = np.float32)
	# 	label[ i ] = 1.
	# 	single_label.append(label.reshape((1, window_size + 1)))

	# np.save( "{}_N_{}_single_sample".format( instrument, N ), np.concatenate( single_sample, axis = 0 ) )
	# np.save( "{}_N_{}_single_label".format( instrument, N ), np.concatenate( single_label, axis = 0 ) )

	# np.save( "{}_N_{}_double_sample".format( instrument, N ), np.concatenate( double_sample, axis = 0 ) )
	# np.save( "{}_N_{}_double_label".format( instrument, N ), np.concatenate( double_label, axis = 0 ) )

	# np.save( "{}_N_{}_total_sample".format( instrument, N ), np.concatenate( single_sample + double_sample, axis = 0 ) )
	# np.save( "{}_N_{}_total_label".format( instrument, N ), np.concatenate( single_label + double_label, axis = 0 ) )

	# np.save( "{}_N_{}_single_all".format( instrument, N ), np.concatenate( single_total, axis = 0 ) )
	# np.save( "{}_N_{}_double_all".format( instrument, N ), np.concatenate( double_total, axis = 0 ) )

instrument = "piano"
file = 4
print("Working on {}".format( instrument ) )
preprocess( instrument = instrument, file_index = file )
print( "Finished ------------------------" )

print( "All Done" )