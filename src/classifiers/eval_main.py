import numpy as np 
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal

def eval_visualize( file_name, window_size, noverlap, side = "left" ):
	fs, Audiodata = wavfile.read(file_name)
	# Audiodata = np.zeros_like(Audiodata)
	freq, t, Sxx = signal.spectrogram( Audiodata, fs, window = signal.blackman( window_size ),
                                    noverlap = noverlap, nfft = window_size, nperseg = window_size )
	
	print(Sxx.shape)
	tmp = Sxx.transpose()
	Sxx = np.transpose( 10*np.log10(Sxx + 1e-8 ) )[:, :300]
	print(Sxx.shape)
	
	graph_i = 0
	plt.imshow(Sxx.transpose()); plt.show()

eval_visualize( "/media/steven/ScratchDisk/SeniorResearch/data/URMP/data/01_Jupiter_vn_vc/AuSep_1_vn_01_Jupiter_16.wav", 1024, 1024 - 128 )