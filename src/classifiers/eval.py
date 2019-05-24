import numpy as np 
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal

def plot( pred, true ):
	fpr, tpr, _ = roc_curve( true.ravel(), pred.ravel() )
	roc_auc = auc(fpr, tpr)
	plt.figure()
	lw = 2
	plt.plot(fpr, tpr, color='darkorange',
	         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.legend(loc="lower right")
	plt.show()

def sigmoid( m ):
	return np.exp( m ) / ( 1 + np.exp( m ) )

def accur( pred, true, thresh = 0.8 ):
	true = true.ravel()
	pred = pred.ravel()
	true =  ( true >= thresh )
	pred =  ( pred >= thresh ) 

	num_sample = true.shape[0]

	accuracy = np.sum( pred == true ) / num_sample
	index = np.arange( num_sample,  )

	positive = ( true == 1 )
	negative = ( true == 0 )

	true_negative = np.sum( pred[negative] == true[negative] )
	true_positive = np.sum( pred[positive] == true[positive] )
	false_positive = np.sum( pred[negative] )
	false_negative = np.sum( pred[positive] == 0 )
	return accuracy, true_negative, true_positive, false_positive, false_negative

def plot_true( pred, true, sample, num_plot = 10, shuffle = True ):
	if(shuffle):
		shuffle = np.arange( pred.shape[0], dtype = np.int )
		np.random.shuffle( shuffle )
		pred = pred[ shuffle ]
		true = true[ shuffle ]
		print(sample.shape)
		sample = sample[ shuffle ]
	x = np.arange( pred.shape[1] )
	for i in range( num_plot ):
		plt.subplot( num_plot , 3 , i * 3 + 1 )
		plt.bar( x, pred[ i ] )
		plt.subplot( num_plot , 3 , i * 3 + 2 )
		plt.bar( x, true[ i ] )
		plt.subplot( num_plot , 3 , i * 3 + 3 )

		plt.imshow( sample[ i ].reshape( 20, 300 ) )
	plt.show()


def eval_by_batch( f ):
	true_list, pred_list = [], []
	sample_list = []
	sample, label = f.opts.s_data_source.get_test()
	res = f.sess.run( f.activated_res, feed_dict = { f.s_input_sample : sample } )
	sample, label = f.opts.s_data_source.get_test()
	true_list.append( np.exp( label ) )
	pred_list.append( res )
	sample_list.append( sample)
	# iteration = 0
	while( label is not None ):
		res = f.sess.run( f.activated_res, feed_dict = { f.s_input_sample : sample } )
		true_list.append( np.exp( label ) )
		pred_list.append( res )
		sample_list.append( sample ) 
		sample, label = f.opts.s_data_source.get_test()
	#     iteration += 1
	true_m = np.concatenate( true_list, axis = 0 )
	pred_m = np.concatenate( pred_list, axis = 0 )
	sample_list = np.concatenate( sample_list, axis = 0 )
	return true_m, pred_m, sample_list

def eval_by_batch_pair( f ):
	true_list, pred_list = [], []
	sample, label = f.opts.s_data_source.get_test()
	t_sample, t_label = f.opts.t_data_source.get_test()
	res = f.sess.run( f.activated_res, feed_dict = { f.s_input_sample : sample, f.t_input_sample: t_sample } )
	sample, label = f.opts.s_data_source.get_test()
	true_list.append( label )
	pred_list.append( res )
	# iteration = 0
	while( label is not None ):
		res = f.sess.run( f.activated_res, feed_dict = { f.s_input_sample : sample, f.t_input_sample: t_sample } )
		true_list.append( label )
		pred_list.append( res )
		sample, label = f.opts.s_data_source.get_test()
		t_sample, t_label = f.opts.t_data_source.get_test()
	#     iteration += 1
	true_m = np.concatenate( true_list, axis = 0 )
	pred_m = np.concatenate( pred_list, axis = 0 )
	return true_m, pred_m

def eval_visualize( file_name, f, window_size, noverlap, side = "left" ):
	fs, Audiodata = wavfile.read(file_name)
	if side == "right":
		Audiodata = Audiodata[:,1]
	else:
		Audiodata = Audiodata[ :,0 ]
	# Audiodata = np.zeros_like(Audiodata)
	freq, t, Sxx = signal.spectrogram( Audiodata, fs, window = signal.blackman( window_size ),
                                    noverlap = noverlap, nfft = window_size, nperseg = window_size )
	
	print(Sxx.shape)
	tmp = Sxx.transpose()
	Sxx = np.transpose( 10*np.log10(Sxx + 1e-8 ) )[:, :f.opts.s_dim]
	print(Sxx.shape)
	
	graph_i = 0
	plt.subplot( 1 + 4 ,1, 1 )
	plt.imshow(Sxx.transpose())
	for shift in range( 0, 20, 5 ):
		res_label = []
		batch = []
		graph_i += 1
		for i in range( 0, Sxx.shape[0], f.opts.s_window_size):
			r = Sxx[ i:i+f.opts.s_window_size, : ]
			if r.shape[ 0 ] == f.opts.s_window_size:
				lol = np.zeros_like( r.reshape( 1, f.opts.s_dim, f.opts.s_window_size, 1 ) )
				lol[ 0, :, :, 0 ] = r.transpose()
				batch.append( lol )
		s = np.vstack( batch )
		print(s.shape)
		sample = np.vstack( ( s, np.zeros( ( f.opts.batch_size - s.shape[0], f.opts.s_dim, f.opts.s_window_size, 1 ) ) ) )
		res = f.sess.run( f.activated_res, feed_dict = { f.s_input_sample : sample } )
		print(res.shape, "HAHAHAHAH")
		for j in range( s.shape[0] ):
			res_label.append( res[ j ].ravel() )
		# print(res_label)
		print(len(res_label), res_label[0].shape)
		res_label = np.hstack( res_label )
		if shift != 0:
			res_label = np.hstack( (np.zeros( [shift] ), res_label) )
		if res_label.shape[0] < Sxx.shape[0]:
			res_label = np.hstack( ( res_label, np.zeros( Sxx.shape[0] - res_label.shape[0] ) ) )

		plt.subplot( 1 + 4, 1, graph_i + 1 )
		plt.bar( np.arange( res_label.shape[0]), res_label )
	plt.show()	

def eval_urpm( file_name, f, window_size, noverlap, side = "left" ):
	fs, Audiodata = wavfile.read(file_name)
	if side == "right":
		Audiodata = Audiodata
	else:
		Audiodata = Audiodata
	# Audiodata = np.zeros_like(Audiodata)
	freq, t, Sxx = signal.spectrogram( Audiodata, fs, window = signal.blackman( window_size ),
                                    noverlap = noverlap, nfft = window_size, nperseg = window_size )
	
	print(Sxx.shape)
	tmp = Sxx.transpose()
	Sxx = np.transpose( 10*np.log10(Sxx + 1e-8 ) )[:, :f.opts.s_dim]
	print(Sxx.shape)
	
	graph_i = 0
	plt.subplot( 2 ,1, 1 )
	plt.imshow(Sxx.transpose())

	batch = []
	for i in range( 0, Sxx.shape[0], f.opts.s_window_size ):
		r = Sxx[ i:i+f.opts.s_window_size, : ]
		if r.shape[0] == f.opts.s_window_size:
			lol = np.zeros_like( r.reshape( 1, f.opts.s_dim, f.opts.s_window_size, 1 ) )
			lol[ 0, :, :, 0 ] = r.transpose()
			batch.append( lol )
	s = np.vstack( batch )

	label = []
	for i in range( 0, s.shape[0], f.opts.batch_size ):
		sample = s[ i: i+ f.opts.batch_size, :]
		if sample.shape[0] == f.opts.batch_size:
			res = f.sess.run( f.activated_res, feed_dict = { f.s_input_sample : sample } )
			label.append( res.ravel() )
	label = np.hstack(label)
	plt.subplot( 2, 1, 2 )
	plt.plot( range(len(label)), label )
	plt.show()	
