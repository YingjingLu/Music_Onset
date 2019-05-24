import numpy as np

class Data_Source( object ):

    def __init__( self, w, h, c, opts ):
        self.input_shape = [ w, h, c ]
        self.width, self.height, self.dim = w, h, c
        self.batch_size = opts.batch_size
        
        self.train_sample = None
        self.test_sample = None 
        self.train_label = None 
        self.test_label = None

        self.num_train = 0
        self.num_test = 0
        self.cur_train = 0
        self.cur_test = 0

        self.train_split = opts.train_split
        self.opts = opts

    def append_train_sample( self, sample_matrix ):
        if self.train_sample is None:
            self.train_sample = sample_matrix 
        else:
            self.train_sample = np.concatenate( ( self.train_sample, sample_matrix ), axis = 0 )

    def append_train_label( self, label_matrix ):
        if self.train_label is None:
            self.train_label = label_matrix 
        else:
            self.train_label = np.concatenate( ( self.train_label, label_matrix ), axis = 0 )

    def append_test_sample( self, sample_matrix ):
        if self.test_sample is None:
            self.test_sample = sample_matrix 
        else:
            self.test_sample = np.concatenate( ( self.test_sample, sample_matrix ), axis = 0 )

    def append_test_label( self, label_matrix ):
        if self.test_label is None:
            self.test_label = label_matrix 
        else:
            self.test_label = np.concatenate( ( self.test_label, label_matrix ), axis = 0 )

    def load_unsplit_samples( self, sample_path, label_path ):
        sample = np.load( sample_path )
        label = np.load( label_path )
        num_sample = sample.shape[ 0 ]
        index = np.arange( num_sample, dtype = np.int )
        np.random.shuffle( index )
        sample = sample[ index ]
        label = label[ index ]
        train = int( num_sample * self.train_split )
        train_sample = sample[ :train, : ]
        test_sample = sample[ train:, : ]
        
        self.append_train_sample( train_sample )
        self.num_train += train 
        self.append_test_sample( test_sample )
        self.num_test += ( num_sample - train )

        self.append_train_label( label[ :train ] )
        self.append_test_label( label[ train: ] )



    def next_batch( self, batch_size = -1 ):
        if batch_size == -1:
            batch_size = self.batch_size
        sample = self.train_sample[ self.cur_train: self.cur_train + batch_size, : ]
        label = self.train_label[ self.cur_train: self.cur_train + batch_size, : ]
        self.cur_train += self.batch_size
        if( self.cur_train + self.batch_size >= self.num_train ):
            index = np.arange( self.num_train, dtype = np.int )
            np.random.shuffle( index )
            self.train_label = self.train_label[ index ]
            self.train_sample = self.train_sample[ index ]
            self.cur_train = 0
        return sample, label

    def get_test( self, batch_size = -1 ):
        if batch_size == -1:
            batch_size = self.batch_size
        sample = self.test_sample[ self.cur_test: self.cur_test + batch_size, : ]
        label = self.test_label[ self.cur_test: self.cur_test + batch_size, : ]
        self.cur_test += batch_size
        if( self.cur_test + self.batch_size >= self.num_test ):
            self.cur_test = 0
            return None, None
        return sample, label