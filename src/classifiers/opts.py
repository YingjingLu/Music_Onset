import os
class Opts( object ):
    def __init__( self ):
        self.batch_size = 64
        self.scheme = "double"
        self.s_data_source = None
        self.t_data_source = None
        self.class_size = 20
        self.optimizer = "adam"
        self.lr = 1e-4
        self.train_iter = 100000

        
        self.N = 1024
        self.train_split = 0.75
        self.s_dim = 300 # truncated freq
        self.clf_type = "square"
        self.scheme = self.clf_type
        self.train = False
        # self.s_sample_path = "data/piano_N_{}_{}_{}_{}_single_left_sample.npy".format( self.N, self.class_size,self.s_dim, self.s_dim )
        # self.s_label_path = "data/piano_N_{}_{}_{}_{}_single_left_label.npy".format( self.N, self.class_size, self.s_dim, self.s_dim )
        self.s_sample_path = "data/tri_sample.npy"
        self.s_label_path = "data/tri_label.npy"

        self.s_window_size = 20
        self.regu = 1.0

        self.t_sample_path = None 
        self.t_label_path = None
        self.t_dim = 300
        self.t_window_size = 20

        self.pretrain_path = '/media/steven/ScratchDisk/SeniorResearch/src/classifiers/ckpt/piano_N_1024_window_20_square/80000/model.ckpt'# "cpt1000000/"

    def set_instrument( self, instrument ):
        base_path = "ckpt/{}_N_{}_window_{}_{}".format( instrument, self.N, self.s_window_size, self.scheme )
        os.mkdir( base_path ) if not os.path.exists( base_path ) else print()
        self.cpt_path = base_path
        self.instrument = instrument 

