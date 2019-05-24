from opts import *
from clf import *
from data_source import *
from eval import *
import numpy as np
import os
from post import *
# piano"cello", "acoustic_guitar_steel", "oprano_sax", "harp"
for instrument in [ "piano"  ]:
    opts = Opts()
    opts.set_instrument( instrument )
    opts.s_data_source = Data_Source( opts.s_dim, opts.s_window_size, 1, opts )
    opts.s_data_source.load_unsplit_samples( opts.s_sample_path, opts.s_label_path )
    if opts.t_sample_path is not None:
        opts.t_data_source = Data_Source( opts.t_dim, opts.t_window_size, 1, opts )
        opts.t_data_source.load_unsplit_samples( opts.t_sample_path, opts.t_label_path )
    f = CLF( opts )
    
    if opts.train:
        f.train()
        np.save( "res/{}_N_{}_{}_{}.npy".format( instrument, opts.N, opts.scheme, "loss" ), np.array( f.loss_list ) )
        np.save( "res/{}_N_{}_{}_{}.npy".format( instrument, opts.N, opts.scheme, "MSE" ), np.array( f.mse_list ) )
        
    else:
        f.saver.restore( f.sess, opts.pretrain_path )
    if opts.t_sample_path  is not None:
        true_m, pred_m = eval_by_batch_pair( f )
    else:
        true_m, pred_m, sample_m = eval_by_batch( f )
    np.save( "res/true_m.npy", true_m )
    np.save( "res/pred_m.npy", pred_m )
    # print(  true_m[:10] )
    print( "-----------------------" )
    # print( pred_m[:10] )
    # print( np.mean( np.square( sigmoid( true_m ) - pred_m ) ) )
    print( accur( np.load(  "res/pred_m.npy"), np.load( "res/true_m.npy" ), thresh = 0.85) )

    eval_log_prob_syn_one( "res/true_m.npy", "res/pred_m.npy", true_thresh=0.85, is_log = False )
    plot_true( pred_m, true_m, sample_m )
    eval_urpm( "/media/steven/ScratchDisk/SeniorResearch/data/URMP/data/02_Sonata_vn_vn/AuMix_02_Sonata_vn_vn_16.wav", f, 1024, 1024 - 128 )