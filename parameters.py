from .meta import SENSOR_LABEL
import argparse


parser = argparse.ArgumentParser(epilog="""
	Passing a filename will estimate the desired parameter from the Rrs 
	contained in that file. Otherwise, a model will be trained (if not 
	already existing), and estimates will be made for the testing data.\n
""")

parser.add_argument("--run_name", default="default",   action ="store_true", help="Allows results to be saved with different names")

parser.add_argument("filename",    nargs  ="?",          help="CSV file containing Rrs values to estimate from")
parser.add_argument("--model_loc", default="Model",      help="Location of trained models")
# parser.add_argument("--sim_loc",   default="/media/brandon/NASA/Data/Train", help="Location of simulated data")

#With Spain and 0's go to 0.1 and no copies/visually bad data, current dataset used in the paper
parser.add_argument("--data_loc",  default="/home/ryanoshea/MDN_PC/MDN/script_formatted_data/Ryan_data/Data/Test_29_0_1_no_copies",  help="Location of in situ data") #
# parser.add_argument("--data_loc",  default="/home/ryanoshea/MDN_PC/MDN/script_formatted_data/Ryan_data/Data/Test_29_0_1_no_copies_PRISMA",  help="Location of in situ data") 


parser.add_argument("--allow_nan_out",  default=0,  help="1 Allows nans to be output (uses imputation), 0 does not allow nan")
parser.add_argument("--return_opt",  default=1,  help="1 runs optimization on benchmarks, to enable a fair comparison")
parser.add_argument("--sim_loc",   default="D:/Data/Train", help="Location of simulated data")
parser.add_argument("--n_redraws", default=50,     type=int,   help="Number of plot redraws during training (i.e. updates plot every n_iter / n_redraws iterations); only used with --plot_loss.")
parser.add_argument("--n_rounds",  default=10,     type=int,   help="Number of models to fit, with median output as the final estimate")



''' Flags '''
parser.add_argument("--threshold", default=None,   type=float, help="Output the maximum prior estimate when the prior is above this threshold, and the weighted average estimate otherwise. Set to None, thresholding is not used.")
parser.add_argument("--avg_est",   action ="store_true", help="Use the prior probability weighted mean as the estimate. Otherwise, use maximum prior.")
parser.add_argument("--no_save",   action ="store_true", help="Do not save the model after training")
parser.add_argument("--no_load",   default=False,   action ="store_true", help="Do load a saved model (and overwrite, if not no_save)")
parser.add_argument("--verbose",   action ="store_true", help="Verbose output printing")
parser.add_argument("--silent",    action ="store_true", help="Turn off all printing")
parser.add_argument("--plot_loss", default=False,action ="store_true", help="Plot the model loss while training")
parser.add_argument("--plot_matchups", default=1,action ="store_true", help="0 for OLCI, 1 for HICO, 2 for precomputing, 3 for PRISMA")


parser.add_argument("--darktheme", action ="store_true", help="Use a dark color scheme in plots")
parser.add_argument("--animate",   action ="store_true", help="Store the training progress as an animation (mp4)")
parser.add_argument("--save_data", action ="store_true", help="Save the data used for the given args")
parser.add_argument("--save_stats",action ="store_true", help="Store partial training statistics & estimates for later analysis")


''' Flags which require model retrain if changed '''

update = parser.add_argument_group('Model Parameters', 'Parameters which require a new model to be trained if they are changed')
update.add_argument("--sat_bands", action ="store_true", help="Use bands specific to certain products when utilizing satellite retrieved spectra")
update.add_argument("--benchmark", default=True, action ="store_true", help="Train only on partial dataset, and use remaining to benchmark")
update.add_argument("--product",   default="chl,PC",        help="Product to estimate")
update.add_argument("--sensor",    default="HICO-noBnoNIR",        help="Sensor to estimate from", choices=SENSOR_LABEL) 
update.add_argument("--align",     default=None,         help="Comma-separated list of sensors to align data with. Passing \"all\" uses all sensors.", choices=['all']+list(SENSOR_LABEL))
update.add_argument("--model_lbl", default="",      	 help="Label for a model")
update.add_argument("--seed",      default=42,   type=int,   help="Random seed")
update.add_argument("--n_train", default=1.0,action ="store_true", help="The proportion of training data we use")
update.add_argument("--n_valid", default=0,action ="store_true", help="The proportion of validation data we use")
update.add_argument("--correlation_threshold",  default=0.35,     type=int,   help="the correlation cutoff threshold for band ratios and line heights") 
update.add_argument("--using_correlations",  default=True,     type=bool,   help="wether or not to use the correlation threshold")
update.add_argument("--only_append_LH",  default=False,     type=int,   help="only appends line heights")
update.add_argument("--band_ratios_thresholded",  default=None,     type=int,   help="Band ratios that have been thresholded with a specific correlation cutoff")
update.add_argument("--line_heights_thresholded",  default=None,     type=int,   help="Line Heights that have been thresholded with a specific correlation cutoff")
update.add_argument("--split_by_set",  default=False,  help="reports error on each subset of the dataset")

''' Flags which have a yet undecided default value ''' 
# update.add_argument("--no_noise",  action ="store_true", help="Do not add noise when training the model")
update.add_argument("--use_noise", default=False,action ="store_true", help="Add noise when training the model")

update.add_argument("--no_ratio",  default=False,action ="store_true", help="Do not add band ratios as input features")
#update.add_argument("--use_ratio", action ="store_true", help="Add band ratios as input features")

update.add_argument("--no_only_ratio",  default=False, action ="store_true", help="Only use band ratios as input")
# update.add_argument("--add_ratios",  default=False, action ="store_true", help="Adds ratios and line heights if true")

update.add_argument("--use_tchlfix",  default=False,action ="store_true", help="Correct chl for pheopigments")
# update.add_argument("--no_tchlfix", action ="store_true", help="Do not correct chl for pheopigments")

# parser.add_argument("--no_cache",  action ="store_true", help="Do not use any cached data")
# parser.add_argument("--use_cache", action ="store_true", help="Use cached data, if available")

update.add_argument("--use_boosting",  default=False,action ="store_true", help="Use boosting when training in multiple trials")
#update.add_argument("--no_boosting",action ="store_true", help="Do not use boosting when training in multiple trials")

#update.add_argument("--no_bagging",action ="store_true", help="Do not use bagging when training in multiple trials")
update.add_argument("--use_bagging",   default=False,action ="store_true", help="Use bagging when training in multiple trials")

parser.add_argument("--use_sim", default=False,action ="store_true", help="Use simulated training data")

''' Hyperparameters '''
hypers = parser.add_argument_group('Hyperparameters', 'Hyperparameters used in training the model (also requires model retrain if changed)') 
hypers.add_argument("--n_iter",      default=10000,  type=int,   help="Number of iterations to train the model")
hypers.add_argument("--n_mix",       default=5,      type=int,   help="Number of gaussians to fit in the mixture model")
hypers.add_argument("--batch",       default=128,    type=int,   help="Number of samples in a training batch")
hypers.add_argument("--n_hidden",    default=100,    type=int,   help="Number of neurons per hidden layer")
hypers.add_argument("--n_layers",    default=5,      type=int,   help="Number of hidden layers")
hypers.add_argument("--imputations", default=5,      type=int,   help="Number of samples used for imputation when handling NaNs in the target")
hypers.add_argument("--lr", 	     default=1e-3,   type=float, help="Learning rate")
hypers.add_argument("--l2", 	     default=1e-3,   type=float, help="L2 regularization")
hypers.add_argument("--epsilon",     default=1e-3,   type=float, help="Variance regularization (ensures covariance has a valid decomposition)")

dataset = parser.add_mutually_exclusive_group()
dataset.add_argument("--all_test",       action="store_const", dest="dataset", const="all")
dataset.add_argument("--sentinel_paper", action="store_const", dest="dataset", const="sentinel_paper")
parser.set_defaults(dataset='all', use_sim=False)



def get_args(kwargs={}, use_cmdline=True, **kwargs2):
	kwargs2.update(kwargs)

	if use_cmdline:	args = parser.parse_args()
	else:           args = parser.parse_args([])
	
	for k, v in kwargs2.items():
		setattr(args, k, v)
	return args
