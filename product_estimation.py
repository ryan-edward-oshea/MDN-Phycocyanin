from pathlib import Path
from sklearn import preprocessing
from tqdm  import trange 
import numpy as np 
import hashlib
import math
from .mdn   import MDN
from .meta  import get_sensor_bands, SENSOR_LABEL, ANCILLARY, PERIODIC
from .utils import get_labels, get_data, generate_config, using_feature, split_data,get_matchups, assign_spectra_OWT, assign_spectrum_OWT,get_in_situ_PRISMA, assign_spectra_OWT_angle_nearest_Rrs
from .metrics import performance, mdsa
from .plot_utils import plot_scatter, plot_remote_insitu, plot_scatter_summary,plot_band_correlations,plot_bar_chart,plot_histogram, plot_Rrs_scatter
from .benchmarks import run_benchmarks
from .parameters import get_args
from .transformers import TransformerPipeline, LogTransformer, RatioTransformer, BaggingColumnTransformer, IdentityTransformer, TransformerPipeline_ratio
from .plot_utils import add_identity

def get_estimates(args, x_train=None, y_train=None, x_test=None, y_test=None, output_slices=None, dataset_labels=None):
	''' 
	Estimate all target variables for the given x_test. If a model doesn't 
	already exist, creates a model with the given training data. 
	'''		
	wavelengths  = get_sensor_bands(args.sensor, args)
	print(args.sensor,wavelengths)
	store_scaler = lambda scaler, args=[], kwargs={}: (scaler, args, kwargs)

	# Note that additional x_scalers are added to the beginning of the pipeline (e.g. Robust( bagging( ratio(x) ) ))
	args.x_scalers = [
			store_scaler(preprocessing.RobustScaler),
	]
	args.y_scalers = [
		store_scaler(LogTransformer),
		store_scaler(preprocessing.MinMaxScaler, [(-1, 1)]),
	]

	# We only want bagging to be applied to the columns if there are a large number of feature (e.g. ancillary features included) 
	many_features = any(x is not None and (x.shape[1]-len(wavelengths)) > 15 for x in [x_train, x_test])

	# Add bagging to the columns (use a random subset of columns, excluding the first <n_wavelengths> columns from the process)
	if using_feature(args, 'bagging') and (using_feature(args, 'ratio') or many_features):
		n_extra = 0 if not using_feature(args, 'ratio') else RatioTransformer.n_features
		args.x_scalers = [
			store_scaler(BaggingColumnTransformer, [len(wavelengths)], {'n_extra':n_extra}),
		] + args.x_scalers
	verbose_arg = 0
	if verbose_arg: print(args)

	# if args.add_ratios: 
	if using_feature(args, 'ratio'):
		if verbose_arg: print('USING RATIO')
		if using_feature(args, 'only_ratio'):
			if verbose_arg: print('OVERWRITING INPUTS get_estimates, using only ratio')
		args.x_scalers = [
			store_scaler(RatioTransformer, [list(wavelengths),using_feature(args, 'only_ratio'),args.band_ratios_thresholded,args.line_heights_thresholded,args.only_append_LH]),
		] + args.x_scalers

	# Add a few additional variables to be stored in the generated config file
	setattr(args, 'data_wavelengths', list(wavelengths))
	if x_train is not None: setattr(args, 'data_xtrain_shape', x_train.shape)
	if y_train is not None: setattr(args, 'data_ytrain_shape', y_train.shape)
	if x_test  is not None: setattr(args, 'data_xtest_shape',  x_test.shape)
	if y_test  is not None: setattr(args, 'data_ytest_shape',  y_test.shape)
	if dataset_labels is not None: 
		sets_str  = ','.join(sorted(map(str, np.unique(dataset_labels))))
		sets_hash = hashlib.sha256(sets_str.encode('utf-8')).hexdigest()
		setattr(args, 'datasets_hash', sets_hash)

	model_path = generate_config(args, create=x_train is not None)	
	args.config_name = model_path.name
	if args.verbose: print(model_path)

	uppers, lowers   = [], []
	x_full, y_full   = x_train, y_train
	x_valid, y_valid = None, None

	estimates = []
	for round_num in trange(args.n_rounds, disable=args.verbose or (args.n_rounds == 1) or args.silent):
		args.curr_round = round_num
		curr_round_seed = args.seed+round_num if args.seed is not None else None
		np.random.seed(curr_round_seed)

		# 75% of rows used in bagging
		if using_feature(args, 'bagging') and x_train is not None and args.n_rounds > 1:
			(x_train, y_train), (x_valid, y_valid) = split_data(x_full, y_full, n_train=0.75, seed=curr_round_seed) 

		datasets = {k: dict(zip(['x','y'], v)) for k,v in {
			'train' : [x_train, y_train],
			'valid' : [x_valid, y_valid],
			'test'  : [x_test, y_test],
			'full'  : [x_full, y_full],
		}.items() if v[0] is not None}

		model_kwargs = {
			'n_mix'      : args.n_mix, 
			'hidden'     : [args.n_hidden] * args.n_layers, 
			'lr'         : args.lr,
			'l2'         : args.l2,
			'n_iter'     : args.n_iter,
			'batch'      : args.batch,
			'avg_est'    : args.avg_est,
			'imputations': args.imputations,
			'epsilon'    : args.epsilon,
			'threshold'  : args.threshold,
			'scalerx'    : TransformerPipeline([S(*args, **kwargs) for S, args, kwargs in args.x_scalers]),
			'scalery'    : TransformerPipeline([S(*args, **kwargs) for S, args, kwargs in args.y_scalers]),
			'model_path' : model_path.joinpath(f'Round_{round_num}'),
			'no_load'    : args.no_load,
			'no_save'    : args.no_save,
			'seed'       : curr_round_seed,
			'verbose'    : args.verbose,
		}

		model = MDN(**model_kwargs)
		model.fit(x_train, y_train, output_slices, args=args, datasets=datasets)

		if x_test is not None:
			partial_est = []
			chunk_size  = args.batch * 100

			# To speed up the process and limit memory consumption, apply the trained model to the given test data in chunks
			for i in trange(0, len(x_test), chunk_size, disable=not args.verbose):
				est = model.predict(x_test[i:i+chunk_size], confidence_interval=None)
				partial_est.append( np.array(est, ndmin=3) )

			estimates.append( np.hstack(partial_est) )
			if hasattr(model, 'session'): model.session.close()

			if args.verbose and y_test is not None:
				median = np.median(np.stack(estimates, axis=1)[0], axis=0)
				labels = get_labels(wavelengths, output_slices, n_out=y_test.shape[1])
				for lbl, y1, y2 in zip(labels, y_test.T, median.T):
					print( performance(f'{lbl:>7s} Median', y1, y2) )
				print(f'--- Done round {round_num} ---\n')

	# Confidence bounds will contain [upper bounds, lower bounds] with the same shape as 
	# estimates) if a confidence_interval within (0,1) is passed into model.predict 
	if x_test is not None:
		estimates, *confidence_bounds = np.stack(estimates, axis=1)
	return estimates, model.output_slices

def get_band_correlations(args, x_data,y_data,output_slices):
	'''
	Gets correlations between the transformer bands being used and the underlying data
	'''
	wavelengths  = get_sensor_bands(args.sensor, args)

	store_scaler = lambda scaler, args=[], kwargs={}: (scaler, args, kwargs)

	class ratio_correlation_tester(object):
		def __init__(self, scalerx=None, scalery=None, **kwargs):

			self.scalerx      = scalerx if scalerx is not None else IdentityTransformer()
			self.scalery      = scalery if scalery is not None else IdentityTransformer()

		def fit(self, X, y, output_slices={'': slice(None)}, **kwargs):
				self.scalerx.fit( self._ensure_format(X) )
				self.scalery.fit( self._ensure_format(y) )

				# Gather all data (train, validation, test, ...) into singular object
				datasets = kwargs['datasets'] = kwargs.get('datasets', {})
				datasets.update({'train': {'x' : X, 'y': y}})

				for key, data in datasets.items(): 
					datasets[key].update({
						'x_t' : self.scalerx.transform( self._ensure_format(data['x']) ),
						'y_t' : self.scalery.transform( self._ensure_format(data['y']) ),
					})
				x_data = datasets['train']['x_t']
				y_data = datasets['train']['y_t'] 

				import matplotlib.pyplot as plt

				BR_numerator_denominator,LH_center_bandwidth = plot_band_correlations(x_data, y_data, products = args.product , run_name=args.run_name,sensor=args.sensor,labels = self.scalerx.return_labels(),PC_correlation_threshold=args.correlation_threshold)
				return BR_numerator_denominator,LH_center_bandwidth;

		def _ensure_format(self, z):
			''' Ensure passed matrix has two dimensions [n_sample, n_feature], and add the n_feature axis if not '''
			z = np.array(z).copy()
			return z[:, None] if len(z.shape) == 1 else z

	if using_feature(args, 'ratio'):
		print('Using ratio')
		if using_feature(args, 'only_ratio'):
			print('Overwriting inputs, using only ratio')
		if using_feature(args, 'only_ratio'):
			args.x_scalers_ratio_only = [store_scaler(RatioTransformer, [list(wavelengths),using_feature(args, 'only_ratio')])]
			args.scaler_x_ratio_only = TransformerPipeline([S(*args, **kwargs) for S, args, kwargs in args.x_scalers_ratio_only])


	ratio_tester_kwargs = {
		'scalerx'    : TransformerPipeline_ratio([S(*args, **kwargs) for S, args, kwargs in args.x_scalers_ratio_only]),
		'scalery'    : None, 
	}

	datasets = {k: dict(zip(['x','y'], v)) for k,v in {
			'train' : [x_data, y_data],
		}.items() if v[0] is not None}


	ratio_tester = ratio_correlation_tester(**ratio_tester_kwargs)
	BR_numerator_denominator,LH_center_bandwidth = ratio_tester.fit(x_data, y_data, output_slices, args=args, datasets=datasets)
	return BR_numerator_denominator,LH_center_bandwidth;
		
def image_estimates(data, args, sensor='', product_name='chl', rhos=False, anc=False, **kwargs):
	''' 
	Takes any number of input bands (shaped [Height, Width]) and
	returns the products for that image, in the same shape. 
	Assumes the given bands are ordered by wavelength from least 
	to greatest, and are the same bands used to train the network.
	Supported products: {chl}

	rhos and anc models are not yet available.  
	'''
	valid_products = ['chl','chl,tss,cdom','chl,tss','chl,PC']

	if rhos:  sensor = sensor.replace('S2B','MSI') + '-rho'
	elif anc: sensor = sensor.replace('S2B','MSI')

	if isinstance(data, list):
		assert(all([data[0].shape == d.shape for d in data])), (
			f'Not all inputs have the same shape: {[d.shape for d in data]}')
		data = np.dstack(data)

	assert(sensor), (
		f'Must pass sensor name to image_estimates function')
	assert(sensor in SENSOR_LABEL), (
		f'Requested sensor {sensor} unknown. Must be one of: {list(SENSOR_LABEL.keys())}')
	assert(product_name in valid_products), (
		f'Requested product unknown. Must be one of {valid_products}')
	assert(len(data.shape) == 3), (
		f'Expected data to have 3 dimensions (height, width, feature). Found shape: {data.shape}')
	
	expected_features = len(get_sensor_bands(sensor)) + (len(ANCILLARY)+len(PERIODIC) if anc or rhos else 0)
	assert(data.shape[-1] == expected_features), (
		f'Got {data.shape[-1]} features; expected {expected_features} features for sensor {sensor}')
	
	if rhos: 
		setattr(args, 'n_iter', 10000)
		setattr(args, 'model_lbl', 'l2gen_rhos-anc')
	elif anc:
		setattr(args, 'n_iter', 10000)
		setattr(args, 'model_lbl', 'l2gen_Rrs-anc')
		
	im_shape = data.shape[:-1] 
	im_data  = np.ma.masked_invalid(data.reshape((-1, expected_features)))
	im_mask  = np.any(im_data.mask, axis=1)
	im_data  = im_data[~im_mask]
	pred,idx = get_estimates(args, x_test=im_data)
	products = np.median(pred, 0) 
	product  = np.atleast_2d( products)
	est_mask = np.tile(im_mask[:,None], (1, product.shape[1]))
	est_data = np.ma.array(np.zeros(est_mask.shape)*np.nan, mask=est_mask, hard_mask=True)
	est_data.data[~im_mask] = product
	return [p.reshape(im_shape) for p in est_data.T], idx

def apply_model(x_test, use_cmdline=True, **kwargs):
	''' Apply a model (defined by kwargs and default parameters) to x_test '''
	args = get_args(kwargs, use_cmdline=use_cmdline)

	if args.using_correlations:
		args = assign_band_ratio_correlations(args)
	preds, idxs = get_estimates(args, x_test=x_test)
	return np.median(preds, 0), idxs
	
def main(dictionary_of_run_parameters=None): 

	if dictionary_of_run_parameters != None:
		args = get_args(kwargs=dictionary_of_run_parameters)

		if args.plot_matchups == 2:
			args.no_load=False
			dictionary_of_matchups = get_matchups(args.sensor)	
			print(dictionary_of_matchups.keys())
			print(dictionary_of_matchups['site_label'])
			insitu_Rrs = np.reshape(dictionary_of_matchups['insitu_Rrs_resampled'],(1,-1,len(get_sensor_bands(args.sensor))))
			remote_Rrs = np.reshape(dictionary_of_matchups['Rrs_retrieved'],(1,-1,len(get_sensor_bands(args.sensor))))

			print(np.shape(insitu_Rrs))
			
			estimates_in_situ = image_estimates(data=insitu_Rrs,args=args, sensor=args.sensor, product_name='chl,PC',kwargs=dictionary_of_run_parameters)
			estimates_remote  = image_estimates(data=remote_Rrs,args=args, sensor=args.sensor, product_name='chl,PC',kwargs=dictionary_of_run_parameters)
			plot_remote_insitu(y_remote=estimates_remote, y_insitu=estimates_in_situ,dictionary_of_matchups=dictionary_of_matchups,products=['chl','PC'],sensor=args.sensor,run_name=args.run_name)
			exit()

		if args.filename:
			print('In args filename')
			filename = Path(args.filename)
			assert(filename.exists()), (
				f'Expecting path to in situ data as the passed argument, but "{filename}" does not exist.')

			x_test = np.loadtxt(args.filename, delimiter=',')
			print(f'Min Rrs: {x_test.min(0)}')
			print(f'Max Rrs: {x_test.max(0)}')
			print(f'Generating estimates for {len(x_test)} data points ({x_test.shape})')
			preds, idxs = get_estimates(args, x_test=x_test)
			print(f'Min: {np.median(preds, 0).min(0)}')
			print(f'Max: {np.median(preds, 0).max(0)}')

			labels = get_labels(get_sensor_bands(args.sensor, args), idxs, preds[0].shape[1])
			preds  = np.append(np.array(labels)[None,:], np.median(preds, 0), 0)

			filename = filename.parent.joinpath(f'MDN_{filename.stem}.csv').as_posix()
			print(f'Saving estimates at location "{filename}"')
			np.savetxt(filename, preds.astype(str), delimiter=',', fmt='%s')

		# Save data used with the given args
		elif args.save_data:
			print('In Save data')

			x_data, y_data, slices, locs, z_data, z_w_data,x_data_full,lat_data,lon_data = get_data(args)

			valid  = np.any(np.isfinite(x_data), 1)
			x_data = x_data[valid].astype(str)
			y_data = y_data[valid].astype(str)
			locs   = np.array(locs).T[valid].astype(str)
			wvls   = list(get_sensor_bands(args.sensor, args).astype(int).astype(str))
			lbls   = get_labels(get_sensor_bands(args.sensor, args), slices, y_data.shape[1])
			data   = np.append([wvls], x_data.astype(str), 0)
			data_full = np.append(np.append(locs, x_data, 1), y_data, 1)
			data_full = np.append([['index', 'dataset']+wvls+lbls], data_full, 0)
			np.savetxt(f'{args.sensor}_data_full.csv', data_full, delimiter=',', fmt='%s')

		# Train a model with partial data, and benchmark on remaining
		elif args.benchmark:
			import matplotlib.pyplot as plt
			import matplotlib.gridspec as gridspec
			import matplotlib.ticker as ticker
			import matplotlib.patheffects as pe 
			import seaborn as sns 

			if args.dataset == 'sentinel_paper':
				setattr(args, 'fix_tchl', True)
				setattr(args, 'seed', 1234)

			np.random.seed(args.seed)
			
			bands   = get_sensor_bands(args.sensor, args)
			bands_full   = get_sensor_bands(args.sensor.split('-')[0], args) 

			n_train = args.n_train if args.dataset != 'sentinel_paper' else 1000
			n_valid = args.n_valid if args.dataset != 'sentinel_paper' else 1000

			x_data, y_data, slices, locs, z_data, z_w_data,x_data_full,lat_data,lon_data         = get_data(args)

			locations_list=[]

			if args.split_by_set:
				locations_list = assign_locations(locs)
				unique_locations = list(sorted(set(locations_list)))
				print(type(unique_locations))
				print('Unique Locations: ', (unique_locations))
				unique_locations.append("full_dataset_split")
				number_of_unique_locations = len(unique_locations)

			else:
				unique_locations = ["full_dataset_split"]
				number_of_unique_locations = len(unique_locations)

			print("Unique locations: {}".format(unique_locations))
			print("{} locations".format(number_of_unique_locations))

			y_test_list_subset = []
			estimates_list_subset = []
			run_name_list_subset=[]
			for unique_location_counter, unique_location in enumerate(unique_locations):
				print('TESTING ON:', unique_location)

				if unique_location == "full_dataset_split":
					args.split_by_set = False 

				if n_valid == 0:
					(x_train, y_train,x_train_full), (x_test, y_test,x_test_full) = split_data(x_data, [y_data, x_data_full], n_train=n_train, n_valid=n_valid,seed=args.seed,split_by_set=args.split_by_set,testing_dataset=unique_location,data_subset_name_list=locations_list )

				else:
					(x_train, y_train), (x_valid,y_valid), (x_test, y_test) = split_data(x_data, y_data, n_train=n_train, n_valid=n_valid,seed=args.seed,split_by_set=args.split_by_set,testing_dataset=unique_location,data_subset_name_list=locations_list)
				
				#Set testing data to be the same as training data in the event that all training data is used, so we get error metrics on the training data
				if args.n_train == 1.0 and args.split_by_set==False and unique_location == "full_dataset_split":
					print('Setting testing data to be training data, just for plotting purposes')
					x_test = x_train
					y_test = y_train
					x_test_full = x_train_full

				get_minmax = lambda d: list(zip(np.nanmin(d, 0).round(2), np.nanmax(d, 0).round(2)))
				if n_valid != 0:
					print(f'\nShapes: x_train={x_train.shape}  x_test={x_test.shape}  x_valid={x_valid.shape}  y_train={y_train.shape} y_valid={y_valid.shape} y_test={y_test.shape}')
				else:
					print(f'\nShapes: x_train={x_train.shape}  x_test={x_test.shape}  y_train={y_train.shape}  y_test={y_test.shape}')

				print('Min/Max Train X:', get_minmax(x_train))
				print('Min/Max Train Y:', get_minmax(y_train))
				print(f'Train valid: {np.isfinite(y_train).sum(0)}')

				if n_valid != 0:
					print('Min/Max Valid X:', get_minmax(x_valid))
					print('Min/Max Valid Y:', get_minmax(y_valid))			
					print(f'Validation valid: {np.isfinite(y_valid).sum(0)}')

				if len(x_test) != 0:
					print('Min/Max Test X:', get_minmax(x_test))
					print('Min/Max Test Y:', get_minmax(y_test))
					print(f'Test valid: {np.isfinite(y_test).sum(0)}')
				print(f'Min/Max wavelength: {bands[0]}, {bands[-1]}\n')

				products   = args.product.split(',') 

				if n_valid !=0:
					print("Error reported on validation data:")

					labels     = get_labels(bands, slices, y_valid.shape[1])
					benchmarks = run_benchmarks(args.sensor, x_valid, y_valid, x_train, y_train, {p:slices[p] for p in products}, verbose=True, return_ml=False,return_opt=args.return_opt)

				else:
					print("Error reported on testing data:")

					labels     = get_labels(bands, slices, y_test.shape[1])

					benchmarks = run_benchmarks(args.sensor.split('-')[0]+'-SimisFull', x_test_full, y_test, x_train_full, y_train, {p:slices[p] for p in products}, verbose=True, return_ml=False,return_opt=args.return_opt)

				if args.using_correlations:
					args = assign_band_ratio_correlations(args)
				
				OWT_assignments = assign_spectra_OWT(z_data,z_w_data) 
				OWT_assignments,min_value,nearest_Rrs,member_w_list = assign_spectra_OWT_angle_nearest_Rrs(z_data,z_w_data) 

				plot_bar_chart(OWT_assignments)
				plot_Rrs_scatter(z_data,z_w_data,OWT_assignments,y_data)

				plot_histogram(y_data,products)
				
				if n_valid !=0:
					print("Error reported on validation data:")
					estimates, est_slice = get_estimates(args, x_train, y_train, x_valid, y_valid, slices, dataset_labels=locs[:,0])
				else:
					print("Error reported on testing data:")
					estimates, est_slice = get_estimates(args, x_train, y_train, x_test, y_test, slices, dataset_labels=locs[:,0])

				estimates = np.median(estimates, 0)
				print('Shape Estimates:', estimates.shape)
				print('Min/Max Estimates:', get_minmax(estimates), '\n')

				benchmarks['MDN'] = estimates
				if n_valid !=0:
					for p in products:
						for lbl, y1, y2 in zip(labels[slices[p]], y_valid.T[slices[p]], estimates.T[slices[p]]):
							print( performance(f'MDN {lbl}', y1, y2) ) 
					plot_scatter(y_valid, benchmarks, bands, labels, products, args.sensor,args.return_opt,run_name=args.run_name + '_' + unique_location + '_testing_data_')

				else:
					for p in products:
						for lbl, y1, y2 in zip(labels[slices[p]], y_test.T[slices[p]], estimates.T[slices[p]]):
							print( performance(f'MDN {lbl}', y1, y2) ) 
					plot_scatter(y_test, benchmarks, bands, labels, products, args.sensor,args.return_opt,run_name=args.run_name + '_' + unique_location + '_testing_data_')


				del benchmarks

				#Plots summary information for each subset during round robin training
				y_test_list_subset.append(y_test)
				estimates_list_subset.append(estimates)
				run_name_list_subset.append(unique_location)
			plot_scatter_summary(y_test_list_subset,estimates_list_subset,products='chl,PC',run_name=run_name_list_subset)

			if args.plot_matchups == 3: # PRISMA matchups, e.g. curonian lagoon
				args.no_load = False 
				args.verbose = True
				x_data_Curonia, y_data_Curonia, slices_Curonia, locs_Curonia, z_data_Curonia, z_w_data_Curonia,x_data_full_Curonia,lat_data_Curonia,lon_data_Curonia = get_data(args,return_PRISMA_testing=True)
				x_data_Curonia = np.reshape(x_data_Curonia,(1,-1,len(get_sensor_bands(args.sensor))))
				print(x_data_Curonia,y_data_Curonia)
				print(np.shape(x_data_Curonia),np.shape(y_data_Curonia))
				chl_curonia = [i[0] for i in y_data_Curonia]

				PC_curonia = [i[1] for i in y_data_Curonia]
				estimates_curonia = image_estimates(data=x_data_Curonia,args=args, sensor=args.sensor, product_name='chl,PC')
				
				print(np.shape(x_data_Curonia),np.shape(y_data_Curonia))
				x_data_Curonia, y_data_Curonia, slices_Curonia, locs_Curonia, z_data_Curonia, z_w_data_Curonia,x_data_full_Curonia,lat_data_Curonia,lon_data_Curonia = get_data(args,return_PRISMA_testing=True)
				args.sensor = 'PRISMA-SimisFull'
				x_data_full_Curonia = np.reshape(x_data_full_Curonia,(1,-1,len(get_sensor_bands(args.sensor))))
				benchmarks = run_benchmarks(args.sensor, np.squeeze(x_data_full_Curonia), None, None, None, {p:slices_Curonia[p] for p in products}, verbose=True, return_ml=False,return_opt=args.return_opt)
				
				for i in range(len(PC_curonia)):
					print('{} Truth Data chl: {} PC:{} Estimates: {}'.format(i,chl_curonia[i],PC_curonia[i],np.asarray(estimates_curonia[1][0])[i]))
				
				print('PC Keys',(benchmarks['PC'].keys()))

				print('HUNTER ESTIMATES',(benchmarks['PC']['Hunter']))
				print('Schalles ESTIMATES',(benchmarks['PC']['Schalles']))
				print('Simis2007 ESTIMATES',(benchmarks['PC']['Sim2007']))



				exit()
			if args.plot_matchups == 1:
				args.no_load = False 
				args.verbose = True
				dictionary_of_matchups = get_matchups(args.sensor)	

				chl_truths = dictionary_of_matchups['chl'].flatten()
				PC_truths =dictionary_of_matchups['PC'].flatten()
				truth_values = np.vstack((chl_truths,PC_truths)).T

				insitu_Rrs = np.reshape(dictionary_of_matchups['insitu_Rrs_resampled'],(1,-1,len(get_sensor_bands(args.sensor))))
				insitu_Rrs_squeezed = np.squeeze(insitu_Rrs)
				insitu_Rrs_Simis = np.reshape(dictionary_of_matchups['Rrs_insitu_resampled_full_NIR'],(1,-1,len(get_sensor_bands(args.sensor.split('-')[0]+'-SimisFull'))))
				insitu_Rrs_Simis_squeezed = np.squeeze(insitu_Rrs_Simis)
				remote_Rrs = np.reshape(dictionary_of_matchups['Rrs_retrieved'],(1,-1,len(get_sensor_bands(args.sensor))))
				remote_Rrs_squeezed = np.squeeze(remote_Rrs)
				valid =np.isfinite(PC_truths)

	
				benchmarks = run_benchmarks(args.sensor.split('-')[0]+'-SimisFull', (insitu_Rrs_Simis_squeezed[valid]), None, None, None, {p:slices[p] for p in products}, verbose=True, return_ml=False,return_opt=0) 
				print('PC Keys',(benchmarks['PC'].keys()))

				print('HUNTER ESTIMATES',(benchmarks['PC']['Hunter'][range(5)]))
				print('Schalles ESTIMATES',(benchmarks['PC']['Schalles'][range(5)]))
				print('Simis2007 ESTIMATES',(benchmarks['PC']['Sim2007'][range(5)]))

				print(dictionary_of_matchups['site_label'][range(5)])

				estimates_in_situ = image_estimates(data=insitu_Rrs,args=args, sensor=args.sensor, product_name='chl,PC',kwargs=dictionary_of_run_parameters)
				estimates_remote  = image_estimates(data=remote_Rrs,args=args, sensor=args.sensor, product_name='chl,PC',kwargs=dictionary_of_run_parameters)
				plot_remote_insitu(y_remote=estimates_remote, y_insitu=estimates_in_situ,dictionary_of_matchups=dictionary_of_matchups,products=['chl','PC'],sensor=args.sensor,run_name=args.run_name)
			return y_test, estimates, args.run_name,estimates_in_situ,estimates_remote,dictionary_of_matchups,args.sensor;


		# Otherwise, train a model with all data (if not already existing)
		else:
			x_data, y_data, slices, locs, z_data, z_w_data,x_data_full,lat_data,lon_data = get_data(args)
			get_estimates(args, x_data, y_data, output_slices=slices, dataset_labels=locs[:,0])

def assign_band_ratio_correlations(args):
	x_data, y_data, slices, locs, z_data, z_w_data,x_data_full,lat_data,lon_data         = get_data(args)

	n_train = args.n_train if args.dataset != 'sentinel_paper' else 1000
	n_valid = args.n_valid if args.dataset != 'sentinel_paper' else 1000
	unique_location = "full_dataset_split"

	if args.split_by_set == True:
		locations_list = assign_locations(locs)
	else:
		locations_list = []
	(x_train, y_train), (x_test, y_test) = split_data(x_data, y_data, n_train=n_train, n_valid=n_valid,seed=args.seed,split_by_set=False,testing_dataset=unique_location,data_subset_name_list=locations_list )
	spearmans_ranked_list_LH_threshold_PC_center_bandwidth = []
	spearmans_ranked_list_BR_threshold_PC_numerator_denominator = []
	spearmans_ranked_list_BR_threshold_PC_numerator_denominator,spearmans_ranked_list_LH_threshold_PC_center_bandwidth = get_band_correlations(args=args,x_data=x_data,y_data=y_data,output_slices=slices)
	args.band_ratios_thresholded = spearmans_ranked_list_BR_threshold_PC_numerator_denominator
	args.line_heights_thresholded = spearmans_ranked_list_LH_threshold_PC_center_bandwidth
	return args

def assign_locations(locs):
	dictionary_of_locations = {
		'ASDFR' : 'xManitova',
		'SpectralEvolutionNewWvl' : 'xManitova',
		'WISP' : 'xManitova',
		'SpectralEvolution' : 'xManitova',

		'Curonia' : 'Curonia and Lithuania',
		'Lithuania' : 'Curonia and Lithuania',

		'Dutch Lakes' : 'Dutch Lakes',
		'Erie' : 'Erie',
		'NOAA_combined' : 'Erie',
		'South_Africa_Matthews' : 'South Africa',
		'Indiana_U_Lin' : 'Indiana',
		'SpainRuizVerdu' : 'Spain',
		'UNL_combined' : 'AUNL',
	}
	locations_list = []
	for location_counter, location in enumerate(locs):

		locations_list.append(dictionary_of_locations[locs[location_counter][0]])
	return locations_list


def examine_outlier_estimates(truth,estimates,x_data,products,args,locs=None,z_data=None,z_w_data=None,OWT_assignments=None,min_OWT_angle = None,nearest_member_Rrs = None,member_w_list=None):
	print(locs)
	print(products)
	for counter,value in enumerate(products):
		if value == 'PC':
			print(counter)
			PC_index = counter

	truth = truth[:,PC_index]
	estimates = estimates[:,PC_index]

	error_list = []
	for i in range(len(truth)):
		error_list.append(mdsa(truth[i],estimates[i]))
	
	outliers_bool = np.asarray(error_list)>300
	outliers_list = [i for i,value in enumerate(outliers_bool) if value == True]

	n_row = 1
	n_col = 2
	fig_size = 5
	import matplotlib.pyplot as plt
	error_list_outliers = []
	min_OWT_angle_list =[]
	for i in outliers_list:
		fig, axes = plt.subplots(n_row, n_col, figsize=(fig_size*n_col, fig_size*n_row))
		axes      = [ax for axs in np.atleast_1d(axes) for ax in np.atleast_1d(axs)]
		ax = axes[0]
		ax.plot(get_sensor_bands(args.sensor),x_data[i,:].T)
		ax.plot(member_w_list[i],nearest_member_Rrs[i])

		indexable_x_data = list(np.asarray(x_data[i,:].T))
		print(indexable_x_data)

		ax.set_title('Outlier Rrs')

		ax = axes[1]
		min_OWT_angle_list.append(min_OWT_angle[i])
		error_list_outliers.append(error_list[i])
		ax.scatter(min_OWT_angle_list,error_list_outliers)

		ax.set_title('Error vs. Angle')
		ax.set_xlabel('Angle')
		ax.set_ylabel('Error')
		ax.set_xscale('Linear')
		ax.set_yscale('log')
		print('{} Truth: {} Estimates: {} OWT: {} Min Angle: {} loc: {}'.format(i,truth[i],estimates[i],OWT_assignments[i],min_OWT_angle[i],locs[i]))
		plt.show()

