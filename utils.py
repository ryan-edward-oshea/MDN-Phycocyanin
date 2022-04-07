from .transformers import CustomUnpickler, RatioTransformer, ColumnTransformer
from .meta import get_sensor_bands, SENSOR_BANDS, ANCILLARY, PERIODIC
from .parameters import update, hypers
from .__version__ import __version__

from collections import defaultdict as dd
from datetime import datetime as dt
from pathlib import Path
from tqdm import trange

import pickle as pkl
import numpy as np 
import hashlib, re, warnings, functools
from .meta import get_sensor_bands
import pandas as pd
import numpy as np

def set_kwargs_PC(sensor):

	sensor_hash = {
	'S3A-noBnoNIR' : 'c81355fa03928c9deb2e5d9519780eff42d3a961dda234a4d18327b0469843ff',
    	'S3B-noBnoNIR' : 'b03734e194de026165126378b565a9b262ac52625bfb84bd4be0d19d7a1b6313',
	'HICO-noBnoNIR' : '08b2bce6e53b12d544b2424854430499db38c468224477bdecf247a67cc7d77f',
	'PRISMA-noBnoNIR' : 'e9249d7825c92d34b91c21af17e4c58ddd1576ad9746205129c0a0d46833d3f0',
	}

	band_ratios_thresholded = {
	'HICO-noBnoNIR' : [[507, 501], [512, 501], [512, 507], [518, 501], [518, 507], [518, 512], [524, 501], [524, 507], [524, 512], [524, 518], [530, 501], [530, 507], [530, 512], [530, 518], [530, 524], [535, 501], [535, 507], [535, 512], [535, 518], [535, 524], [535, 530], [541, 501], [541, 507], [541, 512], [541, 518], [541, 524], [541, 530], [541, 535], [547, 501], [547, 507], [547, 512], [547, 518], [547, 524], [547, 530], [547, 535], [547, 541], [553, 501], [553, 507], [553, 512], [553, 518], [553, 524], [553, 530], [553, 535], [553, 541], [558, 501], [558, 507], [558, 512], [558, 518], [558, 524], [558, 530], [558, 535], [564, 501], [564, 507], [564, 512], [564, 518], [564, 524], [564, 530], [570, 501], [570, 507], [570, 512], [570, 518], [570, 524], [575, 501], [575, 507], [575, 512], [575, 518], [575, 524], [581, 501], [581, 507], [581, 512], [581, 518], [587, 501], [587, 507], [587, 512], [587, 518], [593, 501], [593, 507], [593, 512], [593, 518], [598, 501], [598, 507], [598, 512], [604, 501], [604, 507], [604, 512], [610, 501], [610, 507], [616, 501], [616, 507], [616, 604], [616, 610], [621, 501], [621, 604], [621, 610], [621, 616], [627, 501], [633, 501], [633, 507], [633, 621], [633, 627], [638, 501], [638, 507], [638, 512], [638, 610], [638, 616], [638, 621], [638, 627], [638, 633], [644, 501], [644, 507], [644, 512], [644, 518], [644, 593], [644, 598], [644, 604], [644, 610], [644, 616], [644, 621], [644, 627], [644, 633], [644, 638], [650, 501], [650, 507], [650, 512], [650, 518], [650, 524], [650, 581], [650, 587], [650, 593], [650, 598], [650, 604], [650, 610], [650, 616], [650, 621], [650, 627], [650, 633], [650, 638], [650, 644], [656, 501], [656, 507], [656, 512], [656, 518], [656, 524], [656, 581], [656, 587], [656, 593], [656, 598], [656, 604], [656, 610], [656, 616], [656, 621], [656, 627], [656, 633], [656, 638], [656, 644], [661, 501], [661, 507], [661, 512], [661, 518], [661, 524], [661, 587], [661, 593], [661, 598], [661, 604], [661, 610], [661, 616], [661, 621], [661, 627], [661, 633], [661, 638], [661, 656], [667, 501], [667, 507], [667, 512], [667, 621], [667, 627], [667, 650], [667, 656], [667, 661], [673, 501], [673, 507], [673, 644], [673, 650], [673, 656], [673, 661], [673, 667], [679, 501], [679, 507], [679, 650], [679, 656], [679, 661], [684, 501], [684, 507], [684, 512], [684, 518], [684, 587], [684, 593], [684, 598], [684, 604], [684, 610], [684, 616], [684, 621], [684, 627], [684, 633], [684, 638], [684, 667], [684, 673], [684, 679], [690, 501], [690, 507], [690, 512], [690, 518], [690, 524], [690, 530], [690, 535], [690, 541], [690, 547], [690, 553], [690, 558], [690, 564], [690, 570], [690, 575], [690, 581], [690, 587], [690, 593], [690, 598], [690, 604], [690, 610], [690, 616], [690, 621], [690, 627], [690, 633], [690, 638], [690, 644], [690, 650], [690, 656], [690, 661], [690, 667], [690, 673], [690, 679], [690, 684], [696, 501], [696, 507], [696, 512], [696, 518], [696, 524], [696, 530], [696, 535], [696, 541], [696, 547], [696, 553], [696, 558], [696, 564], [696, 570], [696, 575], [696, 581], [696, 587], [696, 593], [696, 598], [696, 604], [696, 610], [696, 616], [696, 621], [696, 627], [696, 633], [696, 638], [696, 644], [696, 650], [696, 656], [696, 661], [696, 667], [696, 673], [696, 679], [696, 684], [696, 690], [701, 501], [701, 507], [701, 512], [701, 518], [701, 524], [701, 530], [701, 535], [701, 541], [701, 547], [701, 553], [701, 558], [701, 564], [701, 570], [701, 575], [701, 581], [701, 587], [701, 593], [701, 598], [701, 604], [701, 610], [701, 616], [701, 621], [701, 627], [701, 633], [701, 638], [701, 644], [701, 650], [701, 656], [701, 661], [701, 667], [701, 673], [701, 679], [701, 684], [701, 690], [701, 696], [707, 501], [707, 507], [707, 512], [707, 518], [707, 524], [707, 530], [707, 535], [707, 541], [707, 547], [707, 553], [707, 558], [707, 564], [707, 570], [707, 575], [707, 581], [707, 587], [707, 593], [707, 598], [707, 604], [707, 610], [707, 616], [707, 621], [707, 627], [707, 633], [707, 638], [707, 644], [707, 650], [707, 656], [707, 661], [707, 667], [707, 673], [707, 679], [707, 684], [707, 690], [707, 696], [707, 701], [713, 501], [713, 507], [713, 512], [713, 518], [713, 524], [713, 530], [713, 535], [713, 541], [713, 547], [713, 553], [713, 558], [713, 564], [713, 570], [713, 575], [713, 581], [713, 587], [713, 593], [713, 598], [713, 604], [713, 610], [713, 616], [713, 621], [713, 627], [713, 633], [713, 638], [713, 644], [713, 650], [713, 656], [713, 661], [713, 667], [713, 673], [713, 679], [713, 684], [713, 690], [713, 696], [713, 701], [713, 707], [719, 501], [719, 507], [719, 512], [719, 518], [719, 524], [719, 530], [719, 535], [719, 541], [719, 547], [719, 553], [719, 558], [719, 564], [719, 570], [719, 575], [719, 581], [719, 587], [719, 593], [719, 598], [719, 604], [719, 610], [719, 616], [719, 621], [719, 627], [719, 633], [719, 638], [719, 644], [719, 650], [719, 656], [719, 661], [719, 667], [719, 673], [719, 679], [719, 684], [719, 690], [719, 696], [719, 701], [719, 707], [719, 713], [724, 501], [724, 507], [724, 512], [724, 518], [724, 524], [724, 530], [724, 535], [724, 541], [724, 547], [724, 553], [724, 558], [724, 564], [724, 570], [724, 575], [724, 581], [724, 587], [724, 593], [724, 598], [724, 604], [724, 610], [724, 616], [724, 621], [724, 627], [724, 633], [724, 638], [724, 644], [724, 650], [724, 656], [724, 661], [724, 667], [724, 673], [724, 679], [724, 684], [724, 690], [724, 696], [724, 701], [724, 707]],
	'PRISMA-noBnoNIR' : [[507, 500], [515, 500], [515, 507], [523, 500], [523, 507], [523, 515], [530, 500], [530, 507], [530, 515], [530, 523], [538, 500], [538, 507], [538, 515], [538, 523], [538, 530], [546, 500], [546, 507], [546, 515], [546, 523], [546, 530], [546, 538], [554, 500], [554, 507], [554, 515], [554, 523], [554, 530], [554, 538], [563, 500], [563, 507], [563, 515], [563, 523], [563, 530], [571, 500], [571, 507], [571, 515], [571, 523], [579, 500], [579, 507], [579, 515], [579, 523], [588, 500], [588, 507], [588, 515], [596, 500], [596, 507], [596, 515], [605, 500], [605, 507], [605, 515], [614, 500], [614, 507], [614, 605], [623, 500], [623, 605], [623, 614], [632, 500], [632, 507], [632, 623], [641, 500], [641, 507], [641, 515], [641, 605], [641, 614], [641, 623], [641, 632], [651, 500], [651, 507], [651, 515], [651, 523], [651, 588], [651, 596], [651, 605], [651, 614], [651, 623], [651, 632], [651, 641], [660, 500], [660, 507], [660, 515], [660, 523], [660, 588], [660, 596], [660, 605], [660, 614], [660, 623], [660, 632], [660, 641], [670, 500], [670, 507], [670, 515], [670, 651], [670, 660], [679, 500], [679, 507], [679, 651], [679, 660], [689, 500], [689, 507], [689, 515], [689, 523], [689, 530], [689, 538], [689, 546], [689, 554], [689, 563], [689, 571], [689, 579], [689, 588], [689, 596], [689, 605], [689, 614], [689, 623], [689, 632], [689, 641], [689, 651], [689, 660], [689, 670], [689, 679], [699, 500], [699, 507], [699, 515], [699, 523], [699, 530], [699, 538], [699, 546], [699, 554], [699, 563], [699, 571], [699, 579], [699, 588], [699, 596], [699, 605], [699, 614], [699, 623], [699, 632], [699, 641], [699, 651], [699, 660], [699, 670], [699, 679], [699, 689], [709, 500], [709, 507], [709, 515], [709, 523], [709, 530], [709, 538], [709, 546], [709, 554], [709, 563], [709, 571], [709, 579], [709, 588], [709, 596], [709, 605], [709, 614], [709, 623], [709, 632], [709, 641], [709, 651], [709, 660], [709, 670], [709, 679], [709, 689], [709, 699], [719, 500], [719, 507], [719, 515], [719, 523], [719, 530], [719, 538], [719, 546], [719, 554], [719, 563], [719, 571], [719, 579], [719, 588], [719, 596], [719, 605], [719, 614], [719, 623], [719, 632], [719, 641], [719, 651], [719, 660], [719, 670], [719, 679], [719, 689], [719, 699], [719, 709]],
	
	'S3A-noBnoNIR' : [[560, 510], [664, 510], [664, 619], [673, 664], [681, 510], [681, 664], [681, 673], [708, 510], [708, 560], [708, 619], [708, 664], [708, 673], [708, 681]],
	'S3B-noBnoNIR' : [[560, 510], [664, 510], [664, 619], [673, 664], [681, 510], [681, 664], [681, 673], [708, 510], [708, 560], [708, 619], [708, 664], [708, 673], [708, 681]],
	}

	line_heights_thresholded = {
	'HICO-noBnoNIR' : [[507, 5], [512, 10], [518, 15], [535, 5], [535, 10], [541, 5], [541, 10], [541, 15], [541, 20], [547, 5], [547, 10], [547, 15], [547, 20], [553, 5], [553, 10], [553, 15], [553, 20], [558, 5], [558, 10], [558, 15], [558, 20], [575, 5], [575, 10], [581, 5], [581, 10], [581, 15], [604, 5], [616, 5], [616, 10], [616, 15], [616, 20], [621, 5], [621, 10], [621, 15], [621, 20], [627, 5], [627, 10], [627, 15], [627, 20], [633, 5], [633, 10], [633, 15], [633, 20], [638, 15], [638, 20], [644, 10], [644, 15], [644, 20], [650, 5], [650, 10], [650, 15], [650, 20], [656, 5], [656, 10], [656, 15], [656, 20], [661, 10], [661, 15], [661, 20], [667, 5], [667, 10], [667, 15], [667, 20], [673, 5], [673, 10], [673, 15], [673, 20], [679, 5], [679, 10], [679, 15], [679, 20], [684, 5], [684, 10], [684, 15], [684, 20], [690, 5], [690, 10], [690, 15], [690, 20], [696, 10], [696, 15], [696, 20], [701, 5], [701, 10], [701, 15], [701, 20], [707, 5], [707, 10], [707, 15], [713, 5], [713, 10], [719, 5]],
	'PRISMA-noBnoNIR' : [[507, 5], [515, 5], [515, 10], [515, 15], [523, 5], [523, 10], [538, 15], [546, 5], [546, 10], [546, 15], [546, 20], [554, 5], [554, 10], [554, 15], [554, 20], [563, 5], [563, 10], [563, 15], [563, 20], [579, 5], [579, 10], [579, 15], [579, 20], [605, 5], [605, 10], [623, 5], [623, 10], [623, 15], [623, 20], [632, 5], [632, 10], [632, 15], [632, 20], [651, 5], [651, 10], [651, 15], [651, 20], [660, 10], [660, 15], [660, 20], [670, 5], [670, 10], [670, 15], [670, 20], [679, 10], [679, 15], [679, 20], [689, 10], [689, 15], [689, 20], [699, 10], [699, 15], [699, 20], [709, 10]],
	'S3A-noBnoNIR' : [[673, 5], [673, 10]],
	'S3B-noBnoNIR' : [[673, 5], [673, 10]]
	
	}

	kwargs_PC = {
	'sensor' : sensor,
	'model_hash' : sensor_hash[sensor],
	'band_ratios_thresholded' : band_ratios_thresholded[sensor],
	'line_heights_thresholded' : line_heights_thresholded[sensor],
	}

	return kwargs_PC

def ignore_warnings(func):
	''' Decorator to silence warnings (Runtime, User, Deprecation, etc.) '''
	@functools.wraps(func)
	def helper(*args, **kwargs):
		with warnings.catch_warnings():
			warnings.filterwarnings('ignore') 
			return func(*args, **kwargs)
	return helper 


def find_wavelength(k, waves, validate=True, tol=5):
	''' Index of closest wavelength '''
	i = np.abs(np.array(waves) - k).argmin() 
	assert(not validate or (abs(k-waves[i]) <= tol)), f'Needed {k}nm, but closest was {waves[i]}nm in {waves}'
	return i 


def closest_wavelength(k, waves, validate=True, tol=5): 
	''' Value of closest wavelength '''
	return waves[find_wavelength(k, waves, validate, tol)]	


def safe_int(v):
	''' Parse int if possible, and return None otherwise '''
	try: return int(v)
	except: return None


def get_wvl(nc_data, key):
	''' Get all wavelengths associated with the given key, available within the netcdf '''
	wvl = [safe_int(v.replace(key, '')) for v in nc_data.variables.keys() if key in v]
	return np.array(sorted([w for w in wvl if w is not None]))


def line_messages(messages):
	''' 
	Allow multiline message updates via tqdm. 
	Need to call print() after the tqdm loop, 
	equal to the number of messages which were
	printed via this function (to reset cursor).
	
	Usage:
		for i in trange(5):
			messages = [i, i/2, i*2]
			line_messages(messages)
		for _ in range(len(messages)): print()
	'''
	for i, m in enumerate(messages, 1):
		trange(1, desc=str(m), position=i, bar_format='{desc}')


def get_labels(wavelengths, slices, n_out=None):
	''' 
	Helper to get label for each target output. Assumes 
	that any variable in <slices> which has more than a 
	single slice index, will have an associated wavelength
	label. 

	Usage:
		wavelengths = [443, 483, 561, 655]
		slices = {'bbp':slice(0,4), 'chl':slice(4,5), 'tss':slice(5,6)}
		n_out  = 5
		labels = get_labels(wavelengths, slices, n_out) 
			# labels -> ['bbp443', 'bbp483', 'bbp561', 'bbp655', 'chl']
	'''
	return [k + (f'{wavelengths[i]:.0f}' if (v.stop - v.start) > 1 else '') 
			for k,v in sorted(slices.items(), key=lambda s: s[1].start)
			for i   in range(v.stop - v.start)][:n_out]	


def store_pkl(filename, output):
	''' Helper to write pickle file '''
	with Path(filename).open('wb') as f:
		pkl.dump(output, f)
	return output

def read_pkl(filename):
	''' Helper to read pickle file '''
	with Path(filename).open('rb') as f:
		return CustomUnpickler(f).load()

def cache(filename, recache=False):
	''' Decorator for caching function outputs '''
	path = Path(filename)

	def wrapper(function):
		def inner(*args, **kwargs):
			if not recache and path.exists():
				return read_pkl(path)
			return store_pkl(path, function(*args, **kwargs))
		return inner
	return wrapper


def using_feature(args, flag):
	''' 
	Certain hyperparameter flags have a yet undecided default value,
	which means there are two possible names: using the feature, or 
	not using it. This method simply combines both into a single 
	boolean signal, which indicates whether to add the feature. 
	For example:
	 	use_flag = hasattr(args, 'use_ratio') and args.use_ratio
		no_flag  = hasattr(args, 'no_ratio') and not args.no_ratio 
		signal   = use_flag or no_flag  # if true, we add ratios
	becomes
		signal = using_feature(args, 'ratio') # if true, we add ratios  
	'''
	
	assert(hasattr(args,f'use_{flag}') or hasattr(args, f'no_{flag}')), f'"{flag}" flag not found'
	return getattr(args, f'use_{flag}', False) or not getattr(args, f'no_{flag}', True)


def split_data(x_data, other_data=[], n_train=0.5, n_valid=0, seed=None, shuffle=True,split_by_set=False,testing_dataset=None,data_subset_name_list=None):
	''' 
	Split the given data into training, validation, and testing 
	subsets, randomly shuffling the original data order.
	'''
	if not isinstance(other_data, list): other_data = [other_data]
	data    = [d.iloc if hasattr(d, 'iloc') else d for d in [x_data] + other_data]
	if split_by_set:
		testing_indices = [i for i, current_subset_name in enumerate(data_subset_name_list) if current_subset_name == testing_dataset]
		training_indices = [i for i, current_subset_name in enumerate(data_subset_name_list) if current_subset_name != testing_dataset]

		test = [d[testing_indices] for d in data]
		train = [d[training_indices] for d in data]

		return train, test
	else:
		random  = np.random.RandomState(seed)
		idxs    = np.arange(len(x_data))
		if shuffle: random.shuffle(idxs)

		# Allow both a percent to be passed in, as well as an absolute number
		if 0 < n_train <= 1: n_train = int(n_train * len(idxs)) 
		if 0 < n_valid <= 1: n_valid = int(n_valid * len(idxs))
		assert((n_train+n_valid) <= len(x_data)), \
			'Too many training/validation samples requested: {n_train}, {n_valid} ({len(x_data)} available)'

		train = [d[ idxs[:n_train] ]                for d in data]
		valid = [d[ idxs[n_train:n_valid+n_train] ] for d in data]
		test  = [d[ idxs[n_train+n_valid:] ]        for d in data]

		# Return just the split x_data if no other data was given
		if len(data) == 1:
			train = train[0]
			valid = valid[0]
			test  = test[0]

		# If no validation data was requested, just return train/test
		if n_valid == 0:
			return train, test 
		return train, valid, test

@ignore_warnings
def mask_land(data, bands, threshold=0.2, verbose=False):
	''' Modified Normalized Difference Water Index, or NDVI if 1500nm+ is not available '''
	green = closest_wavelength(560,  bands, validate=False)
	red   = closest_wavelength(700,  bands, validate=False)
	nir   = closest_wavelength(900,  bands, validate=False)
	swir  = closest_wavelength(1600, bands, validate=False)
	
	b1, b2 = (green, swir) if swir > 1500 else (red, nir)
	i1, i2 = find_wavelength(b1, bands), find_wavelength(b2, bands)
	n_diff = lambda a, b: np.ma.masked_invalid((a-b) / (a+b))
	if verbose: print(f'Using bands {b1} & {b2} for land masking')
	return n_diff(data[..., i1], data[..., i2]).filled(fill_value=threshold-1) <= threshold


@ignore_warnings
def _get_tile_wavelengths(nc_data, key, sensor, allow_neg=True, landmask=False):
	''' Return the Rrs/rhos data within the netcdf file, for wavelengths of the given sensor '''
	has_key = lambda k: any([k in v for v in nc_data.variables])
	wvl_key = f'{key}_' if has_key(f'{key}_') or key != 'Rrs' else 'Rw' # Polymer stores Rw=Rrs*pi

	if has_key(wvl_key):
		avail = get_wvl(nc_data, wvl_key)
		bands = [closest_wavelength(b, avail) for b in get_sensor_bands(sensor)]
		div   = np.pi if wvl_key == 'Rw' else 1
		data  = np.ma.stack([nc_data[f'{wvl_key}{b}'][:] / div for b in bands], axis=-1)
		
		if not allow_neg: data[data <= 0] = np.nan
		if landmask:      data[ mask_land(data, bands) ] = np.nan

		return bands, data.filled(fill_value=np.nan)
	return [], np.array([])

def get_tile_data(filenames, sensor, allow_neg=True, rhos=False, anc=False):
	''' Gather the correct Rrs/rhos bands from a given scene, as well as ancillary features if necessary '''
	from netCDF4 import Dataset

	filenames = np.atleast_1d(filenames) 
	features  = ['rhos' if rhos else 'Rrs'] + (ANCILLARY if anc or rhos else [])
	data      = {}
	available = []

	# Some sensors use different bands for their rhos models 
	if rhos and '-rho' not in sensor: sensor += '-rho'

	for filename in filenames:
		with Dataset(filename, 'r') as nc_data:
			if 'geophysical_data' in nc_data.groups.keys():
				nc_data = nc_data['geophysical_data']
	
			for feature in features:
				if feature not in data:
					if feature in ['Rrs', 'rhos']:
						bands, band_data = _get_tile_wavelengths(nc_data, feature, sensor, allow_neg, landmask=rhos)
	
						if len(bands) > 0: 
							assert(len(band_data.shape) == 3), \
								f'Different shape than expected: {band_data.shape}'
							data[feature] = band_data
	
					elif feature in nc_data.variables:
						var = nc_data[feature][:]
						assert(len(var.shape) == 2), f'Different shape than expected: {var.shape}'
	
						if feature in PERIODIC:
							assert(var.min() >= -180 and var.max() <= 180), \
								f'Need to adjust transformation for variables not within [-180,180]: {feature}=[{var.min()}, {var.max()}]'
							data[feature] = np.stack([
								np.sin(2*np.pi*(var+180)/360),
								np.cos(2*np.pi*(var+180)/360),
							], axis=-1)
						else: data[feature] = var
	
	# Time difference should just be 0: we want estimates for the exact time of overpass
	if 'time_diff' in features:
		assert(features[0] in data), f'Missing {features[0]} data: {list(data.keys())}'
		data['time_diff'] = np.zeros_like(data[features[0]][:, :, 0])

	assert(len(data) == len(features)), f'Missing features: Found {list(data.keys())}, Expecting {features}'
	return bands, np.dstack([data[f] for f in features])


def generate_config(args, create=True, verbose=True):
	''' 
	Create a config file for the current settings, and store in
	a folder location determined by certain parameters: 
		MDN/model_loc/sensor/model_lbl/model_hash/config
	"model_hash" is computed within this function, but a value can 
	also be passed in manually via args.model_hash in order to allow
	previous MDN versions to run.
	'''
	root = Path(__file__).parent.resolve().joinpath(args.model_loc, args.sensor, args.model_lbl)

	# Can override the model hash in order to allow prior MDN versions to be run
	if hasattr(args, 'model_hash'):
		if args.verbose: print(f'Using manually set model hash: {args.model_hash}')
		return root.joinpath(args.model_hash)

	dependents = [getattr(act, 'dest', '') for group in [hypers, update] for act in group._group_actions]
	dependents+= ['x_scalers', 'y_scalers']

	config = [f'Version: {__version__}']
	config+= [''.join(['-']*len(config[-1]))]
	others = [''.join(['-']*len(config[-1]))]

	for k,v in sorted(args.__dict__.items(), key=lambda z: z[0]):
		if k in ['x_scalers', 'y_scalers']:
			v = [(s[0].__name__,)+s[1:] for s in v] # stringify scaler and its arguments

		if k in dependents: config.append(f'{k}: {v}') 
		else:               others.append(f'{k}: {v}') 
				
	config = '\n'.join(config) # Model is dependent on some arguments, so they change the hash
	others = '\n'.join(others) # Other arguments are stored for replicability
	ver_re = r'(Version\: \d+\.\d+)(?:\.\d+\n[-]+\n)' # Match major/minor version within subgroup, patch/dashes within pattern
	h_str  = re.sub(ver_re, r'\1.0\n', config)        # Substitute patch version for ".0" to allow patches within the same hash
	uid    = hashlib.sha256(h_str.encode('utf-8')).hexdigest()
	folder = root.joinpath(uid)
	c_file = folder.joinpath('config')

	if args.verbose: 
		print(f'Using model path {folder}')

	if create:
		folder.mkdir(parents=True, exist_ok=True)
		
		if not c_file.exists():
			with c_file.open('w+') as f:
				f.write(f'Created: {dt.now()}\n{config}\n{others}')
	elif not c_file.exists() and verbose:
		print('\nCould not find config file with the following parameters:')
		print('\t'+config.replace('\n','\n\t'),'\n')
	return folder 


def _load_datasets(keys, locs, wavelengths, allow_missing=False,args=None,load_lat_lon=False):
	''' 
	Load data from [<locs>] using <keys> as the columns. 
	Only loads data which has all the bands defined by 
	<wavelengths> (if necessary, e.g. for Rrs or bbp).
	First key is assumed to be the x_data, remaining keys
	(if any) are y_data.
	  - allow_missing=True will allow datasets which are missing bands
	    to be included in the returned data

	Usage:
		# Here, data/loc/Rrs.csv, data/loc/Rrs_wvl.csv, data/loc/bbp.csv, 
		# and data/chl.csv all exist, with the correct wavelengths available
		# for Rrs and bbp (which is determined by Rrs_wvl.csv)
		keys = ['Rrs', 'bbp', '../chl']
		locs = 'data/loc'
		wavelengths = [443, 483, 561, 655]
		_load_datasets(keys, locs, wavelengths) # -> [Rrs443, Rrs483, Rrs561, Rrs665], 
												 [bbp443, bbp483, bbp561, bbp655, chl], 
											 	 {'bbp':slice(0,4), 'chl':slice(4,5)}
	'''
	def loadtxt(name, loc, _wavelengths): 
		''' Error handling wrapper over np.loadtxt, with the addition of wavelength selection'''

		#If the key is Rrs_full_wavelengths, the name is reset to Rrs
		# print(name,loc,_wavelengths)
		if name == 'Rrs_full_wavelengths':
			name = 'Rrs'
			return_full_wavelength = True
		else:
			return_full_wavelength = False

		dloc = Path(loc).joinpath(f'{name}.csv')

		# TSS / TSM are synonymous
		if 'tss' in name and not dloc.exists():
			dloc = Path(loc).joinpath(f'{name.replace("tss","tsm")}.csv')

		# CDOM is just an alias for a_cdom(443)
		if 'cdom' in name and not dloc.exists():
			dloc = Path(loc).joinpath('ag.csv')

		try:
			assert(dloc.exists()), (f'Key {name} does not exist at {loc} ({dloc})') 
			data = np.loadtxt(dloc, delimiter=',', dtype=float if name not in ['../Dataset', '../meta'] else str, comments=None)

			if len(data.shape) == 1:
				data = data[:, None]

			if data.shape[1] > 1 and data.dtype.type is not np.str_:
				# print('Within IF')

				# If we want to get all data, regardless of if bands are available...
				if allow_missing:
					new_data = [[np.nan]*len(data)] * len(_wavelengths)
					wvls  = np.loadtxt(Path(loc).joinpath(f'{name}_wvl.csv'), delimiter=',')[:,None]
					idxs  = np.abs(wvls - np.atleast_2d(_wavelengths)).argmin(0)
					valid = np.abs(wvls - np.atleast_2d(_wavelengths)).min(0) < 2

					for j, (i, v) in enumerate(zip(idxs, valid)):
						if v: new_data[j] = data[:, i]
					data = np.array(new_data).T
				else:
					if str(name) == "../Rrs_OG" or str(name) == "../Rrs_OG_wvl":
						pad_size = 1000-np.shape(data)[1]
						data = np.pad(data,[(0,0),(0,pad_size)],'constant',constant_values=(np.nan,np.nan))
					else:
						if return_full_wavelength:
							sensor   = args.sensor.split('-')[0]
							bands    = get_sensor_bands(sensor+'-SimisFull', args)
							data = data[:, get_valid_full(name, loc, _wavelengths,bands)]
						else:
							data = data[:, get_valid(name, loc, _wavelengths)]
						
			if 'cdom' in name and dloc.stem == 'ag':
				data = data[:, find_wavelength(443, np.loadtxt(Path(loc).joinpath(f'{dloc.stem}_wvl.csv'), delimiter=','))].flatten()[:, None]
			return data 
		except Exception as e:
			if dloc.exists():
				print(f'Error fetching {name} from {loc}: {e}')
			if name not in ['Rrs']:# ['../chl', '../tss', '../cdom']:
				return np.array([]).reshape((0,0))
			assert(0), e

	def get_valid(name, loc, _wavelengths, margin=2):
		''' Dataset at <loc> must have all bands in <_wavelengths> within <margin>nm '''
		if 'HYPER' in str(loc): margin=1

		wvls = np.loadtxt(Path(loc).joinpath(f'{name}_wvl.csv'), delimiter=',')[:,None]
		assert(np.all([np.abs(wvls-w).min() <= margin for w in _wavelengths])), (
			f'{loc} is missing wavelengths: \n{_wavelengths} needed,\n{wvls.flatten()} found')
		
		if len(wvls) != len(_wavelengths):
			valid = np.abs(wvls - np.atleast_2d(_wavelengths)).min(1) < margin
			assert(valid.sum() == len(_wavelengths)), [wvls[valid].flatten(), _wavelengths]
			return valid 
		return np.array([True] * len(_wavelengths))

	def get_valid_full(name, loc, _wavelengths, wavelengths_full, margin=2):
		''' Dataset at <loc> must have all bands in <_wavelengths> within <margin>nm '''
		if 'HYPER' in str(loc): margin=1

		wvls = np.loadtxt(Path(loc).joinpath(f'{name}_wvl.csv'), delimiter=',')[:,None]
		assert(np.all([np.abs(wvls-w).min() <= margin for w in _wavelengths])), (
			f'{loc} is missing wavelengths: \n{_wavelengths} needed,\n{wvls.flatten()} found')
		
		if len(wvls) != len(wavelengths_full):
			valid = np.abs(wvls - np.atleast_2d(wavelengths_full)).min(1) < margin
			assert(valid.sum() == len(wavelengths_full)), [wvls[valid].flatten(), _wavelengths]
			return valid 
		return np.array([True] * len(wavelengths_full))


	print('\n-------------------------')
	print(f'Loading data for sensor {locs[0].parts[-1]}')
	if allow_missing:
		print('Allowing data regardless of whether all bands exist')

	x_data = []
	y_data = []
	l_data = []
	z_data = [] 
	z_w_data = [] 
	x_data_full = [] 
	lat_data = []
	lon_data = []
	for loc in np.atleast_1d(locs):
		try:
			loc_data = [loadtxt(key, loc, wavelengths) for key in keys]
			print(f'\tN={len(loc_data[0]):>5} | {loc.parts[-1]} / {loc.parts[-2]} ({[np.isfinite(ld).all(1).sum() for ld in loc_data[1:]]})')
			assert(all([len(l) in [len(loc_data[0]), 0] for l in loc_data])), dict(zip(keys, map(np.shape, loc_data)))

			if all([l.shape[1] == 0 for l in loc_data[1:]]):
				print(f'Skipping dataset {loc}: missing all features')
				continu

			x_data  += [loc_data.pop(0)] 
			z_data  += [loc_data.pop(0)] 
			z_w_data  += [loc_data.pop(0)] 
			x_data_full  += [loc_data.pop(0)] 
			if load_lat_lon:
				lat_data += [loc_data.pop(0)]
				lon_data += [loc_data.pop(0)]
				starting_len = 6
			else:
				starting_len = 4

			y_data  += [loc_data]
			l_data  += list(zip([loc.parent.name] * len(x_data[-1]), np.arange(len(x_data[-1]))))
			
		except Exception as e:
			print(f'Error {loc}: {e}')
			if len(np.atleast_1d(locs)) == 1:
				raise e
	assert(len(x_data) > 0 or len(locs) == 0), 'No datasets are valid with the given wavelengths'
	assert(all([x.shape[1] == x_data[0].shape[1] for x in x_data])), f'Differing number of {keys[0]} wavelengths: {[x.shape for x in x_data]}'

	# Determine the number of features each key should have
	slices = []
	for i, key in enumerate(keys[starting_len:]): 
		print('KEYS',key)
		shapes = [y[i].shape[1] for y in y_data]
		slices.append(max(shapes))

		for x, y in zip(x_data, y_data):
			if y[i].shape[1] == 0:
				y[i] = np.full((x.shape[0], max(shapes)), np.nan)
		assert(all([y[i].shape[1] == y_data[0][i].shape[1] for y in y_data])), f'{key} shape mismatch: {[y.shape for y in y_data]}'

	# Drop any missing features
	drop = []
	for i, s in enumerate(slices):
		if s == 0:
			print(f'Dropping {keys[i+1]}: feature has no samples available')
			drop.append(i)

	slices = np.cumsum([0] + [s for i,s in enumerate(slices) if i not in drop])
	keys   = [k for i,k in enumerate(keys[starting_len:]) if i not in drop]
	for y in y_data:
		y = [z for i,z in enumerate(y) if i not in drop]

	# Combine everything together
	l_data = np.vstack(l_data)
	x_data = np.vstack(x_data)
	y_data = np.vstack([np.hstack(y) for y in y_data])
	z_data = np.vstack(z_data)
	x_data_full = np.vstack(x_data_full)
	z_w_data = np.vstack(z_w_data)
	if load_lat_lon:
		lon_data = np.vstack(lon_data)
		lat_data = np.vstack(lat_data)
	assert(slices[-1] == y_data.shape[1]), [slices, y_data.shape]
	assert(y_data.shape[0] == x_data.shape[0]), [x_data.shape, y_data.shape]
	slices = {k.replace('../','') : slice(slices[i], s) for i,(k,s) in enumerate(zip(keys, slices[1:]))}
	print(f'\tTotal prior to filtering: {len(x_data)}')

	# Fit exponential function to ad and ag values, and eliminate samples with too much error
	for product in ['ad', 'ag']:
		if product in slices:
			from .metrics import mdsa
			from scipy.optimize import curve_fit

			exponential = lambda x, a, b, c: a * np.exp(-b*x) + c 
			remove      = np.zeros_like(y_data[:,0]).astype(bool)

			for i, sample in enumerate(y_data):
				sample = sample[slices[product]]
				assert(len(sample) > 5), f'Number of bands should be larger, when fitting exponential: {product}, {sample.shape}'
				assert(len(sample) == len(wavelengths)), f'Sample size / wavelengths mismatch: {len(sample)} vs {len(wavelengths)}'
				
				if np.all(np.isfinite(sample)) and np.min(sample) > -0.1:
					try:
						x = np.array(wavelengths) - np.min(wavelengths)
						params, _  = curve_fit(exponential, x, sample, bounds=((1e-3, 1e-3, 0), (1e2, 1e0, 1e1)))
						new_sample = exponential(x, *params)

						# Should be < 10% error between original and fitted exponential 
						if mdsa(sample[None,:], new_sample[None,:]) < 10:
							y_data[i, slices[product]] = new_sample
						else: remove[i] = True # Exponential could be fit, but error was too high
					except:   remove[i] = True # Sample deviated so much from a smooth exponential decay that it could not be fit
				# else:         remove[i] = True # NaNs / negatives in the sample

			# Don't actually drop them yet, in case we are fetching all samples regardless of nan composition
			x_data[remove] = np.nan
			y_data[remove] = np.nan
			l_data[remove] = np.nan

			if remove.sum():
				print(f'Removed {remove.sum()} / {len(remove)} samples due to poor quality {product} spectra')
				assert((~remove).sum()), f'All data removed due to {product} spectra quality...'

	return x_data, y_data, slices, l_data, z_data, z_w_data,x_data_full, lat_data, lon_data


def _filter_invalid(x_data, y_data, slices, allow_nan_inp=False, allow_nan_out=False, other=[],z_data=[],z_w_data=[],x_data_full=[],lat_data=[],lon_data=[],load_lat_lon=False):
	''' 
	Filter the given data to only include samples which are valid. By 
	default, valid samples include all which are not nan, and greater 
	than zero (for all target features). 
	- allow_nan_inp=True can be set to allow a sample as valid if _any_ 
	  of a sample's input x features are not nan and greater than zero.
	- allow_nan_out=True can be set to allow a sample as valid if _any_ 
	  of a sample's target y features are not nan and greater than zero.
	- "other" is an optional set of parameters which will be pruned with the 
	  test sets (i.e. passing a list of indices will return the indices which
	  were kept)
	Multiple data sets can also be passed simultaneously as a list to the 
	respective parameters, in order to filter the same samples out of all
	data sets (e.g. OLI and S2B data, containing same samples but different
	bands, can be filtered so they end up with the same samples relative to
	each other).
	'''
	
	# Allow multiple sets to be given, and align them all to the same sample subset
	if type(x_data) is not list: x_data = [x_data]
	if type(y_data) is not list: y_data = [y_data]
	if type(z_data) is not list: z_data = [z_data]
	if type(z_w_data) is not list: z_w_data = [z_w_data]
	if type(x_data_full) is not list: x_data_full = [x_data_full]
	if type(lat_data) is not list: lat_data = [lat_data]
	if type(lon_data) is not list: lon_data = [lon_data]


	if type(other)  is not list: other  = [other]

	both_data  = [x_data, y_data]
	set_length = [len(fullset) for fullset in both_data]
	set_shape  = [[len(subset) for subset in fullset] for fullset in both_data]
	assert(np.all([length == len(x_data) for length in set_length])), \
		f'Mismatching number of subsets: {set_length}'
	assert(np.all([[shape == len(fullset[0]) for shape in shapes] 
					for shapes, fullset in zip(set_shape, both_data)])), \
		f'Mismatching number of samples: {set_shape}'		
	assert(len(other) == 0 or all([len(o) == len(x_data[0]) for o in other])), \
		f'Mismatching number of samples within other data: {[len(o) for o in other]}'

	# Ensure only positive / finite testing features, but allow the
	# possibility of some nan values in x_data (if allow_nan_inp is
	# set) or in y_data (if allow_nan_out is set) - so long as the 
	# sample has other non-nan values in the respective feature set
	valid = np.ones(len(x_data[0])).astype(np.bool)
	for i, fullset in enumerate(both_data):
		for subset in fullset:
			subset[np.isnan(subset)] = -999.
			subset[np.logical_or(subset <= 0, not i and (subset >= 10))] = np.nan 
			has_nan = np.any if (i and allow_nan_out) or (not i and allow_nan_inp) else np.all 
			valid   = np.logical_and(valid, has_nan(np.isfinite(subset), 1))

	x_data = [x[valid] for x in x_data]
	y_data = [y[valid] for y in y_data]
	z_data =  [z[valid] for z in z_data]
	x_data_full =[x_full[valid] for x_full in x_data_full]
	z_w_data =  [z_w[valid] for z_w in z_w_data]
	if load_lat_lon:
		lat_data =  [lat[valid] for lat in lat_data]
		lon_data =  [lon[valid] for lon in lon_data]
	else:
		lat_data = [0]
		lon_data = [0]

	print(f'Removed {(~valid).sum()} invalid samples ({valid.sum()} remaining)')
	assert(valid.sum()), 'All samples have nan or negative values'

	if len(other) > 0:
		return x_data, y_data, [np.array(o)[valid]  for o in other], z_data, z_w_data, x_data_full,lat_data,lon_data
	return x_data, y_data, z_data, z_w_data, x_data_full,lat_data,lon_data


def get_data(args,return_PRISMA_testing=False,return_full_bands=False,load_lat_lon=False):
	''' Main function for gathering datasets '''
	np.random.seed(args.seed)
	sensor   = args.sensor.split('-')[0]
	products = args.product.split(',')
	bands    = get_sensor_bands(args.sensor, args)


	# Using Hydrolight simulated data
	if using_feature(args, 'sim'):
		assert(not using_feature(args, 'ratio')), 'Too much memory needed for simulated+ratios'
		data_folder = ['848']
		data_keys   = ['Rrs', 'bb_p', 'a_p', '../chl', '../tss', '../cdom']
		data_path   = Path(args.sim_loc)

	else:
		if products[0] == 'all':
			products = ['chl', 'tss', 'cdom', 'ad', 'ag', 'aph']# + ['a*ph', 'apg', 'a'] 

		data_folder = []
		data_keys   = ['Rrs'] 
		data_keys  += ['../Rrs_OG']	
		data_keys  += ['../Rrs_OG_wvl']
		data_keys  += ['Rrs_full_wavelengths']
		if load_lat_lon:
			data_keys  += ['../lat'] 	
			data_keys  += ['../lon']


		if return_PRISMA_testing:
			data_path   = Path("/home/ryanoshea/MDN_PC/MDN/script_formatted_data/Ryan_data/Data/Test_21_PRISMA_Curonia")
		else:
			data_path   = Path(args.data_loc)
		get_dataset = lambda path, p: Path(path.as_posix().replace(f'/{sensor}','').replace(f'/{p}.csv','')).stem

		for product in products:
			if product in ['chl', 'tss', 'cdom','PC']:
				product = f'../{product}'
		
			# MSI / OLCI paper
			if args.dataset == 'sentinel_paper' and product == '../chl': 
				datasets = ['Sundar', 'UNUSED/Taihu_old', 'UNUSED/Taihu2', 'UNUSED/Schalles_old', 'SeaBASS2', 'Vietnam'] 

			# Find all datasets with the given product available
			else:
				safe_prod = product.replace('*', '[*]') # Prevent glob from getting confused by wildcard
				datasets  = [get_dataset(path, product) for path in data_path.glob(f'*/{sensor}/{safe_prod}.csv')]

				if product == 'aph':
					datasets = [d for d in datasets if d not in ['PACE']]
				
				if product == '../chl':
					datasets = [d for d in datasets if d not in ['Arctic']] # Bunkei contained within sundar's set
				
			data_folder += datasets
			data_keys   += [product]

	# Get only unique entries, while also preserving insertion order
	order_unique = lambda a: [a[i] for i in sorted(np.unique(a, return_index=True)[1])]
	data_folder  = order_unique(data_folder)
	data_keys    = order_unique(data_keys)
	assert(len(data_folder)), f'No datasets found for {products} within {data_path}/*/{sensor}'
	assert(len(data_keys)),  f'No variables found for {products} within {data_path}/*/{sensor}'
	
	sensor_loc = [data_path.joinpath(f, sensor) for f in data_folder]
	x_data, y_data, slices, sources,z_data, z_w_data,x_data_full,lat_data,lon_data = _load_datasets(data_keys, sensor_loc, bands, allow_missing=('-nan' in args.sensor) or (getattr(args, 'align', None) is not None),args=args,load_lat_lon=load_lat_lon)

	# Hydrolight simulated CDOM is incorrectly scaled
	if using_feature(args, 'sim') and 'cdom' in slices:
		y_data[:, slices['cdom']] *= 0.18

	# Allow data from one sensor to be aligned with other sensors (so the samples will be the same across sensors) 
	if getattr(args, 'align', None) is not None:
		assert('-nan' not in args.sensor), 'Cannot allow all samples via "-nan" while also aligning to other sensors'
		align = args.align.split(',')
		if 'all' in align: 
			align = [s for s in SENSOR_LABELS.keys() if s != 'HYPER']
		align_loc = [[data_path.joinpath(f, a.split('-')[0]) for f in data_folder] for a in align]

		print(f'\nLoading alignment data for {align}...')
		x_align, y_align, slices_align, sources_align,z_data, z_w_data,x_data_full,lat_data,lon_data  = map(list,
			zip(*[_load_datasets(data_keys, loc, get_sensor_bands(a, args), allow_missing=True,load_lat_lon=load_lat_lon) for a, loc in zip(align, align_loc)]))
		
		x_data = [x_data] + x_align
		y_data = [y_data] + y_align

	# if -nan IS in the sensor label: do not filter samples; allow all, regardless of nan compositio
	if '-nan' not in args.sensor: 
		(x_data, *_), (y_data, *_), (sources, *_),(z_data,*_),(z_w_data,*_),(x_data_full,*_),(lat_data,*_),(lon_data,*_)= _filter_invalid(x_data, y_data, slices, other=[sources], allow_nan_out=args.allow_nan_out,z_data=z_data,z_w_data=z_w_data,x_data_full=x_data_full,lat_data=lat_data,lon_data=lon_data,load_lat_lon=load_lat_lon) 
			
	print('\nFinal counts:')
	print('\n'.join([f'\tN={num:>5} | {loc}' for loc, num in zip(*np.unique(sources[:, 0], return_counts=True))]))
	print(f'\tTotal: {len(sources)}')

	# Correct chl data for pheopigments
	if 'chl' in args.product and using_feature(args, 'tchlfix'):
		assert(not using_feature(args, 'sim')), 'Simulated data does not need TChl correction'
		y_data = _fix_tchl(y_data, sources, slices, data_path)

	return x_data, y_data, slices, sources, z_data, z_w_data,x_data_full,lat_data,lon_data


def _fix_tchl(y_data, sources, slices, data_path, debug=False):
	''' Very roughly correct chl for pheopigments '''
	import pandas as pd 

	dataset_name, sample_idx = sources.T 
	sample_idx.astype(int)

	fix = np.ones(len(y_data)).astype(np.bool)
	old = y_data.copy()

	set_idx = np.where(dataset_name == 'Sundar')[0]
	dataset = np.loadtxt(data_path.joinpath('Sundar', 'Dataset.csv'), delimiter=',', dtype=str)[sample_idx[set_idx]]
	fix[set_idx[dataset == 'ACIX_Krista']] = False
	fix[set_idx[dataset == 'ACIX_Moritz']] = False

	set_idx = np.where(data_lbl == 'SeaBASS2')[0]
	meta    = pd.read_csv(data_path.joinpath('SeaBASS2', 'meta.csv')).iloc[sample_idx[set_idx]]
	lonlats = meta[['east_longitude', 'west_longitude', 'north_latitude', 'south_latitude']].apply(lambda v: v.apply(lambda v2: v2.split('||')[0]))
	# assert(lonlats.apply(lambda v: v.apply(lambda v2: v2.split('::')[0] == 'rrs')).all().all()), lonlats[~lonlats.apply(lambda v: v.apply(lambda v2: v2.split('::')[0] == 'rrs')).all(1)]
	
	lonlats = lonlats.apply(lambda v: pd.to_numeric(v.apply(lambda v2: v2.split('::')[1].replace('[deg]','')), 'coerce'))
	lonlats = lonlats[['east_longitude', 'north_latitude']].to_numpy()

	# Only needs correction in certain areas, and for smaller chl magnitudes
	fix[set_idx[np.logical_and(lonlats[:,0] < -117, lonlats[:,1] > 32)]] = False
	fix[y_data[:,0] > 80] = False
	print(f'Correcting {fix.sum()} / {len(fix)} samples')

	coef = [0.04, 0.776, 0.015, -0.00046, 0.000004]
	# coef = [-0.12, 0.9, 0.001]
	y_data[fix, slices['chl']] = np.sum(np.array(coef) * y_data[fix, slices['chl']] ** np.arange(len(coef)), 1, keepdims=False)

	if debug:
		import matplotlib.pyplot as plt
		from .plot_utils import add_identity
		plt.scatter(old, y_data)
		plt.xlabel('Old')
		plt.ylabel('New')
		plt.xscale('log')
		plt.yscale('log')
		add_identity(plt.gca(), color='k', ls='--')
		plt.xlim((y_data[y_data > 0].min()/10, y_data.max()*10))
		plt.ylim((y_data[y_data > 0].min()/10, y_data.max()*10))
		plt.show()
	return y_data

def assemble_resampled_matchups(sensor,matchup_folder_name,bands):
	Rrs_insitu_resampled = pd.read_csv(matchup_folder_name + '/' + sensor + '/Rrs.csv',header=None)
	Rrs_insitu_resampled = pd.DataFrame(Rrs_insitu_resampled)
	Rrs_insitu_resampled = Rrs_insitu_resampled.values

	Rrs_insitu_resampled_wvl =  pd.DataFrame(pd.read_csv(matchup_folder_name + '/' + sensor + '/Rrs_wvl.csv',header=None))
	Rrs_insitu_resampled_wvl = Rrs_insitu_resampled_wvl.values
	Rrs_insitu_resampled_full = Rrs_insitu_resampled[:,return_valid_wavelengths(desired_wavelengths=get_sensor_bands(sensor+'-noNIR_LB'),available_wavelengths=Rrs_insitu_resampled_wvl)]
	Rrs_insitu_resampled_full_NIR = Rrs_insitu_resampled[:,return_valid_wavelengths(desired_wavelengths=get_sensor_bands(sensor+'-SimisFull'),available_wavelengths=Rrs_insitu_resampled_wvl)]
	valid_wavelength_bool = return_valid_wavelengths(desired_wavelengths=get_sensor_bands(sensor+'-SimisFull'),available_wavelengths=Rrs_insitu_resampled_wvl)

	if sum(valid_wavelength_bool) ==  0:
		Rrs_insitu_resampled_full_NIR = np.empty((len(Rrs_insitu_resampled_full_NIR),len(get_sensor_bands(sensor+'-SimisFull'))))
		Rrs_insitu_resampled_full_NIR[:] = np.NaN
		print('PRINTING SHAPE OF 0: ', np.shape(Rrs_insitu_resampled_full_NIR))
	Rrs_insitu_resampled = Rrs_insitu_resampled[:,return_valid_wavelengths(desired_wavelengths=bands,available_wavelengths=Rrs_insitu_resampled_wvl)]

	def recover_values(matchup_folder_name,desired_value):
		if desired_value == 'Rrs_retrieved':
			desired_product_df = pd.read_csv(matchup_folder_name + '/' + desired_value + '.csv')
		else:
			desired_product_df = pd.read_csv(matchup_folder_name + '/' + desired_value + '.csv',header=None)
		desired_product_df = pd.DataFrame(desired_product_df)
		desired_product_values = desired_product_df.values
		return desired_product_values

	def recover_header(matchup_folder_name,desired_value):
		desired_product_df = pd.read_csv(matchup_folder_name + '/' + desired_value + '.csv')
		desired_product_df = pd.DataFrame(desired_product_df)
		desired_product_columns = desired_product_df.columns
		return desired_product_columns

	products = ['chl','PC','dataset','insitu_datetime','path','Rrs_retrieved','site_label','plotting_labels','lat','lon']
	dict_of_product_vals=dict() 
	dict_of_product_vals['insitu_Rrs_resampled'] = Rrs_insitu_resampled
	dict_of_product_vals['insitu_Rrs_resampled_full'] = Rrs_insitu_resampled_full
	dict_of_product_vals['Rrs_insitu_resampled_full_NIR'] = Rrs_insitu_resampled_full_NIR

	if return_valid_wavelengths(desired_wavelengths=bands,available_wavelengths=Rrs_insitu_resampled_wvl).any() == False or return_valid_wavelengths(desired_wavelengths=bands,available_wavelengths=Rrs_insitu_resampled_wvl).any() == False:
		print(return_valid_wavelengths(desired_wavelengths=bands,available_wavelengths=Rrs_insitu_resampled_wvl))
		print('MISSING DESIRED WAVELENGTHS for :',matchup_folder_name)

		return False

	dict_of_product_vals['insitu_Rrs_resampled_wvl'] = np.transpose(Rrs_insitu_resampled_wvl[return_valid_wavelengths(desired_wavelengths=bands,available_wavelengths=Rrs_insitu_resampled_wvl)])
	dict_of_product_vals['insitu_Rrs_resampled_wvl_full'] = np.transpose(Rrs_insitu_resampled_wvl[return_valid_wavelengths(desired_wavelengths=get_sensor_bands(sensor+'-noNIR_LB'),available_wavelengths=Rrs_insitu_resampled_wvl)])

	Rrs_retieved_wvl =  np.reshape((np.asarray(recover_header(matchup_folder_name,'Rrs_retrieved'))).astype(float).astype(int),(-1,1))

	if return_valid_wavelengths(desired_wavelengths=bands,available_wavelengths=Rrs_retieved_wvl).any() == False:
		print(return_valid_wavelengths(desired_wavelengths=bands,available_wavelengths=Rrs_retieved_wvl))
		print('MISSING DESIRED WAVELENGTHS for Remote matchup:',matchup_folder_name)

		return False
	dict_of_product_vals['Rrs_retrieved_wvl'] = np.transpose(np.reshape(np.asarray(Rrs_retieved_wvl[return_valid_wavelengths(desired_wavelengths=bands,available_wavelengths=Rrs_retieved_wvl)]),(-1,1)))
	dict_of_product_vals['Rrs_retrieved_wvl_full'] = np.transpose(np.reshape(np.asarray(Rrs_retieved_wvl[return_valid_wavelengths(desired_wavelengths=get_sensor_bands(sensor+'-noNIR_LB'),available_wavelengths=Rrs_retieved_wvl)]),(-1,1)))

	for current_product in products:
		dict_of_product_vals[current_product] = recover_values(matchup_folder_name,current_product)
	Rrs_retrieved = dict_of_product_vals['Rrs_retrieved']

	dict_of_product_vals['Rrs_retrieved'] = Rrs_retrieved[:,return_valid_wavelengths(desired_wavelengths=bands,available_wavelengths=Rrs_retieved_wvl)]
	dict_of_product_vals['Rrs_retrieved_full'] = Rrs_retrieved[:,return_valid_wavelengths(desired_wavelengths=get_sensor_bands(sensor+'-noNIR_LB'),available_wavelengths=Rrs_retieved_wvl)]
	return dict_of_product_vals;

def return_valid_wavelengths(desired_wavelengths,available_wavelengths,margin=1):

    available_wavelengths = np.asarray(available_wavelengths)
   
    if (np.all([np.abs(available_wavelengths-w).min() <= margin for w in desired_wavelengths])) == False:
        print(f'Missing wavelengths: \n{desired_wavelengths} needed,\n{available_wavelengths.flatten()} found')
        return np.array([False] * len(available_wavelengths))

    if len(available_wavelengths) != len(desired_wavelengths):
        valid = np.abs(available_wavelengths - np.atleast_2d(desired_wavelengths)).min(1) < margin
        assert(valid.sum() == len(desired_wavelengths)), [available_wavelengths[valid].flatten(), desired_wavelengths]
        return valid 
    return np.array([True] * len(desired_wavelengths))


def get_matchups(sensor):
	dict_of_product_vals = dict()
	matchup_folder_name = '/home/ryanoshea/MDN_PC/MDN/script_formatted_data/Ryan_data/Data/Matchups_lat_lon/in-situ/'
	
	extension_folders = ['set_0','set_1','set_2','set_3']
	Rrs_insitu_resampled = []

	bands = get_sensor_bands(sensor)
	sensor = sensor.split('-')[0]

	for i,extension in enumerate(extension_folders):
		if i ==0:
			dict_of_product_vals = assemble_resampled_matchups(sensor=sensor,matchup_folder_name=matchup_folder_name+extension,bands=bands)

		else:
			dict_of_product_vals_current = assemble_resampled_matchups(sensor=sensor,matchup_folder_name=matchup_folder_name+extension,bands=bands)
			if dict_of_product_vals_current != False:
				for key in dict_of_product_vals.keys():
					dict_of_product_vals[key] = np.concatenate((dict_of_product_vals[key],(dict_of_product_vals_current[key])))
	return dict_of_product_vals

def get_in_situ_PRISMA(sensor):
	matchup_folder_name = '/home/ryanoshea/MDN_PC/MDN/script_formatted_data/Ryan_data/Data/Test_21_PRISMA/Curonia'

	Rrs_insitu_resampled = pd.read_csv(matchup_folder_name + '/' + sensor + '/Rrs.csv',header=None)
	Rrs_insitu_resampled = pd.DataFrame(Rrs_insitu_resampled)
	Rrs_insitu_resampled = Rrs_insitu_resampled.values

	Rrs_insitu_resampled_wvl =  pd.DataFrame(pd.read_csv(matchup_folder_name + '/' + sensor + '/Rrs_wvl.csv',header=None))
	Rrs_insitu_resampled_wvl = Rrs_insitu_resampled_wvl.values
	Rrs_insitu_resampled_full = Rrs_insitu_resampled[:,return_valid_wavelengths(desired_wavelengths=get_sensor_bands(sensor),available_wavelengths=Rrs_insitu_resampled_wvl)]
	Rrs_insitu_resampled = Rrs_insitu_resampled[:,return_valid_wavelengths(desired_wavelengths=bands,available_wavelengths=Rrs_insitu_resampled_wvl)]

	chl = pd.read_csv(matchup_folder_name + '/' + 'chl' + '.csv',header=None)
	PC = pd.read_csv(matchup_folder_name + '/' + 'PC' + '.csv',header=None)

	return 0

# Define a function that you pass z_data (raw Rrs) and z_w_data (raw Rrs wavelengths) and then applies SAM to sort them. SAM function should apply to individual spectra
def assign_spectra_OWT(z_data,z_w_data):
	OWT_assignments=[]
	import pickle
	#load the end members
	with open('OWT_end_members.pkl','rb') as f:
		Rrs_wvl_list = pickle.load(f)
	loaded_Rrs_list = Rrs_wvl_list[0]
	loaded_Rrs_wvl_list = Rrs_wvl_list[1]
	
	for Rrs_counter,Rrs in enumerate(z_data):
		wavelengths = z_w_data[Rrs_counter]

		OWT_assignments.append(	assign_spectrum_OWT(Rrs,wavelengths,loaded_Rrs_list,loaded_Rrs_wvl_list))
	return OWT_assignments

def assign_spectrum_OWT(z_data,z_w_data,end_members,end_member_wavelengths):
	from spectral import spectral_angles
	from scipy.interpolate import interp1d

	min_end_member_wavelength = np.min(end_member_wavelengths[0])
	max_end_member_wavelength = np.max(end_member_wavelengths[0])
	
	z_w_data_bool_array = [True if wavelength<max_end_member_wavelength and wavelength>min_end_member_wavelength else False for wavelength in z_w_data ]

	z_data=z_data[z_w_data_bool_array]
	z_w_data=z_w_data[z_w_data_bool_array]

	interpolated_member_to_z_w_data_list = []

	for end_member_counter,end_member in enumerate(end_members):
		interp_function = interp1d(end_member_wavelengths[end_member_counter],end_member,kind='cubic')
		interpolated_member_to_z_w_data_list.append(interp_function(z_w_data))

	array_of_end_members = np.asarray(interpolated_member_to_z_w_data_list)

	z_data_reshaped = z_data.reshape(1,1,-1)

	spectral_angles_available = spectral_angles(z_data_reshaped,array_of_end_members)

	minimum_spectral_angle = np.argmin(spectral_angles_available)
	
	minimum_spectral_angle = minimum_spectral_angle +1

	return minimum_spectral_angle

def assign_spectrum_OWT_angle_Rrs(z_data,z_w_data,end_members,end_member_wavelengths):
	from spectral import spectral_angles
	from scipy.interpolate import interp1d

	min_end_member_wavelength = np.min(end_member_wavelengths[0])
	max_end_member_wavelength = np.max(end_member_wavelengths[0])

	z_w_data_bool_array = [True if wavelength<max_end_member_wavelength and wavelength>min_end_member_wavelength else False for wavelength in z_w_data ]

	z_data=z_data[z_w_data_bool_array]
	z_w_data=z_w_data[z_w_data_bool_array]

	interpolated_member_to_z_w_data_list = []

	for end_member_counter,end_member in enumerate(end_members):
		interp_function = interp1d(end_member_wavelengths[end_member_counter],end_member,kind='cubic')
		interpolated_member_to_z_w_data_list.append(interp_function(z_w_data))

	array_of_end_members = np.asarray(interpolated_member_to_z_w_data_list)

	z_data_reshaped = z_data.reshape(1,1,-1)

	spectral_angles_available = spectral_angles(z_data_reshaped,array_of_end_members)

	minimum_spectral_angle = np.argmin(spectral_angles_available)

	minimum_spectral_angle_value = spectral_angles_available[0][0][minimum_spectral_angle]

	nearest_member_Rrs = interpolated_member_to_z_w_data_list[minimum_spectral_angle]
	
	minimum_spectral_angle = minimum_spectral_angle +1 

	return minimum_spectral_angle, minimum_spectral_angle_value, nearest_member_Rrs,z_w_data

def assign_spectra_OWT_angle_nearest_Rrs(z_data,z_w_data):
	OWT_assignments=[]
	nearest_member_Rrs = []
	min_OWT_angle = []
	member_w_list = []
	import pickle

	with open('OWT_end_members.pkl','rb') as f:
		Rrs_wvl_list = pickle.load(f)
	loaded_Rrs_list = Rrs_wvl_list[0]
	loaded_Rrs_wvl_list = Rrs_wvl_list[1]

	for Rrs_counter,Rrs in enumerate(z_data):
		wavelengths = z_w_data[Rrs_counter]
		OWT_assignment, min_OWT,nearest_member,member_w = assign_spectrum_OWT_angle_Rrs(Rrs,wavelengths,loaded_Rrs_list,loaded_Rrs_wvl_list)

		OWT_assignments.append(OWT_assignment)
		min_OWT_angle.append(min_OWT)
		nearest_member_Rrs.append(nearest_member)
		member_w_list.append(member_w)
	return OWT_assignments, min_OWT_angle, nearest_member_Rrs,member_w_list

def load_geotiff_bands(sensor,path_to_tile="projected_calimnos.tif",wvl_key="",allow_neg=False,atmospheric_correction="",OVERRIDE_DIVISOR=None):
    #from osgeo import gdal
    #import rasterio
    #from rasterio.plot import show
	#if '.tif' in path_to_tile:
    if wvl_key not in ["_Rw","Rrs_","Oax"]:
        if '_calimnos' in path_to_tile or atmospheric_correction=="POLYMER":
            wvl_key = "_Rw"
            div   = np.pi 
        if '_acolite' in path_to_tile or atmospheric_correction=="ACOLITE":
            wvl_key = "Rrs_"
            div   = 1
        if '_processed_reprojected' in path_to_tile or atmospheric_correction=="iCOR":
            wvl_key = "Oax"
            div   = np.pi 
        if atmospheric_correction =='asi':
            div   = np.pi
            print(f'Using {atmospheric_correction} atmospherically corrected image and dividing by {div} (unless overwritten by OVERRIDE_DIVISOR)')
            print('Assuming the bands match the desired sensor bands, are correctly ordered from lowest to highest, and that the AC output is in Rrs')
            input('Press enter to confirm that this is true')
            
    if OVERRIDE_DIVISOR:
        div = OVERRIDE_DIVISOR
        #print("Will be dividing by", div)
    print("Wvl_key is set to:",wvl_key)
    if div != 1: 
        print("Converting to Rrs by dividing by:", div)
    else:
        print("Image already in Rrs, dividing by",div)
    import tifffile
    from tifffile import TiffFile
    #array = TiffFile.imread(path_to_tile)
    #array = img.read()

    array = []
    for page in TiffFile(path_to_tile).pages:
        image = page.asarray()
        if atmospheric_correction == 'asi':
                return get_sensor_bands(sensor), np.ma.array(image).filled(fill_value=np.nan)/div
        array.append(image)
        for tag in page.tags.values():
            if tag.name =="65000":
                tag_value = tag.value
                #print("tag.name", "tag.code", "tag.dtype", "tag.count", "tag.value")
                #print(tag.name, tag.code, tag.dtype, tag.count, tag.value)
    #print("tag_value")

#    print(tag_value)
    #print("----Array shape",np.shape(array))
    array = np.squeeze(np.asarray(array))
    #print("----Array shape",np.shape(array))

    #Write xml document        
    text_file = open("available_geotiff_info.xml", "w")
    n = text_file.write(tag_value)
    text_file.close()
    
    from xml.etree import ElementTree as ET
    
    xml = ET.parse("available_geotiff_info.xml")
    root_element = xml.getroot()
    
    #products
    available_band_names=[]
    wavelengths=[]
    band_imagery=[]
   
    for i,child_el in enumerate(root_element):
        #print(i,child_el.items(),child_el.keys(),child_el.tag)
        if child_el.tag == 'Image_Interpretation':
            Image_Interpretation_integer = i
    #print(root_element[dataset_sources_integer])
    dict_of_Oa_bands = {'Oa1':400.0,
                    'Oa2':412.5,
                    'Oa3':442.5,
                    'Oa4':490.0,
                    'Oa5':510.0,
                    'Oa6':560.0,
                    'Oa7':620.0,
                    'Oa8':665.0,
                    'Oa9':673.75,
                    'Oa10':681.25,
                    'Oa11':708.75,
                    'Oa12':753.75,
                    'Oa13':761.25,
                    'Oa14':764.375,
                    'Oa15':767.5,
                    'Oa16':778.75,
                    'Oa17':865.0,
                    'Oa18':885.0,
                    'Oa19':900.0,
                    'Oa20':940.0,
                    'Oa21':1020.0}
    for i,child_el in enumerate(root_element[Image_Interpretation_integer]):
        #print(i,child_el.items(),child_el.keys(),child_el.tag)  
        for i_2, child_el_2 in enumerate(root_element[Image_Interpretation_integer][i]):
            #print(i_2,child_el_2.tag,root_element[Image_Interpretation_integer][i][i_2].text)
            child_tag = child_el.tag
            if child_el_2.tag == "BAND_INDEX":
                band_index = int(root_element[Image_Interpretation_integer][i][i_2].text)
            if child_el_2.tag == 'BAND_NAME':
                current_band_name = root_element[Image_Interpretation_integer][i][i_2].text
                available_band_names.append(current_band_name)
                #print("CURRENT BAND NAME",current_band_name)
                if current_band_name in dict_of_Oa_bands.keys() and wvl_key=="Oax":
                     wavelengths.append(dict_of_Oa_bands[current_band_name])
                     band_imagery.append(array[band_index,:,:])
                else:
                    if wvl_key in current_band_name:
                        wavelengths.append(current_band_name.split(wvl_key)[1])
                        band_imagery.append(array[band_index,:,:])
                
    wavelengths_descending, bands_descending = (list(t) for t in zip(*sorted(zip([np.int64(wavelength) for wavelength in wavelengths], band_imagery),reverse=False)))
    sensor_bands_required = get_sensor_bands(sensor)
    indices_of_required_wavelengths = []
    
    #find closest index
    for sensor_band_required in sensor_bands_required:
        index_of_required_wavelength = find_wavelength(sensor_band_required,wavelengths_descending)
        indices_of_required_wavelengths.append(index_of_required_wavelength)
    #div   = np.pi if 'Rw' in wvl_key or 'Oa' in wvl_key else 1
    bands_descending_required = [band_descending/div for i,band_descending in enumerate(bands_descending) if i in indices_of_required_wavelengths]   
    bands_descending_required  = np.ma.stack(bands_descending_required,axis=-1)
    #print("Divided input image by: ", div)

    if not allow_neg: bands_descending_required[bands_descending_required <= 0] = np.nan

    return sensor_bands_required,bands_descending_required.filled(fill_value=np.nan)
