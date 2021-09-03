from sklearn.base import TransformerMixin
from sklearn import preprocessing
import pickle as pkl
import numpy as np 
import warnings
#from .utils import closest_wavelength, ignore_warnings

# must redefine functions here as .utils depends on transformers
def find_wavelength(k, waves, validate=True, tol=5):
	''' Index of closest wavelength '''
	i = np.abs(np.array(waves) - k).argmin() 
	assert(not validate or (abs(k-waves[i]) <= tol)), f'Needed {k}nm, but closest was {waves[i]}nm in {waves}'
	return i 

def validate_wavelength(k, waves, validate=True, tol=5):
	''' Index of closest wavelength '''
	i = np.abs(np.array(waves) - k).argmin() 
	less_than_tol_bool = abs(k-waves[i]) <= tol
	#if less_than_tol_bool:
		#print('Available wavelength {} is within {} nm of desired wavelength {}'.format(waves[i],tol,k))
	#assert(not validate or (abs(k-waves[i]) <= tol)), f'Needed {k}nm, but closest was {waves[i]}nm in {waves}'
	return less_than_tol_bool

def closest_wavelength(k, waves, validate=True, tol=5): 
	''' Value of closest wavelength '''
	return waves[find_wavelength(k, waves, validate, tol)]	
	
class CustomTransformer(TransformerMixin):
	''' Data transformer class which validates data shapes. 
		Child classes should override _fit, _transform, _inverse_transform '''
	_input_shape  = None 
	_output_shape = None

	def fit(self, X, *args, **kwargs):				 
		self._input_shape = X.shape[1]
		return self._fit(X.copy(), *args, **kwargs)

	def transform(self, X, *args, **kwargs):
		# print('XSCALER SHAPES',self._input_shape,X.shape[1],X.shape[0])
		if self._input_shape is not None:
			assert(X.shape[1] == self._input_shape), f'Number of data features changed: {self._input_shape} vs {X.shape[1]}'
		X = self._transform(X.copy(), *args, **kwargs)
		
		if self._output_shape is not None:
			assert(X.shape[1] == self._output_shape), f'Number of data features changed: {self._output_shape} vs {X.shape[1]}'
		self._output_shape = X.shape[1]
		return X 

	def inverse_transform(self, X, *args, **kwargs):
		if self._output_shape is not None:
			assert(X.shape[1] == self._output_shape), f'Number of data features changed: {self._output_shape} vs {X.shape[1]}'
		X = self._inverse_transform(X.copy(), *args, **kwargs)
		
		if self._input_shape is not None:
			assert(X.shape[1] == self._input_shape), f'Number of data features changed: {self._input_shape} vs {X.shape[1]}'
		self._input_shape = X.shape[1]
		return X 

	def return_labels(self):
		return self.return_labels()

	def _fit(self, X, *args, **kwargs):				  return self
	def _transform(self, X, *args, **kwargs):		  raise NotImplemented
	def _inverse_transform(self, X, *args, **kwargs): raise NotImplemented



class IdentityTransformer(CustomTransformer):
	def _transform(self, X, *args, **kwargs):		 return X
	def _inverse_transform(self, X, *args, **kwargs): return X


class LogTransformer(CustomTransformer):
	def _transform(self, X, *args, **kwargs):		 return np.log(X)
	def _inverse_transform(self, X, *args, **kwargs): return np.exp(X)


class NegLogTransformer(CustomTransformer):
	''' 
	Log-like transformation which allows negative values (Whittaker et al. 2005)
	http://fmwww.bc.edu/repec/bocode/t/transint.html
	'''
	def _transform(self, X, *args, **kwargs):		 return np.sign(X) *  np.log(np.abs(X)  + 1)
	def _inverse_transform(self, X, *args, **kwargs): return np.sign(X) * (np.exp(np.abs(X)) - 1)


class ColumnTransformer(CustomTransformer):
	''' Reduce columns to specified selections (feature selection) '''
	def __init__(self, columns, *args, **kwargs):	 self._c = columns 
	def _transform(self, X, *args, **kwargs):		 return X[:, self._c]


class BaggingColumnTransformer(CustomTransformer):
	''' Randomly select a percentage of columns to drop '''
	percent = 0.75

	def __init__(self, n_bands, *args, n_extra=0, **kwargs):
		self.n_bands = n_bands
		self.n_extra = n_extra

	def _fit(self, X, *args, **kwargs):
		# if X.shape[1] > 60: 
		# 	self.percent = 0.05
		# 	n_bands_tmp  = self.n_bands
		# 	self.n_bands = 27

		shp  = X.shape[1] - self.n_bands
		ncol = int(shp*self.percent)
		cols = np.arange(shp-self.n_extra) + self.n_bands
		np.random.shuffle(cols)

		# if X.shape[1] > 60:
		# 	shp2  = self.n_bands - n_bands_tmp
		# 	ncol2 = int(shp2*0.75)
		# 	cols2 = np.arange(shp2) + n_bands_tmp
		# 	np.random.shuffle(cols2)
		# 	self.cols = np.append(np.arange(n_bands_tmp), cols2)
		# 	self.cols = np.append(self.cols, cols[:ncol])
		# 	ncol += ncol2
		# else:

		if self.n_extra:
			self.cols = np.append(np.arange(self.n_bands), list(cols[:ncol]) + list(X.shape[1]-(np.arange(self.n_extra)+1)), 0)
		else:
			self.cols = np.append(np.arange(self.n_bands), list(cols[:ncol]), 0)
		# print(f'Reducing bands from {shp} ({X.shape[1]} total) to {ncol} ({len(self.cols)} total) ({self.cols})')
		return self

	def _transform(self, X, *args, **kwargs):
		return X[:, self.cols.astype(int)]


class ExclusionTransformer(CustomTransformer):
	''' 
	Exclude certain columns from being transformed by the given transformer.
	The passed in transformer should be a transformer class, and exclude_slice can
	be any object which, when used to slice a numpy array, will give the 
	appropriate columns which should be excluded. So, for example:
		- slice(1)
		- slice(-3, None)
		- slice(1,None,2)
		- np.array([True, False, False, True])
		etc.
	'''
	def __init__(self, exclude_slice, transformer, transformer_args=[], transformer_kwargs={}):
		self.excl = exclude_slice
		self.transformer = transformer(*transformer_args, **transformer_kwargs)

	def _fit(self, X):
		cols = np.arange(X.shape[1])
		cols = [c for c in cols if c not in cols[self.excl]]
		self.transformer.fit(X[:, cols])
		self.keep = cols
		return self

	def _transform(self, X, *args, **kwargs):
		Z = np.zeros_like(X)
		Z[:, self.keep] = self.transformer.transform(X[:, self.keep])
		Z[:, self.excl] = X[:, self.excl]
		return Z 

	def _inverse_transform(self, X, *args, **kwargs):
		Z = np.zeros_like(X)
		Z[:, self.keep] = self.transformer.inverse_transform(X[:, self.keep])
		Z[:, self.excl] = X[:, self.excl]
		return Z 


class RatioTransformer(CustomTransformer):	
	''' Add ratio features '''
	def __init__(self, wavelengths, only_ratio_bool=False,BRs=None,LHs=None,only_append_LH=None,*args, label='', **kwargs):
		self.wavelengths = list(wavelengths)
		self.label = label 
		self.only_ratio_bool = only_ratio_bool
		self.BRs = BRs if BRs != None else None
		self.LHs = LHs if LHs != None else None
		self.only_append_LH = only_append_LH if only_append_LH  != None else None
	def _fit(self, X):
		self.shape = X.shape[1]
		return self 

	def _transform(self, X, *args, **kwargs):		 
		''' 
		Simple feature engineering method. Add band 
		ratios as features. Does not add reciprocal 
		ratios or any other duplicate features; 
		adds a band sum ratio (based on	three-band 
		Chl retrieval method).

		Usage:
			# one sample with three features, shaped [samples, features]
			x = [[a, b, c]] 
			y = ratio(x)
				# y -> [[a, b, c, b/a, b/(a+c), c/a, c/(a+b), c/b, a/(b+c)]
		'''

		def LH(L1, L2, L3, R1, R2, R3):
			c  = (L3 - L2) / (L3 - L1)
			return R2 - c*R1 - (1-c)*R3

		x     = np.atleast_2d(X)
		
		#If we only want to use the ratios, remove the Rrs
		if self.only_ratio_bool:
			x_new = []
		else:
			x_new = [v for v in x.T]

		label = []


		def wavelength_check(wavelength_list,wavelength,greater_bool):
			if greater_bool:
				return (any(x>wavelength for x in wavelength_list))
			else:
				return (any(x<wavelength for x in wavelength_list))


		#default wrapper that checks that the wavelengths exist, adds specified label and formula if they do 
		def appendFormula(self,desired_wavelengths,X,x_new,formula,label): #args will be a tuple of the wavelengths, followed by the Rrs
			found_wavelengths = []
			Rrs = []
			for wavelength_count, desired_wavelength in enumerate(desired_wavelengths):
				if validate_wavelength(desired_wavelength,self.wavelengths):
					found_wavelengths.append(closest_wavelength(desired_wavelength,self.wavelengths))
					Rrs.append(x[:, self.wavelengths.index(found_wavelengths[wavelength_count])])
				else:
					#print('Wavelength {} Not Found'.format(desired_wavelength))
					return False # returns false if a wavelength is not found

			if self.wavelengths != [500, 507, 515, 523, 530,
				  538, 546, 554, 563, 571, 579, 588, 596, 605, 614, 623, 632, 641, 651, 660, 670, 679, 689, 
				  699, 709, 719, ]: 
				if len(set(list(desired_wavelengths))) == len(list(desired_wavelengths)):
					if len(set(found_wavelengths)) != len(found_wavelengths):
						print('FOUND WAVELENGTHS ARE IDENTICAL',label)
						return False

			formula_result = formula(found_wavelengths,Rrs)

			formula_result[np.isposinf(formula_result) == True] = 1e8
			formula_result[np.isneginf(formula_result) == True] = -1e8

			x_new.append(formula_result)
			self.labels.append(label)
		self.labels = []
		

		wavelength_range = self.wavelengths

		if self.only_append_LH:
			print('NOT applying band ratios')
		else:
			if self.BRs == None:
				for numerator in wavelength_range:
					for denominator in wavelength_range:
						label_txt = f'{numerator}|{denominator}'
						appendFormula(self,[denominator,numerator],x,x_new, formula=lambda wavelengths,Rrs: Rrs[1]/Rrs[0],label=label_txt)
			else:
				for num_denominator in self.BRs:
					#if numerator > denominator:
					label_txt = f'{num_denominator[0]}|{num_denominator[1]}'
					appendFormula(self,[num_denominator[1],num_denominator[0]],x,x_new, formula=lambda wavelengths,Rrs: Rrs[1]/Rrs[0],label=label_txt)
					
		if self.LHs == None:
			# appends on a wide range of line height algorithms following the standard setup
			encircling_wavelengths = [5, 10,15,20]

			highest_wavelength = max(wavelength_range) 
			lowest_wavelength = min(wavelength_range)
			for center_wavelength in wavelength_range:
				for wavelengths_above_wavelength_below in encircling_wavelengths:
					lower_wavelength = center_wavelength - wavelengths_above_wavelength_below
					upper_wavelength = center_wavelength + wavelengths_above_wavelength_below
					if (lower_wavelength >= lowest_wavelength) and upper_wavelength<=highest_wavelength:
						label_txt = f'{lower_wavelength}|{center_wavelength}|{upper_wavelength}'

						appendFormula(self,[lower_wavelength,center_wavelength,upper_wavelength],x,x_new, formula=lambda wavelengths,Rrs: Rrs[1]-(Rrs[2]+((Rrs[0]-Rrs[2])*(wavelengths[2]-wavelengths[1])/(wavelengths[2]-wavelengths[0]))),label=label_txt)
		else:
			for center_bandwidth in self.LHs:
				lower_wavelength = center_bandwidth[0] - center_bandwidth[1]
				upper_wavelength = center_bandwidth[0] + center_bandwidth[1]
				center_wavelength = center_bandwidth[0]
				label_txt = f'{lower_wavelength}|{center_wavelength}|{upper_wavelength}'

				appendFormula(self,[lower_wavelength,center_wavelength,upper_wavelength],x,x_new, formula=lambda wavelengths,Rrs: Rrs[1]-(Rrs[2]+((Rrs[0]-Rrs[2])*(wavelengths[2]-wavelengths[1])/(wavelengths[2]-wavelengths[0]))),label=label_txt)

		# SLH algorithm from Kudela et Al. RSE
		appendFormula(self,[654,714,754],x,x_new, formula=lambda wavelengths,Rrs: Rrs[1]-(Rrs[0] +(Rrs[2]-Rrs[0])*((wavelengths[1]-wavelengths[0])/(wavelengths[2]-wavelengths[0]))),label='SLH')

		# MCI L1 665, L2 709, L3 754 ,#Multi-Algorithm Indices and Look-Up Table for Chlorophyll-a Retrieval in Highly Turbid WaterBodies Using Multispectral Data
		appendFormula(self,[665,709,754],x,x_new, formula=lambda wavelengths,Rrs: Rrs[1]-Rrs[0]*(((wavelengths[1]-wavelengths[0])/(wavelengths[2]-wavelengths[0]))*Rrs[2]-Rrs[0]),label='MCI_665')
		appendFormula(self,[680,709,754],x,x_new, formula=lambda wavelengths,Rrs: Rrs[1]-Rrs[0]*(((wavelengths[1]-wavelengths[0])/(wavelengths[2]-wavelengths[0]))*Rrs[2]-Rrs[0]),label='MCI_680')


		# CI, from OCx (https://oceancolor.gsfc.nasa.gov/atbd/chlor_a/)
		appendFormula(self,[443,555,670],x,x_new, formula=lambda wavelengths,Rrs: Rrs[1]-(Rrs[0]+(wavelengths[1]-wavelengths[0])/(wavelengths[2]-wavelengths[0])*(Rrs[2]-Rrs[0])),label='Color Index (chl)')

		# NDCI, Multi-Algorithm Indices and Look-Up Table forChlorophyll-a Retrieval in Highly Turbid WaterBodies Using Multispectral Data
		appendFormula(self,[665,709],x,x_new, formula=lambda wavelengths,Rrs: (Rrs[1] - Rrs[0])/(Rrs[1] + Rrs[0]),label='NDCI (chl)')

		#Mishra PC
		appendFormula(self,[600,700],x,x_new, formula=lambda wavelengths,Rrs: Rrs[1]/Rrs[0],label='mishra (600,700)')

		# directly calculate Green and NIR max wavelength, add to input
		# Schalles et al. found that these locations varied with PC and chl concentration
		green_wavelengths = np.asarray(self.wavelengths)
		NIR_wavelengths = np.asarray(self.wavelengths)

		green_wavelengths = green_wavelengths[np.logical_and(green_wavelengths>550 , green_wavelengths<600)]
		NIR_wavelengths = NIR_wavelengths[np.logical_and(NIR_wavelengths>694 , NIR_wavelengths<716)]
		green_wavelengths = np.ndarray.tolist(green_wavelengths)
		NIR_wavelengths = np.ndarray.tolist(NIR_wavelengths)

		appendFormula(self,green_wavelengths,x,x_new, formula=lambda wavelengths,Rrs: np.argmax(np.asarray(Rrs),axis=0),label='Max green location')
		appendFormula(self,NIR_wavelengths,x,x_new, formula=lambda wavelengths,Rrs: np.argmax(np.asarray(Rrs),axis=0),label='Max NIR location')

		#Hunter PC
		appendFormula(self,[600,615,725],x,x_new, formula=lambda wavelengths,Rrs: Rrs[2]*(1/Rrs[1]-1/Rrs[0]),label='hunter (600,615,725)')

		#Schalles BR
		appendFormula(self,[625,650],x,x_new, formula=lambda wavelengths,Rrs: Rrs[1]/Rrs[0],label='Schalles 650/625')
		
		#Decker 1993, Using Rrs instead of R(0-)
		appendFormula(self,[600,624,648],x,x_new, formula=lambda wavelengths,Rrs: 0.5*(Rrs[0]+Rrs[2])-Rrs[1],label='Decker 0.5*(R(600)+R(648))-R(624)')

		# Mishra 2014
		appendFormula(self,[629,659,724],x,x_new, formula=lambda wavelengths,Rrs: (1/Rrs[0]-1/Rrs[1])*Rrs[2],label='Mishra 2014 (1/629-1/659) * 724')

		# Simis BR
		appendFormula(self,[620,709],x,x_new, formula=lambda wavelengths,Rrs: Rrs[1]/Rrs[0],label='Simis 709/620')

		appendFormula(self,[665,709],x,x_new, formula=lambda wavelengths,Rrs: Rrs[1]/Rrs[0],label='Simis 709/665')

		appendFormula(self,[560,620,665],x,x_new, formula=lambda wavelengths,Rrs: Rrs[1]-(Rrs[2]+((Rrs[0]-Rrs[2])*(wavelengths[2]-wavelengths[1])/(wavelengths[2]-wavelengths[0]))),label='Nima LH 560,620,665')
		appendFormula(self,[665,673,681],x,x_new, formula=lambda wavelengths,Rrs: Rrs[1]-(Rrs[2]+((Rrs[0]-Rrs[2])*(wavelengths[2]-wavelengths[1])/(wavelengths[2]-wavelengths[0]))),label='Nima LH 665,673,681')
		appendFormula(self,[690,709,720],x,x_new, formula=lambda wavelengths,Rrs: Rrs[1]-(Rrs[2]+((Rrs[0]-Rrs[2])*(wavelengths[2]-wavelengths[1])/(wavelengths[2]-wavelengths[0]))),label='Nima LH 690,709,720')
		appendFormula(self,[620,650,670],x,x_new, formula=lambda wavelengths,Rrs: Rrs[1]-(Rrs[2]+((Rrs[0]-Rrs[2])*(wavelengths[2]-wavelengths[1])/(wavelengths[2]-wavelengths[0]))),label='Nima LH 620,650,670')
		appendFormula(self,[640,650,660],x,x_new, formula=lambda wavelengths,Rrs: Rrs[1]-(Rrs[2]+((Rrs[0]-Rrs[2])*(wavelengths[2]-wavelengths[1])/(wavelengths[2]-wavelengths[0]))),label='Nima LH 640,650,660')
		appendFormula(self,[613,620,627],x,x_new, formula=lambda wavelengths,Rrs: Rrs[1]-(Rrs[2]+((Rrs[0]-Rrs[2])*(wavelengths[2]-wavelengths[1])/(wavelengths[2]-wavelengths[0]))),label='Nima LH 613,620,627')


		#Wynne 2010 Characterizing a cyanobacterial bloom in western Lake Erie using satellite imagery and meteorological data

		# 665,681,709
		appendFormula(self,[665,681,709],x,x_new, formula=lambda wavelengths,Rrs: -1 * (np.pi*Rrs[1]-np.pi*Rrs[0]-(np.pi*Rrs[2]-np.pi*Rrs[0])*(wavelengths[1]-wavelengths[0])/(wavelengths[2]-wavelengths[0])),label='Cyanobacteria Index 665,681,709')

		self.n_features = len(self.labels)

		return np.hstack([v[:,None] for v in x_new])


	def transform2(self, X):
		x     = np.atleast_2d(X)
		x_new = [v for v in x.T]
		label = []
		# Band ratios
		for i, L1 in enumerate(self.wavelengths):
			for j, L2 in enumerate(self.wavelengths):
				if L1 < L2:
					R1 = x[:, i]
					R2 = x[:, j] 
					x_new.append(R2 / R1)
					label.append(f'{self.label}{L2}/{L1}')

					for k, L3 in enumerate(self.wavelengths):
						R3 = x[:, k]

						if L3 not in [L1, L2]:

							if L1 < L3:
								x_new.append(R2 * (1/R1 - 1/R3))
								label.append(f'{self.label}{L2}*(1/{L1}-1/{L3})')

							else:
								x_new.append(R3 * (1/R1 - 1/R2))
								label.append(f'{self.label}{L3}*(1/{L1}-1/{L2})')
		

		# Line height variations, examining height of center between two shoulder bands
		for i, L1 in enumerate(self.wavelengths):
			for j, L2 in enumerate(self.wavelengths):
				for k, L3 in enumerate(self.wavelengths):
					if (L3 > L2) and (L2 > L1):

						c  = (L3 - L2) / (L3 - L1)
						R1 = x[:, i]
						R2 = x[:, j]
						R3 = x[:, k]
						x_new.append(R2 - c*R1 - (1-c)*R3)
						label.append(f'{self.label}({L2}-a{L1}-b{L3})')

		self.labels = label
		return np.hstack([v[:,None] for v in x_new])

	def _inverse_transform(self, X, *args, **kwargs): 
		return np.array(X)[:, :self.shape]

	def return_labels(self):
		available_labels = self.labels
		return available_labels

class TanhTransformer(CustomTransformer):
	''' tanh-estimator (Hampel et al. 1986; Latha & Thangasamy, 2011) '''
	scale = 0.01

	def _fit(self, X, *args, **kwargs):
		m = np.median(X, 0)
		d = np.abs(X - m)

		a = np.percentile(d, 70, 0)
		b = np.percentile(d, 85, 0)
		c = np.percentile(d, 95, 0)

		Xab = np.abs(X)
		Xsi = np.sign(X)
		phi = np.zeros(X.shape)
		idx = np.logical_and(0 <= Xab, Xab < a)
		phi[idx] = X[idx]
		idx = np.logical_and(a <= Xab, Xab < b)
		phi[idx] = (a * Xsi)[idx]
		idx = np.logical_and(b <= Xab, Xab < c)
		phi[idx] = (a * Xsi * ((c - Xab) / (c - b)))[idx]

		self.mu_gh  = np.mean(phi, 0)
		self.sig_gh = np.std(phi, 0) 
		return self

	def _transform(self, X, *args, **kwargs):
		return 0.5 * (np.tanh(self.scale * ((X - self.mu_gh)/self.sig_gh)) + 1)

	def _inverse_transform(self, X, *args, **kwargs):
		return ((np.tan(X * 2 - 1) / self.scale) * self.sig_gh) + self.mu_gh
	


class TransformerPipeline(CustomTransformer):
	''' Apply multiple transformers seamlessly '''
	
	def __init__(self, scalers=None):
		if scalers is None or len(scalers) == 0: 	
			self.scalers = [
				LogTransformer(),
				preprocessing.RobustScaler(),
				preprocessing.MinMaxScaler((-1, 1)),
			]
		else:
			self.scalers = scalers 

	def _fit(self, X, *args, **kwargs):
		for scaler in self.scalers:
			X = scaler.fit_transform(X, *args, **kwargs)
		return self 

	def _transform(self, X, *args, **kwargs):
		for scaler in self.scalers:
			X = scaler.transform(X, *args, **kwargs)
		return X

	def _inverse_transform(self, X, *args, **kwargs):
		for scaler in self.scalers[::-1]:
			X = scaler.inverse_transform(X, *args, **kwargs)
		return X

	def fit_transform(self, X, *args, **kwargs):
		# Manually apply a fit_transform to avoid transforming twice
		for scaler in self.scalers:
			X = scaler.fit_transform(X, *args, **kwargs)
		return X

class TransformerPipeline_ratio(CustomTransformer):
	''' Apply multiple transformers seamlessly '''
	
	def __init__(self, scalers=None):
		if scalers is None or len(scalers) == 0: 	
			self.scalers = [
				LogTransformer(),
				preprocessing.RobustScaler(),
				preprocessing.MinMaxScaler((-1, 1)),
			]
		else:
			self.scalers = scalers 

	def _fit(self, X, *args, **kwargs):
		for scaler in self.scalers:
			X = scaler.fit_transform(X, *args, **kwargs)
		return self 

	def _transform(self, X, *args, **kwargs):
		for scaler in self.scalers:
			X = scaler.transform(X, *args, **kwargs)
		return X

	def _inverse_transform(self, X, *args, **kwargs):
		for scaler in self.scalers[::-1]:
			X = scaler.inverse_transform(X, *args, **kwargs)
		return X

	def fit_transform(self, X, *args, **kwargs):
		# Manually apply a fit_transform to avoid transforming twice
		for scaler in self.scalers:
			X = scaler.fit_transform(X, *args, **kwargs)
		return X

	def return_labels(self):
		for scaler in self.scalers:
			labels = scaler.return_labels()
		return labels


class CustomUnpickler(pkl.Unpickler):
	''' Ensure the classes are found, without requiring an import '''
	_warned = False

	def find_class(self, module, name):
		if name in globals():
			return globals()[name]
		return super().find_class(module, name)

	def load(self, *args, **kwargs):
		with warnings.catch_warnings(record=True) as w:
			pickled_object = super().load(*args, **kwargs)

		# For whatever reason, warnings does not respect the 'once' action for
		# sklearn's "UserWarning: trying to unpickle [...] from version [...] when
		# using version [...]". So instead, we catch it ourselves, and set the 
		# 'once' tracker via the unpickler itself.
		if len(w) and not CustomUnpickler._warned: 
			warnings.warn(w[0].message, w[0].category)
			CustomUnpickler._warned = True 
		return pickled_object
