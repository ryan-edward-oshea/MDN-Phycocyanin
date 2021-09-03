'''
From: A Novel algorithm for Predicting Phycocyanin Concentrations in Cyanobacteria: A Proximal Hyperspectral Remote Sensing Approach
'''

from ...utils import get_required, optimize

# Define any optimizable parameters
@optimize(['a', 'b'], [0, 0])
def model(Rrs, wavelengths, *args, **kwargs):
	required = [600, 700]
	tol = kwargs.get('tol', 5) # allowable difference from the required wavelengths
	Rrs = get_required(Rrs, wavelengths, required, tol) # get values as a function: Rrs(443)

	# Set default values for these parameters
	a = kwargs.get('a',1.0085 )
	b = kwargs.get('b',pow(2.2589,-6))
		
	mishra = a + b*Rrs(700)/Rrs(600)


	return mishra
