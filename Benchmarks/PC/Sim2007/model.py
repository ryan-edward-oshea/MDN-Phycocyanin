'''
'''
from ...utils import get_required, optimize, loadtxt, to_rrs, closest_wavelength
from scipy.interpolate import CubicSpline as Interpolate
from ....Benchmarks.multiple.QAA.model import model as QAA 

import numpy as np
# Define any optimizable parameters
@optimize(['a'],[0.24])
def model(Rrs, wavelengths, *args, **kwargs):


	eta = kwargs.get('a', 0.24)
	# beta = kwargs.get('b', 170)
	beta = 170
	Rrs_779 = get_required(Rrs, wavelengths, [779], 20)

	required = [620, 665, 709]
	tol = kwargs.get('tol', 5) # allowable difference from the required wavelengths
	Rrs = get_required(Rrs, wavelengths, required, tol) # get values as a function: Rrs(443)

	a_w_620 = 0.281
	a_w_665 = 0.401
	a_w_709 = 0.727

	b_b_779 = 1.61*Rrs_779(779)/(0.082-(0.6*Rrs_779(779)))
	a_ph_665 = 1.47*(((Rrs(709)/Rrs(665))*(a_w_709 + b_b_779))- a_w_665 - b_b_779)

	PC_RAD = beta*((((Rrs(709)/Rrs(620))*(a_w_709+b_b_779))-b_b_779-a_w_620)-(eta*a_ph_665)) # Equation 5b
	return PC_RAD