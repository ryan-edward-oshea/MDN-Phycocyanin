'''
'''

from ...utils import get_required, optimize




@optimize(['a', 'b'],[0.97, 0.000912])
def model(Rrs, wavelengths, *args, **kwargs):
	required = [622, 650]
	tol = kwargs.get('tol', 5)
	Rrs = get_required(Rrs, wavelengths, required, tol)

	a = kwargs.get('a', 0.97)
	b = kwargs.get('b', 0.000912)
	result = (Rrs(650)/Rrs(622) - a)/b
	return result