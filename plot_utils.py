from .metrics import slope, sspb, mdsa, rmsle, msa, SpR, performance,r_squared_linear,r_squared,SpR_log,rmse, mape
from .meta import get_sensor_label
from .utils import closest_wavelength, ignore_warnings
from collections import defaultdict as dd
from pathlib import Path
import numpy as np
import math
import datetime as dt
def add_identity(ax, *line_args, **line_kwargs):
	'''
	Add 1 to 1 diagonal line to a plot.
	https://stackoverflow.com/questions/22104256/does-matplotlib-have-a-function-for-drawing-diagonal-lines-in-axis-coordinates

	Usage: add_identity(plt.gca(), color='k', ls='--')
	'''
	line_kwargs['label'] = line_kwargs.get('label', '_nolegend_')
	identity, = ax.plot([], [], *line_args, **line_kwargs)

	def callback(axes):
		low_x, high_x = ax.get_xlim()
		low_y, high_y = ax.get_ylim()
		lo = max(low_x,  low_y)
		hi = min(high_x, high_y)
		identity.set_data([lo, hi], [lo, hi])

	callback(ax)
	ax.callbacks.connect('xlim_changed', callback)
	ax.callbacks.connect('ylim_changed', callback)

	ann_kwargs = {
		'transform'  : ax.transAxes,
		'textcoords' : 'offset points',
		'xycoords'   : 'axes fraction',
		'fontname'   : 'monospace',
		'xytext'     : (0,0),
		'zorder'     : 25,
		'va'         : 'top',
		'ha'         : 'left',
	}
	ax.annotate(r'$\mathbf{1:1}$', xy=(0.87,0.99), size=11, **ann_kwargs)


def _create_metric(metric, y_true, y_est, longest=None, label=None):
	''' Create a position-aligned string which shows the performance via a single metric '''
	if label == None:   label = metric.__name__.replace('SSPB', '\\beta').replace('MdSA', '\\varepsilon\\thinspace').replace('Slope','S \\qquad')
	if longest == None: longest = len(label)

	ispct = metric.__qualname__ in ['mape', 'sspb', 'mdsa']
	diff  = longest-len(label)
	space = r''.join([r'\ ']*diff + [r'\thinspace']*diff)
	prec  = 1 if ispct else 3
	stat  = f'{metric(y_true, y_est):.{prec}f}'
	perc  = r'$\small{\mathsf{\%}}$' if ispct else ''
	return rf'$\mathtt{{{label}}}{space}:$ {stat}{perc}'

def _create_stats(y_true, y_est, metrics, title=None):
	''' Create stat box strings for all metrics, assuming there is only a single target feature '''
	longest = max([len(metric.__name__.replace('SSPB', 'Bias').replace('MdSA', 'Error')) for metric in metrics])
	statbox = [_create_metric(m, y_true, y_est, longest=longest) for m in metrics]

	if title is not None:
		statbox = [rf'$\mathbf{{\underline{{{title}}}}}$'] + statbox
	return statbox

def _create_multi_feature_stats(y_true, y_est, metrics, labels=None):
	''' Create stat box strings for a single metric, assuming there are multiple target features '''
	if labels == None:
		labels = [f'Feature {i}' for i in range(y_true.shape[1])]
	assert(len(labels) == y_true.shape[1] == y_est.shape[1]), f'Number of labels does not match number of features: {labels} - {y_true.shape}'

	title   = metrics[0].__name__.replace('SSPB', 'Bias').replace('MdSA', 'Error')
	longest = max([len(label) for label in labels])
	statbox = [_create_metric(metrics[0], y1, y2, longest=longest, label=lbl) for y1, y2, lbl in zip(y_true.T, y_est.T, labels)]
	statbox = [rf'$\mathbf{{\underline{{{title}}}}}$'] + statbox
	return statbox

def add_stats_box(ax, y_true, y_est, metrics=[slope, sspb, mdsa, rmsle, msa], bottom_right=False, x=0.025, y=0.97, fontsize=16, label=None):
	''' Add a text box containing a variety of performance statistics, to the given axis '''
	import matplotlib.pyplot as plt
	plt.rc('text', usetex=True)
	plt.rcParams['mathtext.default']='regular'

	create_box = _create_stats if len(y_true.shape) == 1 or y_true.shape[1] == 1 else _create_multi_feature_stats
	stats_box  = '\n'.join( create_box(y_true, y_est, metrics, label) )
	ann_kwargs = {
		'transform'  : ax.transAxes,
		'textcoords' : 'offset points',
		'xycoords'   : 'axes fraction',
		'fontname'   : 'monospace',
		'xytext'     : (0,0),
		'zorder'     : 25,
		'va'         : 'top',
		'ha'         : 'left',
		'bbox'       : {
			'facecolor' : 'white',
			'edgecolor' : 'black',
			'alpha'     : 0.7,
		}
	}

	ann = ax.annotate(stats_box, xy=(x,y), size=fontsize, **ann_kwargs)

	# Switch location to (approximately) the bottom right corner
	if bottom_right:
		plt.gcf().canvas.draw()
		bbox_orig = ann.get_tightbbox(plt.gcf().canvas.renderer).transformed(ax.transAxes.inverted())

		new_x = 1 - (bbox_orig.x1 - bbox_orig.x0) + x
		new_y = bbox_orig.y1 - bbox_orig.y0 + (1 - y)
		ann.set_x(new_x)
		ann.set_y(new_y)
		ann.xy = (new_x - 0.04, new_y + 0.06)
	return ann


def draw_map(*lonlats, scale=0.2, world=False, us=True, eu=False, labels=[], ax=None, gray=False, res='i', **scatter_kws):
	''' Helper function to plot locations on a global map '''
	import matplotlib.pyplot as plt
	from matplotlib.transforms import Bbox
	from mpl_toolkits.axes_grid1.inset_locator import TransformedBbox, BboxPatch, BboxConnector
	from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, inset_axes
	from mpl_toolkits.basemap import Basemap
	from itertools import chain

	PLOT_WIDTH  = 8
	PLOT_HEIGHT = 6

	WORLD_MAP = {'cyl': [-90, 85, -180, 180]}
	US_MAP    = {
		'cyl' : [24, 49, -126, -65],
		'lcc' : [23, 48, -121, -64],
	}
	EU_MAP    = {
		'cyl' : [34, 65, -12, 40],
		'lcc' : [30.5, 64, -10, 40],
	}

	def mark_inset(ax, ax2, m, m2, MAP, loc1=(1, 2), loc2=(3, 4), **kwargs):
	    """
	    https://stackoverflow.com/questions/41610834/basemap-projection-geos-controlling-mark-inset-location
	    Patched mark_inset to work with Basemap.
	    Reason: Basemap converts Geographic (lon/lat) to Map Projection (x/y) coordinates

	    Additionally: set connector locations separately for both axes:
	        loc1 & loc2: tuple defining start and end-locations of connector 1 & 2
	    """
	    axzoom_geoLims = (MAP['cyl'][2:], MAP['cyl'][:2])
	    rect = TransformedBbox(Bbox(np.array(m(*axzoom_geoLims)).T), ax.transData)
	    pp   = BboxPatch(rect, fill=False, **kwargs)
	    ax.add_patch(pp)
	    p1 = BboxConnector(ax2.bbox, rect, loc1=loc1[0], loc2=loc1[1], **kwargs)
	    ax2.add_patch(p1)
	    p1.set_clip_on(False)
	    p2 = BboxConnector(ax2.bbox, rect, loc1=loc2[0], loc2=loc2[1], **kwargs)
	    ax2.add_patch(p2)
	    p2.set_clip_on(False)
	    return pp, p1, p2


	if world:
		MAP    = WORLD_MAP
		kwargs = {'projection': 'cyl', 'resolution': res}
	elif us:
		MAP    = US_MAP
		kwargs = {'projection': 'lcc', 'lat_0':30, 'lon_0':-98, 'resolution': res}#, 'epsg':4269}
	elif eu:
		MAP    = EU_MAP
		kwargs = {'projection': 'lcc', 'lat_0':48, 'lon_0':27, 'resolution': res}
	else:
		raise Exception('Must plot world, US, or EU')

	kwargs.update(dict(zip(['llcrnrlat', 'urcrnrlat', 'llcrnrlon', 'urcrnrlon'], MAP['lcc' if 'lcc' in MAP else 'cyl'])))
	if ax is None: f = plt.figure(figsize=(PLOT_WIDTH, PLOT_HEIGHT), edgecolor='w')
	m  = Basemap(ax=ax, **kwargs)
	ax = m.ax if m.ax is not None else plt.gca()

	if not world:
		m.readshapefile(Path(__file__).parent.joinpath('map_files', 'st99_d00').as_posix(), name='states', drawbounds=True, color='k', linewidth=0.5, zorder=11)
		m.fillcontinents(color=(0,0,0,0), lake_color='#9abee0', zorder=9)
		if not gray:
			m.drawrivers(linewidth=0.2, color='blue', zorder=9)
		m.drawcountries(color='k', linewidth=0.5)
	else:
		m.drawcountries(color='w')
	# m.bluemarble()
	if not gray:
		if us or eu: m.shadedrelief(scale=0.3 if world else 1)
		else:
			# m.arcgisimage(service='ESRI_Imagery_World_2D', xpixels = 2000, verbose= True)
			m.arcgisimage(service='World_Imagery', xpixels = 2000, verbose= True)
	else:
		pass
	# lats = m.drawparallels(np.linspace(MAP[0], MAP[1], 13))
	# lons = m.drawmeridians(np.linspace(MAP[2], MAP[3], 13))

	# lat_lines = chain(*(tup[1][0] for tup in lats.items()))
	# lon_lines = chain(*(tup[1][0] for tup in lons.items()))
	# all_lines = chain(lat_lines, lon_lines)

	# for line in all_lines:
	# 	line.set(linestyle='-', alpha=0.0, color='w')

	if labels:
		colors = ['aqua', 'orangered',  'xkcd:tangerine', 'xkcd:fresh green', 'xkcd:clay', 'magenta', 'xkcd:sky blue', 'xkcd:greyish blue', 'xkcd:goldenrod', ]
		markers = ['o', '^', 's', '*',  'v', 'X', '.', 'x',]
		mod_cr = False
		assert(len(labels) == len(lonlats)), [len(labels), len(lonlats)]
		for i, (label, lonlat) in enumerate(zip(labels, lonlats)):
			lonlat = np.atleast_2d(lonlat)
			if 'color' not in scatter_kws or mod_cr:
				scatter_kws['color'] = colors[i]
				scatter_kws['marker'] = markers[i]
				mod_cr = True
			ax.scatter(*m(lonlat[:,0], lonlat[:,1]), label=label, zorder=12, **scatter_kws)
		ax.legend(loc='lower left', prop={'weight':'bold', 'size':8}).set_zorder(20)

	else:
		for lonlat in lonlats:
			if len(lonlat):
				lonlat = np.atleast_2d(lonlat)
				s = ax.scatter(*m(lonlat[:,0], lonlat[:,1]), zorder=12, **scatter_kws)
				# plt.colorbar(s, ax=ax)
	hide_kwargs = {'axis':'both', 'which':'both'}
	hide_kwargs.update(dict([(k, False) for k in ['bottom', 'top', 'left', 'right', 'labelleft', 'labelbottom']]))
	ax.tick_params(**hide_kwargs)

	for axis in ['top','bottom','left','right']:
		ax.spines[axis].set_linewidth(1.5)
		ax.spines[axis].set_zorder(50)
	# plt.axis('off')

	if world:
		size = 0.35
		if us:
			loc = (0.25, -0.1) if eu else (0.35, -0.01)
			ax_ins = inset_axes(ax, width=PLOT_WIDTH*size, height=PLOT_HEIGHT*size, loc='center', bbox_to_anchor=loc, bbox_transform=ax.transAxes, axes_kwargs={'zorder': 5})

			scatter_kws.update({'s': 6})
			m2 = draw_map(*lonlats, labels=labels, ax=ax_ins, **scatter_kws)

			mark_inset(ax, ax_ins, m, m2, US_MAP, loc1=(1,1), loc2=(2,2), edgecolor='grey', zorder=3)
			mark_inset(ax, ax_ins, m, m2, US_MAP, loc1=[3,3], loc2=[4,4], edgecolor='grey', zorder=0)


		if eu:
			ax_ins = inset_axes(ax, width=PLOT_WIDTH*size, height=PLOT_HEIGHT*size, loc='center', bbox_to_anchor=(0.75, -0.05), bbox_transform=ax.transAxes, axes_kwargs={'zorder': 5})

			scatter_kws.update({'s': 6})
			m2 = draw_map(*lonlats, us=False, eu=True, labels=labels, ax=ax_ins, **scatter_kws)

			mark_inset(ax, ax_ins, m, m2, EU_MAP, loc1=(1,1), loc2=(2,2), edgecolor='grey', zorder=3)
			mark_inset(ax, ax_ins, m, m2, EU_MAP, loc1=[3,3], loc2=[4,4], edgecolor='grey', zorder=0)

	return m


def default_dd(d={}, f=lambda k: k):
	''' Helper function to allow defaultdicts whose default value returned is the queried key '''

	class key_dd(dd):
		''' DefaultDict which allows the key as the default value '''
		def __missing__(self, key):
			if self.default_factory is None:
				raise KeyError(key)
			val = self[key] = self.default_factory(key)
			return val
	return key_dd(f, d)


@ignore_warnings
def plot_scatter(y_test, benchmarks, bands, labels, products, sensor,return_opt,run_name=""):
	import matplotlib.patheffects as pe
	import matplotlib.ticker as ticker
	import matplotlib.pyplot as plt
	import seaborn as sns
	folder = Path('scatter_plots')
	folder.mkdir(exist_ok=True, parents=True)
	benchmarks_OG = benchmarks
	product_labels = default_dd({
		'chl' : 'Chl\\textit{a}',
#		'aph' : '\\textit{a}_{ph}',
		'PC'  : 'PC'
	})
	product_units = default_dd({
		'chl' : '[mg m^{-3}]',
		'PC' : '[mg m^{-3}]',

		'tss' : '[g m^{-3}]',
		'aph' : '[m^{-1}]',
	}, lambda k: '')
	model_labels = default_dd({
		'MDN' : 'MDN_{A}',
	})

	plt.rc('text', usetex=True)
	plt.rcParams['mathtext.default']='regular'
	plot_order = [] 
	if len(labels) > 3 and 'chl' not in products:
		product_bands = {
			'default' : [443, 482, 561, 655],
			# 'aph'     : [443, 530],
		}

		target     = [closest_wavelength(w, bands) for w in product_bands.get(products[0], product_bands['default'])]
		plot_label = [w in target for w in bands]
		plot_order = ['QAA', 'GIOP']
		plot_bands = True
	else:
		plot_label = [True] * len(labels)
		plot_order = {"chl" : "MDN-chl,Mishra_NDCI,Gilerson_2band,Gurlin_2band,Moses_2band,Yang_bandindex", 
					  "PC"  : "MDN-PC,Schalles,Sim2007,Hunter"
					 }		

		plot_bands = False

	#add MDN results to each sub-directory of results, we will plot each sub-directory separately
	for product_index, current_product in enumerate(products):
		benchmarks[current_product]['MDN-'+str(current_product)] = np.reshape(benchmarks['MDN'][...,product_index],(-1,1))

	labels = [(p,label) for label in labels for p in products if p in label]

	assert(len(labels) == y_test.shape[-1]), [len(labels), y_test.shape]

	plot_order_OG = plot_order
	
	
	#Produces a second set of results for the Optimized data, by amending _opt to the product names
	if return_opt:
		benchmark_options = ['','_opt']
	else:
		benchmark_options = ['']

	for benchmark_end in benchmark_options:
		plot_order = []
		benchmarks = []
		plot_order = plot_order_OG
		benchmarks = benchmarks_OG
		for current_product in products:
			plot_order[current_product] = ",".join(["".join([p, benchmark_end]) for p in plot_order_OG[current_product].split(",") if "".join([p, benchmark_end]) in benchmarks[current_product]]) 

			if benchmark_end =='_opt':
				plot_order[current_product] = "".join([ 'MDN-{},'.format(current_product), plot_order[current_product]])

		for plt_idx, (label, y_true) in enumerate(zip(labels, y_test.T)):
			if not plot_label[plt_idx]: continue

			product, title = label
			benchmarks=benchmarks_OG[product]

			fig_size   = 5
			n_col      = 2
			total_products      = len(plot_order[product].split(","))
			n_row      = math.ceil(float(total_products)/float(n_col))

			if plot_bands:
				n_col, n_row = n_row, n_col

			fig, axes = plt.subplots(n_row, n_col, figsize=(fig_size*n_col, fig_size*n_row))
			axes      = [ax for axs in np.atleast_1d(axes) for ax in np.atleast_1d(axs)]
			colors    = ['xkcd:sky blue', 'xkcd:tangerine', 'xkcd:fresh green', 'xkcd:greyish blue', 'xkcd:goldenrod',  'xkcd:clay', 'xkcd:bluish purple', 'xkcd:reddish', 'xkcd:neon purple']

			print('Order:', plot_order)
			print(f'Plot size: {n_row} x {n_col}')

			curr_idx = 0
			full_ax  = fig.add_subplot(111, frameon=False)
			full_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False, pad=10)

			plabel = f'{product_labels[products[plt_idx]]} {product_units[products[plt_idx]]}'
			xlabel = fr'$\mathbf{{Measured {plabel}}}$'
			ylabel = fr'$\mathbf{{Modeled {plabel}}}$'
			full_ax.set_xlabel(xlabel.replace(' ', '\ '), fontsize=20, labelpad=10)
			full_ax.set_ylabel(ylabel.replace(' ', '\ '), fontsize=20, labelpad=10)

			s_lbl = get_sensor_label(sensor).replace('-',' ')
			n_pts = len(y_test)
			mean_test = round(np.mean(y_true),1)
			std_test = round(np.std(y_true),1)

			title = fr'$\mathbf{{\underline{{\large{{{s_lbl}}}}}}}$' + '\n' + fr'$\small{{\mathit{{N\small{{=}}}}{n_pts}}}$'
			for est_idx, est_lbl in enumerate(plot_order[product].split(",")):
				y_est = benchmarks[est_lbl][..., 0] 

				ax    = axes[curr_idx]
				cidx  = (curr_idx % n_col) if plot_bands else curr_idx
				color = colors[cidx]

				first_row = curr_idx < n_col 
				last_row  = curr_idx >= ((n_row-1)*n_col) 
				first_col = (curr_idx % n_col) == 0
				last_col  = ((curr_idx+1) % n_col) == 0

				y_est_log  = np.log10(y_est).flatten()
				y_true_log = np.log10(y_true).flatten()
				curr_idx  += 1

				l_kws = {'color': color, 'path_effects': [pe.Stroke(linewidth=4, foreground='k'), pe.Normal()], 'zorder': 22, 'lw': 1}
				s_kws = {'alpha': 0.4, 'color': color}

				if est_lbl == 'MDN':
					[i.set_linewidth(5) for i in ax.spines.values()]
					est_lbl = 'MDN_{A}'
					est_lbl = 'MDN-I'
				else:
					est_lbl = est_lbl.replace('Gons_2band', 'Gons').replace('Gilerson_2band', 'GI2B').replace('Smith_','').replace('_','-').replace('Mishra','Mishra \: et \: al. \: 2009 \:').replace('Schalles','Schalles  \: et  \: al.  \: 2000  \:').replace('Hunter','Hunter  \: et  \: al.  \: 2010').replace('Sim2007','Simis  \: et  \: al.  \: 2007')

				if product not in ['chl', 'tss','PC'] and last_col:
					ax2 = ax.twinx()
					ax2.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False, pad=0)
					ax2.grid(False)
					ax2.set_yticklabels([])
					ax2.set_ylabel(fr'$\mathbf{{{bands[plt_idx]:.0f}nm}}$', fontsize=20)

				minv = int(np.nanmin(y_true_log)) - 1 if product != 'aph' else -4
				maxv = int(np.nanmax(y_true_log)) + 1 if product != 'aph' else 1
				loc  = ticker.LinearLocator(numticks=maxv-minv+1)
				fmt  = ticker.FuncFormatter(lambda i, _: r'$10$\textsuperscript{%i}'%i)

				ax.set_ylim((minv, maxv))
				ax.set_xlim((minv, maxv))
				ax.xaxis.set_major_locator(loc)
				ax.yaxis.set_major_locator(loc)
				ax.xaxis.set_major_formatter(fmt)
				ax.yaxis.set_major_formatter(fmt)

				if not last_row:  ax.set_xticklabels([])
				if not first_col: ax.set_yticklabels([])

				valid = np.logical_and(np.isfinite(y_true_log), np.isfinite(y_est_log))
				if valid.sum():
					sns.regplot(y_true_log[valid], y_est_log[valid], ax=ax, scatter_kws=s_kws, line_kws=l_kws, fit_reg=True, truncate=False, robust=True, ci=None)
					kde = sns.kdeplot(y_true_log[valid], y_est_log[valid], shade=False, ax=ax, bw='scott', n_levels=4, legend=False, gridsize=100, color=color)
					kde.collections[2].set_alpha(0)

				if len(valid.flatten()) != valid.sum():
					ax.scatter(y_true_log[~valid], [minv]*(~valid).sum(), color='r', alpha=0.4, label=r'$\mathbf{%s\ invalid, %s\ nan}$' % ((~valid).sum(), np.isnan(y_true_log[~valid]).sum()) )
					ax.legend(loc='lower right', prop={'weight':'bold', 'size': 16})

				add_identity(ax, ls='--', color='k', zorder=20)
				add_stats_box(ax, y_true, y_est,metrics=[mdsa, sspb, slope])

				if first_row or not plot_bands:
					ax.set_title(fr'$\mathbf{{\large{{{est_lbl}}}}}$', fontsize=18)

				ax.tick_params(labelsize=18)
				ax.grid('on', alpha=0.3)

			filename = folder.joinpath(f'{run_name}_{products}_{sensor}_{product}_mean_{mean_test}_N_{n_pts}_std_{std_test}_test{benchmark_end}.jpg')
			plt.tight_layout()
			plt.savefig(filename.as_posix(), dpi=600, bbox_inches='tight', pad_inches=0.1,)

@ignore_warnings
def plot_remote_insitu(y_remote, y_insitu, dictionary_of_matchups=None, products='chl', sensor='HICO',run_name=""):
	y_remote_OG =y_remote
	y_insitu_OG=y_insitu

	import matplotlib.patheffects as pe
	import matplotlib.ticker as ticker
	import matplotlib.pyplot as plt
	import seaborn as sns
	from pylab import text

	folder = Path('scatter_plots')
	folder.mkdir(exist_ok=True, parents=True)
	n_row = 1
	n_col = 2
	fig_size   = 5
	plt_idx = 0
	plt.rc('text', usetex=True)
	plt.rcParams['mathtext.default']='regular'

	fig, axes = plt.subplots(n_row, n_col, figsize=(fig_size*n_col, fig_size*n_row))
	axes      = [ax for axs in np.atleast_1d(axes) for ax in np.atleast_1d(axs)]
	colors    = ['xkcd:sky blue', 'xkcd:tangerine', 'xkcd:fresh green', 'xkcd:greyish blue', 'xkcd:goldenrod',  'xkcd:clay', 'xkcd:bluish purple', 'xkcd:reddish', 'xkcd:neon purple']

	full_ax  = fig.add_subplot(111, frameon=False)
	full_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False, pad=10)

	product_labels = default_dd({
		'chl' : 'Chl\\textit{a}',
		'PC'  : 'PC'
	})

	product_units = default_dd({
		'chl' : '[mg m^{-3}]',
		'PC' : '[mg m^{-3}]',

		'tss' : '[g m^{-3}]',
		'aph' : '[m^{-1}]',
	}, lambda k: '')

	font= {'size':15}
	for current_product in products:
		print(plt_idx,'current product:',current_product )
		y_remote = np.squeeze(np.asarray(y_remote_OG[plt_idx]))
		y_insitu = np.squeeze(np.asarray(y_insitu_OG[plt_idx]))



		plabel_1 = f'{product_labels[products[plt_idx]]}'
		plabel_2 = f'{product_units[products[plt_idx]]}'
		xlabel = fr'$\mathbf{{{plabel_1}\textsuperscript{{e}}{plabel_2}}}$'
		ylabel = fr'$\mathbf{{{plabel_1}\textsuperscript{{r}}{plabel_2}}}$'


		s_lbl = get_sensor_label(sensor).replace('-',' ')
		n_pts = len(y_insitu)
		title = fr'$\mathbf{{\underline{{\large{{{s_lbl}}}}}}}$' + '\n' + fr'$\small{{\mathit{{N\small{{=}}}}{n_pts}}}$'
		full_ax.set_title(title.replace(' ', '\ '), fontsize=24, y=1.04)

		curr_idx = 0
		cidx  = plt_idx
		color = colors[cidx]
		l_kws = {'color': color, 'path_effects': [pe.Stroke(linewidth=4, foreground='k'), pe.Normal()], 'zorder': 22, 'lw': 1}
		s_kws = {'alpha': 0.4, 'color': color}

		ax = axes[plt_idx]
		ax.set_xlabel(xlabel.replace(' ', '\ '), fontsize=20, labelpad=10)
		ax.set_ylabel(ylabel.replace(' ', '\ '), fontsize=20, labelpad=10)


		y_true_log = np.log10(y_insitu).flatten()
		y_est_log = np.log10(y_remote).flatten()
		minv = int(np.nanmin(y_true_log)) - 1
		maxv = int(np.nanmax(y_true_log)) + 1
		loc  = ticker.LinearLocator(numticks=maxv-minv+1)
		fmt  = ticker.FuncFormatter(lambda i, _: r'$10$\textsuperscript{%i}'%i)

		ax.set_ylim((minv, maxv))
		ax.set_xlim((minv, maxv))
		ax.xaxis.set_major_locator(loc)
		ax.yaxis.set_major_locator(loc)
		ax.xaxis.set_major_formatter(fmt)
		ax.yaxis.set_major_formatter(fmt)
		valid = np.logical_and(np.isfinite(y_true_log), np.isfinite(y_est_log))
		print('valid matchups:',sum(valid))

		if valid.sum():
			sns.regplot(y_true_log[valid], y_est_log[valid], ax=ax, scatter_kws=s_kws, line_kws=l_kws, fit_reg=True, truncate=False, robust=True, ci=None)
			kde = sns.kdeplot(y_true_log[valid], y_est_log[valid], shade=False, ax=ax, bw='scott', n_levels=4, legend=False, gridsize=100, color=color)
			kde.collections[2].set_alpha(0)

		if len(valid.flatten()) != valid.sum():
			ax.scatter(y_true_log[~valid], [minv]*(~valid).sum(), color='r', alpha=0.4, label=r'$\mathbf{%s\ invalid, %s\ nan}$' % ((~valid).sum(), np.isnan(y_true_log[~valid]).sum()) ) 
			ax.legend(loc='lower right', prop={'weight':'bold', 'size': 16})

		add_identity(ax, ls='--', color='k', zorder=20)
		add_stats_box(ax, y_insitu.flatten()[valid], y_remote.flatten()[valid])

		ax.tick_params(labelsize=12)
		ax.grid('on', alpha=0.3)

		filename = folder.joinpath(f'remote_vs_insitu_summary_{run_name}_{products}_{sensor}.jpg')
		plt.tight_layout()
		plt_idx = plt_idx+1

	plt.savefig(filename.as_posix(), dpi=600, bbox_inches='tight', pad_inches=0.1,)

	plt_idx = 0 

	n_row = 4
	n_col = 4
	fig_size   = 5
	plt_idx = 0
	plt.rc('text', usetex=True)
	plt.rcParams['mathtext.default']='regular'

	fig, axes = plt.subplots(n_row, n_col, figsize=(fig_size*n_col, fig_size*n_row))
	axes      = [ax for axs in np.atleast_1d(axes) for ax in np.atleast_1d(axs)]
	colors    = ['xkcd:sky blue', 'xkcd:tangerine', 'xkcd:fresh green', 'xkcd:greyish blue', 'xkcd:goldenrod',  'xkcd:clay', 'xkcd:bluish purple', 'xkcd:reddish', 'xkcd:neon purple']

	full_ax  = fig.add_subplot(111, frameon=False)
	full_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False, pad=10)

	xlabel = fr'$\mathbf{{Band [nm]}}$'
	ylabel = fr'$\mathbf{{R\textsubscript{{rs}} [sr\textsuperscript{{-1}}]}}$'
	full_ax.set_xlabel(xlabel.replace(' ', '\ '), fontsize=20, labelpad=10)
	full_ax.set_ylabel(ylabel.replace(' ', '\ '), fontsize=20, labelpad=30)
	
	site_labels_of_interest = ['WE2','WE6','WE13', 'Lake Erie St. 970','St. Andrews Bay (SA11)\nApr. 14, 2010','Pensacola Bay (PB09)\nAug. 26, 2011','Pensacola Bay (PB05)\nAug. 26, 2011','Pensacola Bay (PB04)\nAug. 26, 2011', 'Pensacola Bay (PB14)\nJun. 02, 2011','Pensacola Bay (PB08)\nJun. 02, 2011','Choctawhatchee Bay (CH01)\nJul. 30, 2011','Choctawhatchee Bay (CH03)\nJul. 30, 2011','WE4','WE8','Gulf_Mexico 72','Gulf_Mexico 82']

	site_labels_of_interest_no_newline_dict = {
	'WE2' : 'WE2',
	'WE6' : 'WE6',
	'WE13' : 'WE13',
	'WE4' : 'WE4',
	'WE8' : 'WE8',

	'Lake Erie St. 970' : 'Lake Erie St. 970',
	'St. Andrews Bay (SA11)\nApr. 14, 2010' : 'St. Andrews Bay (SA11)',
	'Pensacola Bay (PB14)\nJun. 02, 2011' : 'Pensacola Bay (PB14)',
	'Pensacola Bay (PB06)\nJun. 02, 2011' : 'Pensacola Bay (PB06)',
	'Pensacola Bay (PB04)\nAug. 26, 2011' : 'Pensacola Bay (PB04)',
	'Pensacola Bay (PB08)\nJun. 02, 2011' : 'Pensacola Bay (PB08)',
	'Pensacola Bay (PB05)\nAug. 26, 2011' : 'Pensacola Bay (PB05)',

	'Pensacola Bay (PB09)\nAug. 26, 2011' : 'Pensacola Bay (PB09)',
	'Choctawhatchee Bay (CH01)\nJul. 30, 2011' : 'Choctawhatchee Bay (CH01)',
	'Choctawhatchee Bay (CH03)\nJul. 30, 2011' : 'Choctawhatchee Bay (CH03)',
	'Gulf_Mexico 72' : 'Gulf of Mexico 72',
	'Gulf_Mexico 82' :'Gulf of Mexico 82',
	}

	def try_to_parse_date(input_text):
		for fmt in ('[\'%Y%m%d %H:%M\']','[\'%Y-%m-%d %H:%M\']','[\'%Y%m%d\']'):
			try:
				return dt.datetime.strptime(input_text,fmt)
			except ValueError:
				pass
		raise ValueError('No Valid date format found')

	round_digits = 1


	for plotting_label_current in site_labels_of_interest:
		if plotting_label_current in dictionary_of_matchups['plotting_labels']:
			index_of_plotting_label = np.where(plotting_label_current == dictionary_of_matchups['plotting_labels'])
			index = index_of_plotting_label[0][0]
		else:
			print("NOT IN DICTIONARY")
			continue

		ax = axes[plt_idx]
		first_row = plt_idx < n_col
		last_row  = plt_idx >= ((n_row-1)*n_col)
		first_col = (plt_idx % n_col) == 0
		last_col  = ((plt_idx+1) % n_col) == 0

		if not last_row:  ax.set_xticklabels([])
		if not first_col: ax.set_yticklabels([])

		chl_truth = round(np.asscalar(dictionary_of_matchups['chl'][index]),round_digits)
		PC_truth = round(np.asscalar(dictionary_of_matchups['PC'][index]),round_digits)
		chl_remote_estimate = round(np.asscalar(np.squeeze(np.asarray(y_remote_OG[0]))[index]),round_digits)
		PC_remote_estimate = round(np.asscalar(np.squeeze(np.asarray(y_remote_OG[1]))[index]),round_digits)
		chl_insitu_estimate = round(np.asscalar(np.squeeze(np.asarray(y_insitu_OG[0]))[index]),round_digits)
		PC_insitu_estimate = round(np.asscalar(np.squeeze(np.asarray(y_insitu_OG[1]))[index]),round_digits)

		text_label = fr'PC: {PC_truth}'+ '\n'  + fr'PC\textsuperscript{{e}}: {PC_insitu_estimate}' + '\n' + fr'PC\textsuperscript{{r}}: {PC_remote_estimate}'
		text(0.9,0.905,text_label,horizontalalignment='center',verticalalignment='center',transform=ax.transAxes,backgroundcolor='1.0',bbox=dict(facecolor='white',edgecolor='black',boxstyle='round'),fontdict=font)
		plot_label ='ABCDEFGHIJKLMNOPQRST'
		text(0.03,0.96,plot_label[plt_idx],horizontalalignment='center',verticalalignment='center',transform=ax.transAxes,backgroundcolor='1.0',bbox=dict(facecolor='white',edgecolor='black',boxstyle='round'),fontdict=font)

		insitu_Rrs = dictionary_of_matchups['insitu_Rrs_resampled_full'][index,:]
		insitu_Rrs_wvl = dictionary_of_matchups['insitu_Rrs_resampled_wvl_full'][0,:]
		retrieved_Rrs = dictionary_of_matchups['Rrs_retrieved_full'][index,:]
		retrieved_Rrs_wvl = dictionary_of_matchups['Rrs_retrieved_wvl_full'][0,:]
		site_label = dictionary_of_matchups['site_label'][index,:]
		plotting_label = str(dictionary_of_matchups['plotting_labels'][index,:])

		date_time = str(dictionary_of_matchups['insitu_datetime'][index,:])
		reformatted_datetime = try_to_parse_date(date_time) 
		reformatted_datetime = reformatted_datetime.strftime('%b, %d, %Y ')

		ax.plot(insitu_Rrs_wvl,insitu_Rrs,'-o',color='b', alpha=0.4,label=fr'Rrs')
		ax.plot(retrieved_Rrs_wvl,retrieved_Rrs,'-o',color='r', alpha=0.4,label=fr'\^{{R}}rs')
		ax.set_ylim((0.0,0.025))
		plotting_label = site_labels_of_interest_no_newline_dict[plotting_label_current]

		title = fr'$\mathbf{{{{\large{{{plotting_label}}}}}}}$' + '\n' + fr'$\small{{{reformatted_datetime}}}$'
		ax.tick_params(labelsize=20)
		ax.grid('on', alpha=0.3)

		ax.set_title(title.replace(' ', '\ '), fontsize=18)
		filename = folder.joinpath(f'remote_vs_insitu_{run_name}_{products}_{sensor}.jpg')
		plt.tight_layout()
		plt_idx = plt_idx+1
		if plt_idx ==1:
			ax.legend(loc='upper center',fontsize=16)
	plt.savefig(filename.as_posix(), dpi=600, bbox_inches='tight', pad_inches=0.1,)
	plt.close()

@ignore_warnings
def plot_remote_insitu_summary(y_remote, y_insitu, dictionary_of_matchups=None, products='chl', sensor='HICO',run_name=""):
	y_remote_OG =y_remote
	y_insitu_OG=y_insitu
	
	import matplotlib.patheffects as pe
	import matplotlib.ticker as ticker
	import matplotlib.pyplot as plt
	import seaborn as sns
	from pylab import text

	folder = Path('scatter_plots')
	folder.mkdir(exist_ok=True, parents=True)
	if len(y_remote_OG) == 1:
		n_row = 1
		n_col = 2
	else:
		n_row = 3
		n_col = 3
	fig_size   = 5
	plt_idx = 0
	plt.rc('text', usetex=True)
	plt.rcParams['mathtext.default']='regular'


	available_PC_matchups = {
    'WE2' : 'WE2',
    'WE6' : 'WE6' ,
    'WE13': 'WE13',
    'WE4' : 'WE4',
    'WE8' : 'WE8',

	}

	product_labels = default_dd({
		'chl' : 'Chl\\textit{a}',
		'PC'  : 'PC'
	})

	product_units = default_dd({
		'chl' : '[mg m^{-3}]',
		'PC' : '[mg m^{-3}]',

		'tss' : '[g m^{-3}]',
		'aph' : '[m^{-1}]',
	}, lambda k: '')

	font= {'size':15}
	products = products.split(',')

	for product_counter,current_product in enumerate(products):
		fig, axes = plt.subplots(n_row, n_col, figsize=(fig_size*n_col, fig_size*n_row))
		axes      = [ax for axs in np.atleast_1d(axes) for ax in np.atleast_1d(axs)]
		colors    = ['xkcd:sky blue', 'xkcd:tangerine', 'xkcd:fresh green', 'xkcd:greyish blue', 'xkcd:goldenrod',  'xkcd:clay', 'xkcd:bluish purple', 'xkcd:reddish', 'xkcd:neon purple']

		full_ax  = fig.add_subplot(111, frameon=False)
		full_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False, pad=10)

		plt_idx = 1
		for run_counter,current_run in enumerate(y_remote_OG):
			current_y_remote = y_remote_OG[run_counter]
			current_y_in_situ = y_insitu_OG[run_counter]
			current_sensor = sensor[run_counter]
			current_dictionary = dictionary_of_matchups[run_counter]
			current_run_name = run_name[run_counter]

			print(plt_idx,'current product:',current_product )
			y_remote = np.squeeze(np.asarray(current_y_remote[product_counter]))
			y_insitu = np.squeeze(np.asarray(current_y_in_situ[product_counter]))

			plabel_1 = f'{product_labels[products[product_counter]]}'
			plabel_2 = f'{product_units[products[product_counter]]}'
			xlabel = fr'$\mathbf{{{plabel_1}\textsuperscript{{e}}{plabel_2}}}$'
			ylabel = fr'$\mathbf{{{plabel_1}\textsuperscript{{r}}{plabel_2}}}$'

			s_lbl = get_sensor_label(current_sensor).replace('-',' ')
			n_pts = len(y_insitu)
			title = fr'$\mathbf{{\underline{{\large{{{s_lbl}}}}}}}$' + '\n' + fr'$\small{{\mathit{{N\small{{=}}}}{n_pts}}}$'
			plain_text_title = 'Remote vs. In-Situ Matchup'
			title = fr'$\mathbf{{\underline{{\large{{{plain_text_title}}}}}}}$'
			if len(y_remote_OG) != 1:
				full_ax.set_title(title.replace(' ', '\ '), fontsize=24, y=1.04)


			curr_idx = 0
			cidx  = plt_idx
			color = colors[cidx]
			l_kws = {'color': color, 'path_effects': [pe.Stroke(linewidth=4, foreground='k'), pe.Normal()], 'zorder': 22, 'lw': 1}
			s_kws = {'alpha': 0.4, 'color': color}

			ax = axes[plt_idx]
			local_title = current_run_name + ' (N={})'.format(n_pts) 
			if len(y_remote_OG) != 1:
				ax.set_title(local_title.replace(' ', '\ ').replace('_', '\ '),fontsize=20)
			ax.set_xlabel(xlabel.replace(' ', '\ '), fontsize=20, labelpad=10)
			
			y_true_log = np.log10(y_insitu).flatten()
			y_est_log = np.log10(y_remote).flatten()


			minv = int(np.nanmin(y_true_log)) - 1
			maxv = int(np.nanmax(y_true_log)) + 1
			loc  = ticker.LinearLocator(numticks=maxv-minv+1)
			fmt  = ticker.FuncFormatter(lambda i, _: r'$10$\textsuperscript{%i}'%i)

			ax.set_ylim((minv, maxv))
			ax.set_xlim((minv, maxv))
			ax.xaxis.set_major_locator(loc)
			ax.yaxis.set_major_locator(loc)
			ax.xaxis.set_major_formatter(fmt)
			ax.yaxis.set_major_formatter(fmt)
			valid = np.logical_and(np.isfinite(y_true_log), np.isfinite(y_est_log))
			if valid.sum():
				sns.regplot(y_true_log[valid], y_est_log[valid], ax=ax, scatter_kws=s_kws, line_kws=l_kws, fit_reg=True, truncate=False, robust=True, ci=None)
				kde = sns.kdeplot(y_true_log[valid], y_est_log[valid], shade=False, ax=ax, bw='scott', n_levels=4, legend=False, gridsize=100, color=color)
				kde.collections[2].set_alpha(0)
				if current_product == 'PC':
					y_measured_log = np.log10(current_dictionary['PC']).flatten()
					valid_PC_measured = np.logical_and(np.isfinite(y_true_log), np.isfinite(y_measured_log))
					l_kws = {'color': color, 'path_effects': [pe.Stroke(linewidth=4, foreground='k'), pe.Normal()], 'zorder': 22, 'lw': 1}
					s_kws = {'alpha': 0.8, 'color': color, 'edgecolor': 'black'}
					
					ax.scatter(y_true_log[valid_PC_measured],y_est_log[valid_PC_measured],alpha= 0.8, color= color, edgecolor= 'black',s=65)
					
					valid_PC_measured = np.logical_and(np.isfinite(y_est_log), np.isfinite(y_measured_log))

					l_kws = {'color': 'xkcd:electric pink', 'path_effects': [pe.Stroke(linewidth=4, foreground='k'), pe.Normal()], 'zorder': 22, 'lw': 1}
					s_kws = {'alpha': 0.8, 'color': 'xkcd:electric pink', 'edgecolor': 'black'}
					ax.scatter(y_measured_log[valid_PC_measured],y_est_log[valid_PC_measured],alpha= 0.8, color= 'xkcd:electric pink', edgecolor= 'black')

					print(performance('HICO Rrs Performance',pow(10,y_measured_log[valid_PC_measured]),pow(10,y_est_log[valid_PC_measured]),metrics=[mdsa, sspb, slope, rmsle, msa]),'N=',len(y_measured_log[valid_PC_measured]))
					print(performance('In-Situ measured Rrs Performance',pow(10,y_measured_log[valid_PC_measured]),pow(10,y_true_log[valid_PC_measured]),metrics=[mdsa, sspb, slope, rmsle, msa]),'N=',len(y_measured_log[valid_PC_measured]))
					print('Printed Performance Metrics')
					
			if len(y_remote_OG) != 1:
				if len(valid.flatten()) != valid.sum():
					ax.scatter(y_true_log[~valid], [minv]*(~valid).sum(), color='r', alpha=0.4, label=r'$\mathbf{%s\ invalid, %s\ nan}$' % ((~valid).sum(), np.isnan(y_true_log[~valid]).sum()) ) 
					ax.legend(loc='lower right', prop={'weight':'bold', 'size': 16})


			add_identity(ax, ls='--', color='k', zorder=20)
			add_stats_box(ax, y_insitu.flatten()[valid], y_remote.flatten()[valid],metrics=[r_squared,SpR, slope])

			plot_label ='ABCDEFGHIJKLMNOPQRST'
			text(0.965,0.02,plot_label[plt_idx],horizontalalignment='center',verticalalignment='bottom',transform=ax.transAxes,backgroundcolor='1.0',bbox=dict(facecolor='white',edgecolor='black',boxstyle='round'),fontdict=font)

			ax.tick_params(labelsize=12)
			ax.grid('on', alpha=0.3)
			ax.set_yticklabels([])

			filename = folder.joinpath(f'Summary_of_remote_vs_insitu_{products}_{current_sensor}_{current_product}.jpg')
			plt.tight_layout()
			plt_idx = plt_idx+1
			
		plt.savefig(filename.as_posix(), dpi=600, bbox_inches='tight', pad_inches=0.1,)
		plt.close()

@ignore_warnings
def plot_remote_insitu_summary_comparison(y_remote, y_insitu, dictionary_of_matchups=None, products='chl', sensor='HICO',run_name="",y_remote_Rrs=None, y_insitu_Rrs=None, dictionary_of_matchups_Rrs=None,sensor_Rrs=None,run_name_Rrs=None):
	y_remote_OG =y_remote
	y_insitu_OG=y_insitu

	y_remote_OG_Rrs =y_remote_Rrs
	y_insitu_OG_Rrs=y_insitu_Rrs

	import matplotlib.patheffects as pe
	import matplotlib.ticker as ticker
	import matplotlib.pyplot as plt
	import seaborn as sns
	from pylab import text

	folder = Path('scatter_plots')
	folder.mkdir(exist_ok=True, parents=True)
	if len(y_remote_OG) == 1:
		n_row = 1
		n_col = 2
	else:
		n_row = 3
		n_col = 3
	fig_size   = 5
	plt_idx = 0
	plt.rc('text', usetex=True)
	plt.rcParams['mathtext.default']='regular'

	available_PC_matchups = {
    'WE2' : 'WE2',
    'WE6' : 'WE6' ,
    'WE13': 'WE13',
    'WE4' : 'WE4',
    'WE8' : 'WE8',
	}

	product_labels = default_dd({
		'chl' : 'Chl\\textit{a}',
		'PC'  : 'PC'
	})

	product_units = default_dd({
		'chl' : '[mg m^{-3}]',
		'PC' : '[mg m^{-3}]',

		'tss' : '[g m^{-3}]',
		'aph' : '[m^{-1}]',
	}, lambda k: '')

	font= {'size':15}
	products = products.split(',')

	for product_counter,current_product in enumerate(products):
		fig, axes = plt.subplots(n_row, n_col, figsize=(fig_size*n_col, fig_size*n_row))
		axes      = [ax for axs in np.atleast_1d(axes) for ax in np.atleast_1d(axs)]
		colors    = ['xkcd:sky blue', 'xkcd:tangerine', 'xkcd:fresh green', 'xkcd:greyish blue', 'xkcd:goldenrod',  'xkcd:clay', 'xkcd:bluish purple', 'xkcd:reddish', 'xkcd:neon purple']

		full_ax  = fig.add_subplot(111, frameon=False)
		full_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False, pad=10)

		plt_idx = 0
		for run_counter in range(2):
			if plt_idx == 1:
				current_y_remote = y_remote_OG_Rrs[0]
				current_y_in_situ = y_insitu_OG_Rrs[0]
				current_sensor = sensor_Rrs[0]
				current_dictionary = dictionary_of_matchups_Rrs[0]
				current_run_name = run_name_Rrs[0]
			else:
				current_y_remote = y_remote_OG[run_counter]
				current_y_in_situ = y_insitu_OG[run_counter]
				current_sensor = sensor[run_counter]
				current_dictionary = dictionary_of_matchups[run_counter]
				current_run_name = run_name[0]

			y_remote = np.squeeze(np.asarray(current_y_remote[product_counter]))
			y_insitu = np.squeeze(np.asarray(current_y_in_situ[product_counter]))

			plabel_1 = f'{product_labels[products[product_counter]]}'
			plabel_2 = f'{product_units[products[product_counter]]}'
			xlabel = fr'$\mathbf{{{plabel_1}\textsuperscript{{e}}{plabel_2}}}$'
			ylabel = fr'$\mathbf{{{plabel_1}\textsuperscript{{r}}{plabel_2}}}$'

			s_lbl = get_sensor_label(current_sensor).replace('-',' ')
			n_pts = len(y_insitu)
			title = fr'$\mathbf{{\underline{{\large{{{s_lbl}}}}}}}$' + '\n' + fr'$\small{{\mathit{{N\small{{=}}}}{n_pts}}}$'
			plain_text_title = 'Remote vs. In-Situ Matchup'
			title = fr'$\mathbf{{\underline{{\large{{{plain_text_title}}}}}}}$'
			if len(y_remote_OG) != 1:
				full_ax.set_title(title.replace(' ', '\ '), fontsize=24, y=1.04)
				
			curr_idx = 0
			cidx  = plt_idx
			color = colors[cidx]
			l_kws = {'color': color, 'path_effects': [pe.Stroke(linewidth=4, foreground='k'), pe.Normal()], 'zorder': 22, 'lw': 1}
			s_kws = {'alpha': 0.4, 'color': color}

			ax = axes[plt_idx] 
			local_title = current_run_name + ' (N={})'.format(n_pts)
			if len(y_remote_OG) != 1:
				ax.set_title(local_title.replace(' ', '\ ').replace('_', '\ '),fontsize=20)
			ax.set_xlabel(xlabel.replace(' ', '\ '), fontsize=20, labelpad=10)
			if plt_idx == 0:
				ax.set_ylabel(ylabel.replace(' ', '\ '), fontsize=20, labelpad=10)
			y_true_log = np.log10(y_insitu).flatten()
			y_est_log = np.log10(y_remote).flatten()

			minv = int(np.nanmin(y_true_log)) - 1
			maxv = int(np.nanmax(y_true_log)) + 1
			loc  = ticker.LinearLocator(numticks=maxv-minv+1)
			fmt  = ticker.FuncFormatter(lambda i, _: r'$10$\textsuperscript{%i}'%i)

			ax.set_ylim((minv, maxv))
			ax.set_xlim((minv, maxv))
			ax.xaxis.set_major_locator(loc)
			ax.yaxis.set_major_locator(loc)
			ax.xaxis.set_major_formatter(fmt)
			ax.yaxis.set_major_formatter(fmt)
			valid = np.logical_and(np.isfinite(y_true_log), np.isfinite(y_est_log))
			if valid.sum():
				sns.regplot(y_true_log[valid], y_est_log[valid], ax=ax, scatter_kws=s_kws, line_kws=l_kws, fit_reg=True, truncate=False, robust=True, ci=None)
				kde = sns.kdeplot(y_true_log[valid], y_est_log[valid], shade=False, ax=ax, bw='scott', n_levels=4, legend=False, gridsize=100, color=color)
				kde.collections[2].set_alpha(0)
				if current_product == 'PC':
					y_measured_log = np.log10(current_dictionary['PC']).flatten()
					valid_PC_measured = np.logical_and(np.isfinite(y_true_log), np.isfinite(y_measured_log))
					l_kws = {'color': color, 'path_effects': [pe.Stroke(linewidth=4, foreground='k'), pe.Normal()], 'zorder': 22, 'lw': 1}
					s_kws = {'alpha': 0.8, 'color': color, 'edgecolor': 'black'}
					
					ax.scatter(y_true_log[valid_PC_measured],y_est_log[valid_PC_measured],alpha= 0.8, color= color, edgecolor= 'black',s=65)
					
					valid_PC_measured = np.logical_and(np.isfinite(y_est_log), np.isfinite(y_measured_log))

					l_kws = {'color': 'xkcd:electric pink', 'path_effects': [pe.Stroke(linewidth=4, foreground='k'), pe.Normal()], 'zorder': 22, 'lw': 1}
					s_kws = {'alpha': 0.8, 'color': 'xkcd:electric pink', 'edgecolor': 'black'}
					ax.scatter(y_measured_log[valid_PC_measured],y_est_log[valid_PC_measured],alpha= 0.8, color= 'xkcd:electric pink', edgecolor= 'black')

					print(performance('HICO Rrs Performance',pow(10,y_measured_log[valid_PC_measured]),pow(10,y_est_log[valid_PC_measured]),metrics=[mdsa, sspb, slope, rmsle, msa]),'N=',len(y_measured_log[valid_PC_measured]))
					print(performance('In-Situ measured Rrs Performance',pow(10,y_measured_log[valid_PC_measured]),pow(10,y_true_log[valid_PC_measured]),metrics=[mdsa, sspb, slope, rmsle, msa]),'N=',len(y_measured_log[valid_PC_measured]))
					print('Printed Performance Metrics')
					
			if len(y_remote_OG) != 1:
				if len(valid.flatten()) != valid.sum():
					ax.scatter(y_true_log[~valid], [minv]*(~valid).sum(), color='r', alpha=0.4, label=r'$\mathbf{%s\ invalid, %s\ nan}$' % ((~valid).sum(), np.isnan(y_true_log[~valid]).sum()) ) 
					ax.legend(loc='lower right', prop={'weight':'bold', 'size': 16})

			add_identity(ax, ls='--', color='k', zorder=20)
			add_stats_box(ax, y_insitu.flatten()[valid], y_remote.flatten()[valid],metrics=[SpR, slope])

			plot_label ='ABCDEFGHIJKLMNOPQRST'
			text(0.965,0.02,plot_label[plt_idx],horizontalalignment='center',verticalalignment='bottom',transform=ax.transAxes,backgroundcolor='1.0',bbox=dict(facecolor='white',edgecolor='black',boxstyle='round'),fontdict=font)

			ax.tick_params(labelsize=12)
			ax.grid('on', alpha=0.3)
			if plt_idx == 1:
				ax.set_yticklabels([])

			filename = folder.joinpath(f'Summary_of_remote_vs_insitu_COMPARISON_{products}_{current_sensor}_{current_product}.jpg')
			plt.tight_layout()
			plt_idx = plt_idx+1

		plt.savefig(filename.as_posix(), dpi=600, bbox_inches='tight', pad_inches=0.1,)
		plt.close()

def plot_band_correlations(x_data, y_data, products = 'chl,PC' , run_name="",sensor='HICO',labels=None,PC_correlation_threshold = 0.7):
	import matplotlib.pyplot as plt
	from matplotlib.figure import figaspect

	folder = Path('scatter_plots')
	folder.mkdir(exist_ok=True, parents=True)
	n_col = 6

	list_of_legacy_algorthms = []

	for label_counter,label in enumerate(labels):
		if (len(label.split('|')) != 2) and (len(label.split('|')) != 3):
			list_of_legacy_algorthms.append(label)
	n_row = math.ceil(len(list_of_legacy_algorthms)/n_col)
	fig_size = 5

	products = products.split(',')
	product_labels = default_dd({
		'chl' : 'Chla',
		'PC'  : 'PC'
	})

	product_units = default_dd({
		'chl' : '[mg m^{-3}]',
		'PC' : '[mg m^{-3}]',

		'tss' : '[g m^{-3}]',
		'aph' : '[m^{-1}]',
	}, lambda k: '')
	spearmans_ranked_list_LH_threshold_PC_center_bandwidth = []
	spearmans_ranked_list_BR_threshold_PC_numerator_denominator = []

	for product_counter,product in enumerate(products):
		plabel_1 = f'{product_labels[products[product_counter]]}'
		plabel_2 = f'{product_units[products[product_counter]]}'
		xlabel_txt = f'ratio value'
		xlabel = fr'${{{xlabel_txt}}}$'
		ylabel = fr'${{{plabel_1} {plabel_2}}}$'
		numerator_wavelength_list = []
		denominator_wavelength_list = []
		spearmans_ranked_list = []
		counter = 0
		center_wavelength_list = [] 
		wavelength_difference_list = [] 
		LH_counter = 0
		spearmans_ranked_list_LH = []

		fig, axes = plt.subplots(n_row, n_col, figsize=(fig_size*n_col, fig_size*n_row))
		axes      = [ax for axs in np.atleast_1d(axes) for ax in np.atleast_1d(axs)]

		for label_counter,label in enumerate(labels):
			x_data_plotting_OG = x_data[:,label_counter]

			x_data_plotting = x_data[:,label_counter]
			x_data_plotting = x_data_plotting - min(x_data_plotting) + .1 

			for index,algorithm in enumerate(list_of_legacy_algorthms):
				if label == algorithm:
					ax = axes[index]
					ax.scatter(x_data_plotting,y_data[:,product_counter])
					ax.set_ylabel(ylabel.replace(' ', '\ '), fontsize=20, labelpad=10)
					ax.set_xlabel(xlabel.replace(' ', '\ '), fontsize=20, labelpad=10)

					ax.set_xticklabels([])
					ax.set_yscale('log')
					max_x = max(x_data_plotting)
					min_x = min(x_data_plotting)

					ax.set_xscale('log')

					ax.set_xticks([])
					ax.set_xticklabels([])
					ax.minorticks_off()
					title_label = fr'$\mathbf{{{label} }}$'
					ax.set_title(title_label,y=1,fontsize=20)
					add_stats_box(ax, x_data_plotting,y_data[:,product_counter], metrics=[ SpR])

			if len(label.split('|')) == 2 :
				num_denom = label.split('|')

				numerator_wavelength_list.append( int(num_denom[0]))
				denominator_wavelength_list.append(int(num_denom[1]))
				spearmans_ranked_score = SpR(x_data[:,label_counter],y_data[:,product_counter]) 

				if np.isfinite(spearmans_ranked_score):
					spearmans_ranked_list.append(spearmans_ranked_score)
				else:
					spearmans_ranked_list.append(0)

				if product == 'PC':
					if abs(spearmans_ranked_list[counter])>PC_correlation_threshold:
						if numerator_wavelength_list[counter]>denominator_wavelength_list[counter]:
							spearmans_ranked_list_BR_threshold_PC_numerator_denominator.append([numerator_wavelength_list[counter],denominator_wavelength_list[counter]])
				counter = counter+1

			if len(label.split('|')) == 3:
				low_center_high_wavelength = label.split('|')

				center_wavelength_list.append(int(low_center_high_wavelength[1]))
				wavelength_difference_list.append( int(low_center_high_wavelength[1]) - int(low_center_high_wavelength[0]))
				
				spearmans_ranked_score_LH = SpR(x_data[:,label_counter],y_data[:,product_counter]) 

				if np.isfinite(spearmans_ranked_score_LH):
					spearmans_ranked_list_LH.append(spearmans_ranked_score_LH)
				else:
					spearmans_ranked_list_LH.append(0)

				if product == 'PC':
					if abs(spearmans_ranked_list_LH[LH_counter])>PC_correlation_threshold:
						spearmans_ranked_list_LH_threshold_PC_center_bandwidth.append([center_wavelength_list[LH_counter],wavelength_difference_list[LH_counter]])
				LH_counter = LH_counter+1

		def compare_two_lists(test_list_1,test_list_2,test_list_3,comparative_value_1,comparative_value_2):
			index_1  =[True if test_list_value == comparative_value_1 else False for index,test_list_value in enumerate(test_list_1) ]
			index_2  =[True  if test_list_value == comparative_value_2 else False for index,test_list_value in enumerate(test_list_2)]

			overlapping_indices_bool = np.logical_and(index_1, index_2)
			overlapping_indices=[index for index,overlapping_index in enumerate(overlapping_indices_bool) if overlapping_index]
			if len(overlapping_indices) == 0:
				return comparative_value_1, comparative_value_2, np.nan 
			else: 
				if len(overlapping_indices) == 1:
					overlapping_index = overlapping_indices[0] 
				else:
					assert(0),f'Non-unique set defined'

			return test_list_1[overlapping_index], test_list_2[overlapping_index], test_list_3[overlapping_index]


		center_wavelength_list_nans = []
		wavelength_difference_list_nans = []
		spearmans_ranked_list_LH_nans = []

		for current_center_wavelength in sorted(set(center_wavelength_list)):
			for wavelength_difference_current in  sorted(set(wavelength_difference_list)):
				center_wavelength_comp,wavelength_diff_comp, spearmans_comp = compare_two_lists(test_list_1=center_wavelength_list,test_list_2=wavelength_difference_list,test_list_3=spearmans_ranked_list_LH,comparative_value_1=current_center_wavelength,comparative_value_2=wavelength_difference_current)		
				
				center_wavelength_list_nans.append(center_wavelength_comp)
				wavelength_difference_list_nans.append(wavelength_diff_comp)
				spearmans_ranked_list_LH_nans.append(spearmans_comp)
		
		center_wavelength_list = center_wavelength_list_nans
		wavelength_difference_list = wavelength_difference_list_nans
		spearmans_ranked_list_LH = spearmans_ranked_list_LH_nans

		filename = folder.joinpath(f'Legacy_correlations{run_name}_{products}_{sensor}_{product}.jpg')
		plt.savefig(filename.as_posix(), dpi=600, bbox_inches='tight', pad_inches=0.1,)
		plt.close()

		W, H = figaspect(0.4)
		fig = plt.figure(figsize=(W,H))
		ax = fig.add_subplot(121)

		numerator_wavelengths_array=np.reshape(np.asarray(numerator_wavelength_list),(int(np.sqrt(len(numerator_wavelength_list))),int(np.sqrt(len(numerator_wavelength_list)))))
		denominator_wavelengths_array=np.reshape(np.asarray(denominator_wavelength_list),(int(np.sqrt(len(numerator_wavelength_list))),int(np.sqrt(len(numerator_wavelength_list)))))
		spearmans_ranked_array=np.reshape(np.asarray(spearmans_ranked_list),(int(np.sqrt(len(numerator_wavelength_list))),int(np.sqrt(len(numerator_wavelength_list)))))

		from matplotlib import cm

		surf =  ax.imshow(np.abs(spearmans_ranked_array),extent=(0,len(numerator_wavelengths_array),0,len(denominator_wavelengths_array)),aspect='auto',interpolation='none',vmin=0, vmax = 1,cmap=cm.coolwarm,origin='lower')
		label_locations = (np.asarray(np.arange(0,len(numerator_wavelengths_array),np.round(len(numerator_wavelengths_array)/13)))).astype(int)

		ax.set_xticks(label_locations)

		ax.set_xticklabels(np.array(sorted(set(numerator_wavelength_list)))[label_locations])

		ax.set_yticks(label_locations)

		ax.set_yticklabels(np.array(sorted(set(numerator_wavelength_list)))[label_locations])

		fig.colorbar(surf,shrink=0.7,aspect=7,label=fr'$|PRCC|$')
		ax.set_title('Band Ratio Correlations: {}'.format(product))
		filename = folder.joinpath(f'Band_correlations_{run_name}_{products}_{sensor}_{product}.jpg')
		ax.set_ylabel('Numerator (nm)')
		ax.set_xlabel('Denominator (nm)')

		ax = fig.add_subplot(122) 
		center_wavelengths_array=np.reshape(np.asarray(center_wavelength_list),(-1,int((len(set(wavelength_difference_list))))))
		wavelengths_difference_array=np.reshape(np.asarray(wavelength_difference_list),(-1,int((len(set(wavelength_difference_list))))))
		spearmans_ranked_LH_array=np.reshape(np.asarray(spearmans_ranked_list_LH),(-1,int((len(set(wavelength_difference_list))))))

		surf = ax.imshow(np.abs(spearmans_ranked_LH_array),extent=(0,len(wavelengths_difference_array[0]),0,len(center_wavelengths_array)),aspect='auto',interpolation='none',vmin=0, vmax = 1,cmap=cm.coolwarm,origin='lower') 
		label_locations = (np.asarray(np.arange(0,len(center_wavelengths_array),np.round(len(center_wavelengths_array)/13.8)))).astype(int) 

		ax.set_yticks(label_locations)
		ax.set_yticklabels(np.array(sorted(set(center_wavelength_list)))[label_locations])

		label_locations = (np.asarray(np.arange(0,len(wavelengths_difference_array[0]),1))).astype(int)

		ax.set_xticks(label_locations)
		ax.set_xticklabels(np.array(sorted(set(wavelengths_difference_array[0])))[label_locations])

		fig.colorbar(surf,shrink=0.7,aspect=7,label=fr'$|PRCC|$')
		ax.set_title('Line Height Correlations: {}'.format(product))
		ax.set_ylabel('Center Wavelength (nm)')
		ax.set_xlabel('Baseline Wavelength +/- Offset (nm)')

		plt.tight_layout()
		plt.savefig(filename.as_posix(),dpi=600, bbox_inches='tight', pad_inches=0.1,)
		plt.close()
	return spearmans_ranked_list_BR_threshold_PC_numerator_denominator,spearmans_ranked_list_LH_threshold_PC_center_bandwidth;

@ignore_warnings
def plot_scatter_summary(y_test, benchmarks, products = 'chl,PC' , run_name="",sensor='HICO'):
	import matplotlib.patheffects as pe
	import matplotlib.ticker as ticker
	import matplotlib.pyplot as plt
	import seaborn as sns
	folder = Path('scatter_plots')
	folder.mkdir(exist_ok=True, parents=True)
	benchmarks_OG = benchmarks
	y_test_OG = y_test
	products = products.split(',')
	product_labels = default_dd({
		'chl' : 'Chl\\textit{a}',
		'PC'  : 'PC'
	})
	product_units = default_dd({
		'chl' : '[mg m^{-3}]',
		'PC' : '[mg m^{-3}]',

		'tss' : '[g m^{-3}]',
		'aph' : '[m^{-1}]',
	}, lambda k: '')
	model_labels = default_dd({
		'MDN' : 'MDN_{A}',
	})

	plt.rc('text', usetex=True)
	plt.rcParams['mathtext.default']='regular'

	plot_order_OG = run_name

	for product_index, product in enumerate(products):

		plt_idx = 0
		fig_size   = 5
		n_col      = 3
		n_row      = math.ceil(float(len(benchmarks))/float(n_col))

		fig, axes = plt.subplots(n_row, n_col, figsize=(fig_size*n_col, fig_size*n_row))
		axes      = [ax for axs in np.atleast_1d(axes) for ax in np.atleast_1d(axs)]
		colors    = ['xkcd:sky blue', 'xkcd:tangerine', 'xkcd:fresh green', 'xkcd:greyish blue', 'xkcd:goldenrod',  'xkcd:clay', 'xkcd:bluish purple', 'xkcd:reddish', 'xkcd:neon purple']
		print(f'Plot size: {n_row} x {n_col}')
		for estimates_index, estimates in enumerate(benchmarks):
			print("Estimates Index:", estimates_index)

			y_true = y_test[estimates_index]
			y_true = y_true[:,product_index]

			plt_idx = plt_idx+1

			curr_idx = estimates_index
			full_ax  = fig.add_subplot(111, frameon=False)
			full_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False, pad=10)

			plabel = f'{product_labels[product]} {product_units[product]}'
			xlabel = fr'$\mathbf{{Measured {plabel}}}$'
			ylabel = fr'$\mathbf{{Modeled {plabel}}}$'
			full_ax.set_xlabel(xlabel.replace(' ', '\ '), fontsize=20, labelpad=10)
			full_ax.set_ylabel(ylabel.replace(' ', '\ '), fontsize=20, labelpad=10)

			s_lbl = get_sensor_label(sensor).replace('-',' ')
			n_pts = len(y_test)
			title = fr'Summary of MDN Performance: {product}'
			full_ax.set_title(title.replace(' ', '\ '), fontsize=24, y=1.04)
			
			y_est = estimates[..., product_index]

			ax    = axes[curr_idx]
			cidx  = curr_idx
			color = colors[cidx]

			first_row = curr_idx < n_col 
			last_row  = curr_idx >= ((n_row-1)*n_col)
			first_col = (curr_idx % n_col) == 0
			last_col  = ((curr_idx+1) % n_col) == 0

			y_est_log  = np.log10(y_est).flatten()
			y_true_log = np.log10(y_true).flatten()

			l_kws = {'color': color, 'path_effects': [pe.Stroke(linewidth=4, foreground='k'), pe.Normal()], 'zorder': 22, 'lw': 1}
			s_kws = {'alpha': 0.4, 'color': color}
			est_lbl = run_name[estimates_index]
			if est_lbl == 'MDN':
				[i.set_linewidth(5) for i in ax.spines.values()]
				est_lbl = 'MDN_{A}'
				est_lbl = 'MDN-I'
			else:
				est_lbl = est_lbl.replace('Gons_2band', 'Gons').replace('Gilerson_2band', 'GI2B').replace('Smith_','').replace('_','-') 

			if product not in ['chl', 'tss','PC'] and last_col:
				ax2 = ax.twinx()
				ax2.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False, pad=0)
				ax2.grid(False)
				ax2.set_yticklabels([])
				ax2.set_ylabel(fr'$\mathbf{{{bands[plt_idx]:.0f}nm}}$', fontsize=20)

			minv = int(np.nanmin(y_true_log)) - 1 if product != 'aph' else -4
			maxv = int(np.nanmax(y_true_log)) + 1 if product != 'aph' else 1
			loc  = ticker.LinearLocator(numticks=maxv-minv+1)
			fmt  = ticker.FuncFormatter(lambda i, _: r'$10$\textsuperscript{%i}'%i)

			ax.set_ylim((minv, maxv))
			ax.set_xlim((minv, maxv))
			ax.xaxis.set_major_locator(loc)
			ax.yaxis.set_major_locator(loc)
			ax.xaxis.set_major_formatter(fmt)
			ax.yaxis.set_major_formatter(fmt)

			if not last_row:  ax.set_xticklabels([])
			if not first_col: ax.set_yticklabels([])

			valid = np.logical_and(np.isfinite(y_true_log), np.isfinite(y_est_log))

			if valid.sum():
				sns.regplot(y_true_log[valid], y_est_log[valid], ax=ax, scatter_kws=s_kws, line_kws=l_kws, fit_reg=True, truncate=False, robust=True, ci=None)
				kde = sns.kdeplot(y_true_log[valid], y_est_log[valid], shade=False, ax=ax, bw='scott', n_levels=4, legend=False, gridsize=100, color=color)
				kde.collections[2].set_alpha(0)

			if len(valid.flatten()) != valid.sum():
				ax.scatter(y_true_log[~valid], [minv]*(~valid).sum(), color='r', alpha=0.4, label=r'$\mathbf{%s\ invalid, %s\ nan}$' % ((~valid).sum(), np.isnan(y_true_log[~valid]).sum()) ) 
				ax.legend(loc='lower right', prop={'weight':'bold', 'size': 16})

			add_identity(ax, ls='--', color='k', zorder=20)
			add_stats_box(ax, y_true, y_est)

			ax.set_title(fr'$\mathbf{{\large{{{est_lbl}}}}}$', fontsize=18)

			ax.tick_params(labelsize=18)
			ax.grid('on', alpha=0.3)

		filename = folder.joinpath(f'Summary_of_MDN_Performance_{products}_{sensor}_{product}_test.jpg')
		plt.tight_layout()
		plt.savefig(filename.as_posix(), dpi=600, bbox_inches='tight', pad_inches=0.1,)
		plt.close()


def plot_bar_chart(input_values):
	import matplotlib.pyplot as plt
	folder = Path('scatter_plots')
	folder.mkdir(exist_ok=True, parents=True)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.grid(color='black',alpha=0.1)
	ax.set_axisbelow(True)
	
	sorted_unique_values = sorted(set(input_values))
	list_of_OWT_counts = []
	for value in sorted_unique_values:
		list_of_OWT_counts.append(sum(input_values == value))
	colors = ['xkcd:royal blue', 'xkcd:cornflower blue',  'xkcd:azure', 'xkcd:shamrock green', 'xkcd:kermit green', 'xkcd:drab', 'xkcd:hazel', ]
	ax.bar(sorted_unique_values,list_of_OWT_counts,color=colors,edgecolor='black')

	ax.set_ylabel('Frequency',fontsize=15)
	ax.tick_params(axis='both',which='minor',labelsize=12)
	ax.tick_params(axis='both',which='major',labelsize=12)
	ax.set_xticklabels(['0','OWT-1','OWT-2','OWT-3','OWT-4','OWT-5','OWT-6','OWT-7'],rotation=45,ha="right")

	filename = folder.joinpath(f'Optical_Water_Types_sum_{sum(list_of_OWT_counts)}.jpg')
	plt.tight_layout()

	plt.savefig(filename.as_posix(), dpi=600, bbox_inches='tight', pad_inches=0.1,)

	plt.close()

def plot_histogram(product_values,products):
	PC_loc = products.index("PC")
	chl_loc = products.index("chl")

	import matplotlib.pyplot as plt
	from matplotlib.figure import figaspect
	import scipy.stats as stats
	folder = Path('scatter_plots')
	folder.mkdir(exist_ok=True, parents=True)
	W, H = figaspect(0.4)
	fig = plt.figure(figsize=(W, H))
	ax = fig.add_subplot(131)
	ax.grid(color='black',alpha=0.1)
	ax.set_axisbelow(True)
	colors = ['aqua', 'orangered',  'xkcd:tangerine', 'xkcd:fresh green', 'xkcd:clay', 'magenta', 'xkcd:sky blue', ]
	product_labels = default_dd({
		'chl' : 'Chl\\textit{a}',
		'PC'  : 'PC',
		'chl/pc': 'Chl\\textit{a}:PC',
		'pc/chl': 'PC:Chl\\textit{a}',

	})
	product_units = default_dd({
		'chl' : '[mg m^{-3}]',
		'PC' : '[mg m^{-3}]',

		'tss' : '[g m^{-3}]',
		'aph' : '[m^{-1}]',
	}, lambda k: '')
	plt_idx = 0
	plabel = f'{product_labels[products[plt_idx]]} {product_units[products[plt_idx]]}'
	xlabel = fr'$\mathbf{{{plabel}}}$'

	bin_locations = np.linspace(-1,3)
	x_vals = np.log10(product_values[:,chl_loc])
	gauss_density = stats.gaussian_kde(x_vals)
	n,bins,patches = ax.hist(x_vals,bins=bin_locations,facecolor='xkcd:dark mint green',edgecolor='white',linewidth=0.5,density=False,log=False,alpha=0.75)
	ax.set_xlabel(xlabel.replace(' ', '\ '),fontsize=15)
	ylabel = fr'$\mathbf{{Frequency}}$'

	ax.set_ylabel(ylabel.replace(' ', '\ '),fontsize=15)
	ax.tick_params(axis='both',which='minor',labelsize=13)
	ax.tick_params(axis='both',which='major',labelsize=13)
	ax.set_facecolor('xkcd:white')

	labels = ax.get_xticklabels(which='both')
	locs = ax.get_xticks()

	xtick_labels = [int(value) for value in locs] 
	xtick_labels = [fr'${{10^{ {value} }}}$' for value in xtick_labels]

	ax.set_xticks(locs)
	ax.set_xticklabels(xtick_labels)
	ax.set_xlim((-1,3))

	plt_idx = 1
	plabel = f'{product_labels[products[PC_loc]]} {product_units[products[PC_loc]]}'
	xlabel = fr'$\mathbf{{{plabel}}}$'

	ax = fig.add_subplot(132)
	ax.grid(color='black',alpha=0.1)
	ax.set_axisbelow(True)
	x_vals = np.log10(product_values[:,PC_loc])
	gauss_density = stats.gaussian_kde(x_vals)
	n,bins,patches  = ax.hist(x_vals,bins=bin_locations,facecolor='xkcd:cool blue',edgecolor='white',linewidth=0.5,density=False,log=False,alpha=0.75)

	ax.set_xlabel(xlabel.replace(' ', '\ '),fontsize=15)
	ax.tick_params(axis='both',which='minor',labelsize=13)
	ax.tick_params(axis='both',which='major',labelsize=13)
	ax.set_facecolor('xkcd:white')

	labels = ax.get_xticklabels(which='both')
	locs = ax.get_xticks()

	xtick_labels = [int(value) for value in locs] 
	xtick_labels = [fr'${{10^{ {value} }}}$' for value in xtick_labels]
	ax.set_xticks(locs)
	ax.set_xticklabels(xtick_labels)
	ax.set_xlim((-1,3))

	plt_idx = 2
	plabel = f'{product_labels[products[PC_loc]]}:{product_labels[products[chl_loc]]}'
	xlabel = fr'$\mathbf{{{plabel}}}$'

	ax = fig.add_subplot(133)
	ax.grid(color='black',alpha=0.1)
	ax.set_axisbelow(True)
	x_vals = np.log10(product_values[:,PC_loc]/product_values[:,chl_loc])
	bin_locations = np.linspace(-3,1)

	gauss_density = stats.gaussian_kde(x_vals)
	n,bins,patches  = ax.hist(x_vals,bins=bin_locations,facecolor='xkcd:burnt orange',edgecolor='white',linewidth=0.5,density=False,log=False,alpha=0.75)
	ax.set_xlabel(xlabel.replace(' ', '\ '),fontsize=15)
	ax.tick_params(axis='both',which='minor',labelsize=13)
	ax.tick_params(axis='both',which='major',labelsize=13)
	ax.set_facecolor('xkcd:White')

	labels = ax.get_xticklabels(which='both')
	locs = ax.get_xticks()

	xtick_labels = [int(value) for value in locs] 
	xtick_labels = [fr'${{10^{ {value} }}}$' for value in xtick_labels]
	ax.set_xticks(locs)
	ax.set_xticklabels(xtick_labels)
	ax.set_xlim((-3,1))

	filename = folder.joinpath(f'Product_histogram_{np.shape(product_values)}.jpg')
	plt.tight_layout()
	plt.savefig(filename.as_posix(), dpi=600, bbox_inches='tight', pad_inches=0.1,)
	plt.close()

	print('Mean of {}: {} Median of {}:'.format(products[0],np.mean(product_values[:,chl_loc]),np.median(product_values[:,chl_loc])))
	print('Mean of {}: {} Median of {}:'.format(products[1],np.mean(product_values[:,PC_loc]),np.median(product_values[:,PC_loc])))
	print('Mean of {}: {} Median of {}:'.format(products[1],np.mean(product_values[:,PC_loc]/product_values[:,chl_loc]),np.median(product_values[:,PC_loc]/product_values[:,chl_loc])))

def plot_Rrs_scatter(Rrs,Rrs_wvl,OWT,y_data):
	import matplotlib.pyplot as plt
	from scipy.interpolate import interp1d

	folder = Path('scatter_plots')
	folder.mkdir(exist_ok=True, parents=True)
	from matplotlib.figure import figaspect
	W, H = figaspect(0.4)
	fig = plt.figure(figsize=(W, H))

	ax = fig.add_subplot(121)
	ax.grid(color='black',alpha=0.1)
	ax.set_axisbelow(True)
	ax.set_ylim((0,4.15*10**-2))
	sorted_unique_values = sorted(set(OWT))
	list_of_OWT_counts = []

	min_wavelength = 390
	max_wavelength = 770
	
	interpolation_range = range(min_wavelength+10,max_wavelength-10,4)
	list_of_standardized_Rrs = []

	for Rrs_counter,current_Rrs in enumerate(Rrs):
		wavelengths = Rrs_wvl[Rrs_counter]

		z_w_data_bool_array = [True if wavelength<max_wavelength and wavelength>min_wavelength else False for wavelength in wavelengths ]

		z_data=current_Rrs[z_w_data_bool_array]
		z_w_data=wavelengths[z_w_data_bool_array]

		interp_function = interp1d(z_w_data,z_data,kind='cubic')
		list_of_standardized_Rrs.append(interp_function(interpolation_range))

	sorted_unique_values = sorted(set(OWT))
	list_of_mean_Rrs_values = []
	list_of_mean_Chl = [] 
	list_of_mean_PC = []
	list_of_median_Chl = []
	list_of_median_PC = []
	array_of_standardized_Rrs = np.asarray(list_of_standardized_Rrs)

	for value in sorted_unique_values:
		available_data = y_data[OWT == value]
		array_of_chl = np.asarray(available_data[:,0])
		array_of_PC =  np.asarray(available_data[:,1])
		
		mean_Rrs_current = np.mean(array_of_standardized_Rrs[OWT == value],axis=0)
		
		list_of_mean_Rrs_values.append(mean_Rrs_current)
		list_of_mean_Chl.append(np.mean(array_of_chl))
		list_of_mean_PC.append(np.mean(array_of_PC))

		list_of_median_Chl.append(np.median(array_of_chl))
		list_of_median_PC.append(np.median(array_of_PC))

	colors = ['xkcd:royal blue', 'xkcd:cornflower blue',  'xkcd:azure', 'xkcd:shamrock green', 'xkcd:kermit green', 'xkcd:drab', 'xkcd:hazel', ]
	marker_list =["^","8","s","<","*","D","v",] 
	for i,mean_Rrs in enumerate(list_of_mean_Rrs_values):
		formatted_label = "OWT-{}, chl {}, PC {}, PC:Chl {}".format(sorted_unique_values[i],convert_float_string_justify(list_of_mean_Chl[i]),convert_float_string_justify(list_of_mean_PC[i]),list_of_mean_PC[i]/list_of_mean_Chl[i])
		formatted_label_median = "OWT-{}, chl {}, PC {},PC:Chl {}".format(sorted_unique_values[i],convert_float_string_justify(list_of_median_Chl[i]),convert_float_string_justify(list_of_median_PC[i]),list_of_median_PC[i]/list_of_median_Chl[i])

		formatted_label = str(formatted_label)
		formatted_label = "OWT-{}".format(sorted_unique_values[i])

		ax.scatter(interpolation_range,mean_Rrs,color=colors[i],label=formatted_label,marker=marker_list[i])
		ax.plot(interpolation_range,mean_Rrs,color=colors[i])
	value = -1
	rs='rs'
	ylabel = fr'$\mathbf{{ R_{{rs}} \  [sr^{ {value}}]}}$'
	ax.set_ylabel(ylabel,fontsize=15)
	xlabel = fr'$\mathbf{{Wavelength \ [nm]}}$'

	ax.set_xlabel(xlabel,fontsize=15)

	ax.tick_params(axis='both',which='minor',labelsize=12)
	ax.tick_params(axis='both',which='major',labelsize=12)

	ax.set_xlim((400,750))
	plt.ticklabel_format(axis="y",style="sci",scilimits=(0,0))
	plt.rcParams["font.weight"] = "bold"

	filename = folder.joinpath(f'Optical_Water_Types_mean_Rrs_{sum(list_of_OWT_counts)}.jpg')
	plt.tight_layout()
	plt.legend()


	ax = fig.add_subplot(122)
	ax.grid(color='black',alpha=0.1)
	ax.set_axisbelow(True)

	sorted_unique_values = sorted(set(OWT))

	list_of_OWT_counts = []
	for value in sorted_unique_values:
		list_of_OWT_counts.append(sum(OWT == value))

	colors = ['xkcd:royal blue', 'xkcd:cornflower blue',  'xkcd:azure', 'xkcd:shamrock green', 'xkcd:kermit green', 'xkcd:drab', 'xkcd:hazel', ]
	ax.bar(sorted_unique_values,list_of_OWT_counts,color=colors,edgecolor='black')

	ax.set_ylabel('Frequency',fontsize=15)
	ax.tick_params(axis='both',which='minor',labelsize=12)
	ax.tick_params(axis='both',which='major',labelsize=12)
	ax.set_xticklabels(['0','OWT-1','OWT-2','OWT-3','OWT-4','OWT-5','OWT-6','OWT-7'],rotation=45,ha="right")

	filename = folder.joinpath(f'Optical_Water_Types_sum_{sum(list_of_OWT_counts)}_and_Rrs.jpg')
	plt.tight_layout()

	plt.rcParams["font.weight"] = "bold"

	plt.savefig(filename.as_posix(),dpi=600, bbox_inches='tight', pad_inches=0.1,)

	plt.close()
	
def convert_float_string_justify(value,total_width=5,precision=1):
	rounded_val = round(value,precision)
	rounded_val_string = str(rounded_val)
	rounded_val_string_length = len(rounded_val_string)
	left_pad_spaces_amount = total_width - rounded_val_string_length
	left_padded_string = rounded_val_string.rjust(total_width, " ")

	return left_padded_string
