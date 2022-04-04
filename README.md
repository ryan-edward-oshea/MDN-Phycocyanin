# MDN-PC
PC retrieval MDN for HICO and PRISMA

Setup:
1. Download: git clone https://github.com/STREAM-RS/MDN-Phycocyanin.git MDNPC
2. conda create --name example_PC_environment python=3.7
3. conda activate example_PC_environment
4. pip install -r requirements.txt 
5. copy below code into main.py, 1 directory above current directory
6. python main.py


#################################### Main.py ####################################################
from MDNPC import image_estimates, get_tile_data
from MDNPC.parameters import get_args
from MDNPC.utils import get_sensor_bands, set_kwargs_PC, load_geotiff_bands
import matplotlib.pyplot as plt
import numpy as np

sensor = "HICO-noBnoNIR" #PRISMA-noBnoNIR
args = get_args(set_kwargs_PC(sensor))

# Tile should be the output of an atmospheric correction program e.g. SeaDAS
file_name = {
    'HICO-noBnoNIR' :"/media/ryanoshea/BackUp/Scenes/Erie/HICO/20140908/unedited/H2014251184102.L1B_ISS/out/l2gen.nc",
    'PRISMA-noBnoNIR' : "/media/ryanoshea/BackUp/Scenes/Trasimeno/PRISMA/20200725/unedited/out/ATCOR.bsq",
}

#loads from netcdf (e.g., l2gen)
bands, Rrs = get_tile_data(file_name[sensor], sensor, allow_neg=False)

##loads from geotiff:
#wavelengths, Rrs = load_geotiff_bands(sensor='HICO-noBnoNIR',path_to_tile="ACOLITE.tif",allow_neg=False,atmospheric_correction="ACOLITE")
#wavelengths, Rrs = load_geotiff_bands(sensor='PRISMA-noBnoNIR',path_to_tile="asi_mdn_bands.tif",allow_neg=False,atmospheric_correction="asi")
#Atmospheric_correction algorithms can be: "ACOLITE", "POLYMER", or "iCOR"
#Automatically corrects from rhow to Rrs by dividing by pi for POLYMER or iCOR imagery, assumes accolite imagery is Rrs, assumes asi is in correct order, with correct wavelengths, in Rrs
#Rw to Rrs Divisor can be overwritten with OVERRIDE_DIVISOR argument

##loads from csv, assuming Rrs_wvl.csv and Rrs.csv are in wavelength ascending order, adds input dimension.
#import pandas as pd
#Rrs_csv = pd.read_csv('Rrs.csv')
#Rrs_csv = np.expand_dims(np.asarray(Rrs_csv),0) #Columns are wavelengths, we add a dimension, to make he input csv into an 'image' format, for image_estimates below.
#wavelengths  = pd.read_csv('Rrs_wvl.csv',headers=None) #1st column contains wavelengths


# Rrs = np.dstack([np.random.rand(3,3) for band in get_sensor_bands(sensor)])
output, idx = image_estimates(Rrs,args=args, sensor=sensor)
PC = output[idx['PC']][0]
#chl = output[idx['chl']][0]

print(PC)
print(np.shape(PC))

plt.imshow(np.flipud(np.transpose(PC)), vmin=0, vmax=100, cmap='jet')
plt.show()


