# MDN-PC
PC retrieval MDN for HICO and PRISMA

Setup:
1. conda create --name example_PC_environment python=3.7
2. conda activate example_PC_environment
3. pip install -r requirements.txt 
4. copy below code into main.py, 1 directory above current directory
5. python main.py


#################################### Main.py ####################################################
from MDN import image_estimates, get_tile_data
from MDN.parameters import get_args
from MDN.utils import get_sensor_bands, set_kwargs_PC
import matplotlib.pyplot as plt
import numpy as np

sensor = "HICO-noBnoNIR" #PRISMA-noBnoNIR
args = get_args(set_kwargs_PC(sensor))

# Tile should be the output of an atmospheric correction program e.g. SeaDAS
file_name = {
    'HICO-noBnoNIR' :"/media/ryanoshea/BackUp/Scenes/Erie/HICO/20140908/unedited/H2014251184102.L1B_ISS/out/l2gen.nc",
    'PRISMA-noBnoNIR' : "/media/ryanoshea/BackUp/Scenes/Trasimeno/PRISMA/20200725/unedited/out/ATCOR.bsq",
}

bands, Rrs = get_tile_data(file_name[sensor], sensor, allow_neg=False)
# Rrs = np.dstack([np.random.rand(3,3) for band in get_sensor_bands(sensor)])
output, idx = image_estimates(Rrs,args=args, sensor=sensor)
PC = output[idx['PC']][0]
#chl = output[idx['chl']][0]

print(PC)
print(np.shape(PC))

plt.imshow(np.flipud(np.transpose(PC)), vmin=0, vmax=100, cmap='jet')
plt.show()


