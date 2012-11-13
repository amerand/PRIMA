import astromNew
from matplotlib import pyplot
import numpy as np
import dpfit
import time
import os

# pupil correction measured in January 2012, overestimated probably
# for AT4 compared to November 2011...

ATsPUP =  {'AT4 x0':-0.484,
           'AT4 y0':-0.365,
           'AT4 a0':10.871,
           'AT4 phi0':164.727,
           'AT4 a1':0.477,
           'AT4 phi1':58.716,
           'AT4 phi2':-68.764,
           'AT3 x0':0.568,
           'AT3 y0':-0.406,
           'AT3 a0':2.129,
           'AT3 phi0':0.957,
           'AT3 a1':0.863,
           'AT3 phi1':-3.755,
           'AT3 phi2':-54.438,
           'ATPUP scale': 1.0}

# guess data directory

data_directory = '/Volumes/DATA500/PRIMA/COMM17/' # external BIG hardrive
if not os.path.isdir(data_directory):
    data_directory = '/Volumes/DATA/PRIMA/COMM17/' # external small hardrive
if not os.path.isdir(data_directory):
    data_directory = '/Users/amerand/DATA/PRIMA/COMM17/' # internal hardrive
if not os.path.isdir(data_directory):
    print 'WARNING: NO DATA DIRECTORY AVAILABLE!'

# following files were taken simultaneously with NACO observations.
# 0006 and 0010 are bad but were taken in the serie

files_hd10360_NACO = [
    '2011-11-20/PACMAN_OBJ_ASTRO_325_0001.fits',
    '2011-11-20/PACMAN_OBJ_ASTRO_325_0002.fits',
    '2011-11-20/PACMAN_OBJ_ASTRO_325_0003.fits',
    '2011-11-20/PACMAN_OBJ_ASTRO_325_0004.fits',
    '2011-11-20/PACMAN_OBJ_ASTRO_325_0005.fits',
    '2011-11-20/PACMAN_OBJ_ASTRO_325_0007.fits',
    '2011-11-20/PACMAN_OBJ_ASTRO_325_0008.fits',
    '2011-11-20/PACMAN_OBJ_ASTRO_325_0009.fits']

# Neil Zimmermann report:
naco = {'Delta RA cos(DEC)':-1.5519,
        'Delta DEC':-11.3351}

# first guess

fgNACO = {'SEP':8, 'PA':180, 'M0':0.00}
for k in ATsPUP.keys():
    fgNACO[k] = ATsPUP[k]
    
def fitHD10360(ATPUP_scale=1.0):
    """
    simple fit. ATPUP_scale=0.0 sets the pupil correction to 0
    """
    fgNACO['ATPUP scale'] = ATPUP_scale
    a = astromNew.fitListOfFiles(files_hd10360_NACO,
                                 directory=data_directory,
                                 firstGuess=fgNACO, plot=True,
                                 doNotFit=['AT3', 'AT4', 'ATPUP scale'])
    a['Delta DEC'] = a['BEST']['SEP']*np.cos(a['BEST']['PA']*np.pi/180)
    a['Delta RA cos(DEC)'] = a['BEST']['SEP']*np.sin(a['BEST']['PA']*np.pi/180)
    a['Delta RA'] = a['Delta RA cos(DEC)']/np.cos(a['DEC']['HD10360']*np.pi/180)
    return a

def bootstrapHD10360(N=100, ATPUP_scale=1.0, plot=False):
    """
    bootstraping. ATPUP_scale=0.0 sets the pupil correction to 0. N is
    the number of bootstrapping iteration.
    """
    t0 = time.time()
    fgNACO['ATPUP scale'] = ATPUP_scale
    b = astromNew.bootstrapListOfFiles(files_hd10360_NACO,
                                       directory=data_directory,
                                       firstGuess=fgNACO,
                                       doNotFit=['AT3', 'AT4', 'ATPUP scale'],
                                       plot=plot, N=N)
    print 'done in', round(time.time()-t0,2), \
          's (', round((time.time()-t0)/float(N), 2),\
          's per iteration)'
    if plot:
        t = np.linspace(0, 2*np.pi, 100)
        pyplot.plot(naco['Delta RA cos(DEC)'], naco['Delta DEC'],
                    '*r', label='NACO position')
        pyplot.plot(naco['Delta RA cos(DEC)']+np.cos(t)*13e-3,
                    naco['Delta DEC']+np.sin(t)*13e-3, 'r',
                    label='13mas circle')
        pyplot.legend(numpoints=1)
    return b

def plotBoot():
    scales = [0, 0.25, 0.5, 0.75, 1.0]
    c = ['w', 'r', 'g', 'b', 'k']
    pyplot.figure(10)
    pyplot.clf()
    pyplot.axes().set_aspect('equal', 'datalim')
    for k,s in enumerate(scales):
        tmp = bootstrapHD10360(N=30, ATPUP_scale=s)
        pyplot.plot(tmp['Delta RA cos(DEC)'],
                    tmp['Delta DEC'], 'o', color=c[k],
                    label='ATPUP scale=%4.2f'%s, alpha=0.5)
    t = np.linspace(0, 2*np.pi, 100)
    pyplot.plot(naco['Delta RA cos(DEC)'], naco['Delta DEC'],
                '*y', label='NACO position', markersize=10)
    pyplot.plot(naco['Delta RA cos(DEC)']+np.cos(t)*13e-3,
                naco['Delta DEC']+np.sin(t)*13e-3, 'y',
                label='13mas circle', linewidth=3, alpha=0.6)
    pyplot.legend(numpoints=1, loc='lower right')
    pyplot.ylabel(r'$\Delta$ dec [arcsec]')
    pyplot.xlabel(r'$\Delta$ RA $\cos$(dec) [arcsec]')
    return
