import os
import numpy as np
from scipy.optimize import leastsq
import pyfits
import time
from matplotlib import pyplot

import prima # low level data reduction
import astromNew # astrometric data analysis
import myfit
import os
import dpfit

"""
here the astrometry dataset that we agreed on using for the separation
fitting / residual analysis.

"""

#data_directory = '/Volumes/DATA/PRIMA/COMM18/' # external hardrive
data_directory = '/Users/amerand/DATA/PRIMA/COMM18/' # internal hardrive
# -----------------------------------------------------------------------

files_hd18622 = ['2012-01-14/PACMAN_OBJ_ASTRO_015_0003.fits',
                 '2012-01-14/PACMAN_OBJ_ASTRO_015_0004.fits',
                 '2012-01-14/PACMAN_OBJ_ASTRO_015_0005.fits',
                 '2012-01-14/PACMAN_OBJ_ASTRO_015_0006.fits',
                 '2012-01-14/PACMAN_OBJ_ASTRO_015_0007.fits',
                 '2012-01-16/PACMAN_OBJ_ASTRO_017_0001.fits',
                 '2012-01-16/PACMAN_OBJ_ASTRO_017_0002.fits',
                 '2012-01-16/PACMAN_OBJ_ASTRO_017_0003.fits',
                 '2012-01-16/PACMAN_OBJ_ASTRO_017_0004.fits',
                 '2012-01-18/PACMAN_OBJ_ASTRO_019_0001.fits',
                 '2012-01-18/PACMAN_OBJ_ASTRO_019_0002.fits',
                 '2012-01-18/PACMAN_OBJ_ASTRO_019_0003.fits',
                 '2012-01-18/PACMAN_OBJ_ASTRO_019_0004.fits']
"""
astromNew.fitListOfFiles(comm18.files_hd18622, directory=comm18.data_directory,
                         maxResiduals=3,
                         firstGuess={'SEP':8.3, 'PA':91.3,
                                     'M0 MJD 55941.00 55941.99':-0.017,
                                     'M0 MJD 55943.00 55943.99':-0.022,
                                     'M0 MJD 55945.00 55945.99':-0.022 })
"""

"""
astromNew.fitListOfFiles(files_hd18622_014, directory=data_directory,
                         maxResiduals=3,
                         firstGuess={'SEP':8.3, 'PA':91.3,
                                     'M0 MJD 55941.00 55941.99':-0.017,
                                     'M0 MJD 55943.00 55943.99':-0.022},
                         fitOnly=['SEP','PA','M0 MJD 55941.00 55941.99','M0 MJD 55943.00 55943.99'])
"""
files_hd18622_019 = ['2012-01-18/PACMAN_OBJ_ASTRO_019_0001.fits',
                     '2012-01-18/PACMAN_OBJ_ASTRO_019_0002.fits',
                     '2012-01-18/PACMAN_OBJ_ASTRO_019_0003.fits',
                     '2012-01-18/PACMAN_OBJ_ASTRO_019_0004.fits']


files_hd65297_019 = [#'2012-01-18/PACMAN_OBJ_ASTRO_019_0005.fits',
                     #'2012-01-18/PACMAN_OBJ_ASTRO_019_0006.fits',
                     #'2012-01-18/PACMAN_OBJ_ASTRO_019_0007.fits', # glicthes inside!
                     # reset delay PRIMET!
                     #'2012-01-18/PACMAN_OBJ_ASTRO_019_0008.fits', # glitches inside!
    '2012-01-18/PACMAN_OBJ_ASTRO_019_0009.fits',
    '2012-01-18/PACMAN_OBJ_ASTRO_019_0010.fits',
    '2012-01-18/PACMAN_OBJ_ASTRO_019_0011.fits',
    '2012-01-18/PACMAN_OBJ_ASTRO_019_0012.fits',
    '2012-01-18/PACMAN_OBJ_ASTRO_019_0013.fits',
    '2012-01-18/PACMAN_OBJ_ASTRO_019_0014.fits',
                     ]

hd65297_019 = {'SEP':35, 'PA':341,
               #'M0 MJD 55945.00 55945.09':-0.017,
               #'M0 MJD 55945.09 55945.12':-0.017,
               'M0 MJD 55945.12 55945.99':-0.022}

files_hd66598_019 = ['2012-01-18/PACMAN_OBJ_ASTRO_019_0015.fits',
                     '2012-01-18/PACMAN_OBJ_ASTRO_019_0016.fits',
                     '2012-01-18/PACMAN_OBJ_ASTRO_019_0017.fits',
                     '2012-01-18/PACMAN_OBJ_ASTRO_019_0018.fits',
                     '2012-01-18/PACMAN_OBJ_ASTRO_019_0019.fits',
                     '2012-01-18/PACMAN_OBJ_ASTRO_019_0020.fits',
                     '2012-01-18/PACMAN_OBJ_ASTRO_019_0021.fits',
                     '2012-01-18/PACMAN_OBJ_ASTRO_019_0022.fits',
                     '2012-01-18/PACMAN_OBJ_ASTRO_019_0023.fits',
                     '2012-01-18/PACMAN_OBJ_ASTRO_019_0024.fits',
                     '2012-01-18/PACMAN_OBJ_ASTRO_019_0025.fits',
                     '2012-01-18/PACMAN_OBJ_ASTRO_019_0026.fits',
                     '2012-01-18/PACMAN_OBJ_ASTRO_019_0027.fits',
                     '2012-01-18/PACMAN_OBJ_ASTRO_019_0028.fits',
                     '2012-01-18/PACMAN_OBJ_ASTRO_019_0029.fits',
                     '2012-01-18/PACMAN_OBJ_ASTRO_019_0030.fits',
                     '2012-01-18/PACMAN_OBJ_ASTRO_019_0031.fits',
                     ]

hd66598_019_simple={'SEP':35, 'PA':315-180,
                    'M0':-0.034}
hd66598_019 = {'SEP':35, 'PA':315-180,
               'M0':-0.034,
               'AT4 x0': -0.484,
               'AT4 y0': -0.365,
               'AT4 a0': 10.871,
               'AT4 phi0': 164.727,
               'AT4 a1': 0.477,
               'AT4 phi1': 58.716,
               'AT4 phi2': -68.764,
               'AT3 x0': 0.568,
               'AT3 y0': -0.406,
               'AT3 a0': 2.129,
               'AT3 phi0': 0.957,
               'AT3 a1': 0.863,
               'AT3 phi1': -3.755,
               'AT3 phi2':  -54.438,
               'ATPUP scale':1}

files_hd100286J = ['2012-01-19/PACMAN_OBJ_ASTRO_020_0005.fits',
                   '2012-01-19/PACMAN_OBJ_ASTRO_020_0006.fits',
                   '2012-01-19/PACMAN_OBJ_ASTRO_020_0007.fits',
                   '2012-01-19/PACMAN_OBJ_ASTRO_020_0008.fits',
                   '2012-01-19/PACMAN_OBJ_ASTRO_020_0009.fits',
                   '2012-01-19/PACMAN_OBJ_ASTRO_020_0010.fits',
                   '2012-01-19/PACMAN_OBJ_ASTRO_020_0011.fits',
                   '2012-01-19/PACMAN_OBJ_ASTRO_020_0012.fits',]

# ======================
# Fit all HD 66598 DATA:
# ======================
#astromNew.fitListOfFiles('', directory='/Users/amerand/DATA/PRIMA/HD66598/', firstGuess={'SEP':35, 'PA':100, 'M0 MJD 55945.00 55945.99':-0.034, 'M0 MJD 55889.0 55889.999':-0.009, 'M0 MJD 55890.0 55890.999':-0.010})


def analyseCorrection(files=files_hd65297_019):
    # traditional fit:
    titles = ['no corr.',
              'blind corr.',
              'adj. scale',
              'adj. angle',
              'adj. all']
    firstGuess=[{'SEP':35, 'PA':135, 'M0':-0.03, 'PUPSCALE':0, 'PUPANGLE':0},
                {'SEP':35, 'PA':135, 'M0':-0.03, 'PUPSCALE':1, 'PUPANGLE':0},
                {'SEP':35, 'PA':135, 'M0':-0.03, 'PUPSCALE':1, 'PUPANGLE':0},
                {'SEP':35, 'PA':135, 'M0':-0.03, 'PUPSCALE':1, 'PUPANGLE':0},
                {'SEP':35, 'PA':135, 'M0':-0.03, 'PUPSCALE':1, 'PUPANGLE':0}]
    doNotFit = [['PUPSCALE', 'PUPANGLE'],
                ['PUPSCALE', 'PUPANGLE'],
                ['PUPANGLE'],
                ['PUPSCALE'],
                []]

    tmp = []
    for k in range(len(titles)):
        tmp.append(
            astromNew.fitListOfFiles(files, directory=data_directory,
                                     firstGuess=firstGuess[k],
                                     doNotFit=doNotFit[k],
                                     plot=False, verbose=0))
    k_ = ['SEP', 'PA', 'M0', 'PUPSCALE', 'PUPANGLE']

    print '| |',
    for i in range(len(k_)):
        print k_[i], '|',
    print 'CHI2|RMS|'
    print '|-'
    for i in range(len(titles)):
        print '|', titles[i], '|',
        for k in k_:
            print round(tmp[i][0][k], 5),'|',
        print round(tmp[i][2], 2), '|',round(tmp[i][3],2), '|'
        print '| |',
        for k in k_:
            print round(tmp[i][1][k], 5) if tmp[i][1][k]>0 else '','|',
        print  '| |'


# ---------------------- SPLIT HD 18622 ---------------
def gaussian(x, params):
    """
    params should contain: 'a', 'x0', 'w'; may contain bias
    """
    res = params['a']*np.exp(-(x-params['x0'])**2/params['w']**2)
    if 'bias' in params.keys():
        res += params['bias']
    return res

def fitHistGaussian(x, bins=50):
    h = np.histogram(x, bins=bins)
    xh = 0.5*(h[1][:-1]+h[1][1:])
    best, uncer, chi2, model = \
          dpfit.leastsqFit(gaussian, xh,
                               {'a':0.1, 'x0':0.0, 'w':10.}, h[0])
    return (best, uncer, xh, model)

def split():
    """
    Analyse the dual fringe tracking data taken in SPLIT mode.
    0001, 0002: NORMAL
    0003, 0004: SWAPPED
    0005, 0006: NORMAL/SPLIT
    0007, 0008: SWAPPED/SPLIT
    """
    # -- open reduced files: --
    a5 = pyfits.open(data_directory+
                     '2012-01-16/PACMAN_OBJ_ASTRO_017_0005_RED.fits')
    a6 = pyfits.open(data_directory+
                     '2012-01-16/PACMAN_OBJ_ASTRO_017_0006_RED.fits')
    a7 = pyfits.open(data_directory+
                     '2012-01-16/PACMAN_OBJ_ASTRO_017_0007_RED.fits')
    a8 = pyfits.open(data_directory+
                     '2012-01-16/PACMAN_OBJ_ASTRO_017_0008_RED.fits')

    pyplot.figure(10)
    pyplot.clf()
    bins = 300
    # -- gaussian fit: --
    h5 = fitHistGaussian(a5['ASTROMETRY_RAW'].data.field('D_AL')*1e6 -
                np.median(a6['ASTROMETRY_RAW'].data.field('D_AL')*1e6),
                         bins=bins)
    h6 = fitHistGaussian(a6['ASTROMETRY_RAW'].data.field('D_AL')*1e6 -
                      np.median(a6['ASTROMETRY_RAW'].data.field('D_AL')*1e6),
                         bins=bins)
    h7 = fitHistGaussian(a7['ASTROMETRY_RAW'].data.field('D_AL')*1e6 -
                      np.median(a6['ASTROMETRY_RAW'].data.field('D_AL')*1e6),
                         bins=bins)
    h8 = fitHistGaussian(a8['ASTROMETRY_RAW'].data.field('D_AL')*1e6 -
                      np.median(a6['ASTROMETRY_RAW'].data.field('D_AL')*1e6),
                         bins=bins)
    # -- plot histogram and corresponding gaussian fit: --
    pyplot.hist(a5['ASTROMETRY_RAW'].data.field('D_AL')*1e6 -
                np.median(a6['ASTROMETRY_RAW'].data.field('D_AL')*1e6),
                bins=bins , color='c', alpha=0.5, label='NORMAL')
    pyplot.plot(h5[2], h5[3], color='c', linewidth=5, alpha=0.5)

    pyplot.hist(a6['ASTROMETRY_RAW'].data.field('D_AL')*1e6 -
                np.median(a6['ASTROMETRY_RAW'].data.field('D_AL')*1e6),
                bins=bins , color='b', alpha=0.5, label='NORMAL')
    pyplot.plot(h6[2], h6[3], color='b', linewidth=5, alpha=0.5)

    pyplot.hist(a7['ASTROMETRY_RAW'].data.field('D_AL')*1e6 -
                np.median(a6['ASTROMETRY_RAW'].data.field('D_AL')*1e6),
                bins=bins , color='r', alpha=0.5, label='SWAPPED')
    pyplot.plot(h7[2], h7[3], color='r', linewidth=5, alpha=0.5)

    pyplot.hist(a8['ASTROMETRY_RAW'].data.field('D_AL')*1e6 -
                np.median(a6['ASTROMETRY_RAW'].data.field('D_AL')*1e6),
                bins=bins , color='y', alpha=0.5, label='SWAPPED')
    pyplot.plot(h8[2], h8[3], color='y', linewidth=5, alpha=0.5)

    pyplot.legend()
    pyplot.xlabel('DOPD ($\mu$m)')
    pyplot.xlim(-20,20)
    print 'difference:',\
          np.median(a6['ASTROMETRY_RAW'].data.field('D_AL')*1e6)-\
          np.median(a7['ASTROMETRY_RAW'].data.field('D_AL')*1e6), 'microns'
    a5.close(); a6.close(); a7.close(); a8.close()
