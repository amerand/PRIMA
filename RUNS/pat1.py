"""
PRIMA Analysis Team run 1 (March 2012)
"""

import astromNew
from matplotlib import pyplot
import numpy as np
import time
import os
import prima

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

# first part of the March 2012 run, after realignment of AT4
Bonnet0 = {'AT3 ctX':  0.8951, 'AT3 ctY': -2.4985,
           'AT3 stX': -2.2969, 'AT3 stY': -1.1044,
           'AT3 cdX':  0.7432, 'AT3 cdY':  0.9352,
           'AT3 sdX': -0.9482, 'AT3 sdY':  0.6817,
           'AT3 c2dX': 0.9919, 'AT3 c2dY':-0.5354,
           'AT3 s2dX': 0.3387, 'AT3 s2dY': 1.0066,
           
           'AT4 ctX':  2.0353, 'AT4 ctY':  0.5416,
           'AT4 stX':  0.3819, 'AT4 stY': -1.0673,
           'AT4 cdX': -0.2974, 'AT4 cdY':  0.5339,
           'AT4 sdX': -0.3778, 'AT4 sdY': -0.3720,
           'AT4 c2dX':-0.8514, 'AT4 c2dY':-0.3457,
           'AT4 s2dX': 0.3992, 'AT4 s2dY':-0.8741,
           'ATPUP scale':1}

# second part of the March 2012 run, after realignment of AT3
Bonnet1 = {'AT3 ctX': -0.3406, 'AT3 ctY':  0.3256,
           'AT3 stX':  0.2783, 'AT3 stY': -0.0845,
           'AT3 cdX':  0.8271, 'AT3 cdY': -0.1259,
           'AT3 sdX':  0.0764, 'AT3 sdY':  0.8549,
           'AT3 c2dX': 0.1896, 'AT3 c2dY':-0.8780,
           'AT3 s2dX': 0.8439, 'AT3 s2dY': 0.2049,

           'AT4 ctX':  0.7587, 'AT4 ctY': -0.5097,
           'AT4 stX': -0.4426, 'AT4 stY': -0.1049,
           'AT4 cdX':  0.2043, 'AT4 cdY':  0.4618,
           'AT4 sdX': -0.4588, 'AT4 sdY':  0.1874,
           'AT4 c2dX':-0.4698, 'AT4 c2dY': 0.3939,
           'AT4 s2dX':-0.3600, 'AT4 s2dY':-0.4426,
           'ATPUP scale':1}
for k in Bonnet1.keys():
    if not 'scale' in k:
        Bonnet1[k] *= -1 # correction of a bug in Henri's code
        
# guess data directory
#######################

data_directory = '/Volumes/DATA500/PRIMA/TT1/' # external BIG hardrive
if not os.path.isdir(data_directory):
    data_directory = '/Volumes/DATA/PRIMA/TT1/' # external small hardrive
if not os.path.isdir(data_directory):
    data_directory = '/Users/amerand/DATA/PRIMA/TT1/' # internal hardrive
if not os.path.isdir(data_directory):
    print 'WARNING: NO DATA DIRECTORY AVAILABLE!'

def testHD66958_ALL(runOffCorrection=True):
    fi = os.listdir(data_directory+'/comm18')
    fi = filter(lambda x: 'RED.fits' in x , fi)
    fi = ['comm18/'+''.join(f.split('_RED')) for f in fi]
    fi = filter(lambda x: '019' in x , fi)
    fg = {'SEP':35, 'PA':100,
          #'SEP MJD 56000.0':35, 'PA MJD 56000.0':180, 'LINDIR':-142.0, 'LINRATE':-3844.0,
          'M0 MJD 55945.00 55945.99':-0.034,
          #'M0 MJD 55889.0 55889.999':-0.009,
          #'M0 MJD 55890.0 55890.999':-0.010
          }
    if runOffCorrection:
        for k in ATsPUP.keys(): #  
            fg[k] = ATsPUP[k]
    if runOffCorrection=='blind':
        doNotFit=['AT3', 'AT4', 'ATPUP scale', 'LIN']
    else:
        doNotFit=['AT3', 'AT4', 'LIN']
    a = astromNew.fitListOfFiles(fi, data_directory,
                             firstGuess=fg, verbose=2,
                             doNotFit=doNotFit,
                             maxResiduals=30)
    return a

def testHD66958(allX=False, runOffCorrection=True):
    """
    - functionnal test;

    - astrometric fit on HD 66598 to see the amplitude of the
    residuals compared to january
    """
    files_hd66958 = [
        ### March 2012 data
        '2012-03-13/PACMAN_OBJ_ASTRO_074_0001.fits',
        '2012-03-13/PACMAN_OBJ_ASTRO_074_0002.fits',
        '2012-03-13/PACMAN_OBJ_ASTRO_074_0003.fits',
        '2012-03-13/PACMAN_OBJ_ASTRO_074_0004.fits',
        '2012-03-13/PACMAN_OBJ_ASTRO_074_0005.fits',
        # no 0006
        '2012-03-13/PACMAN_OBJ_ASTRO_074_0007.fits',
        '2012-03-13/PACMAN_OBJ_ASTRO_074_0008.fits',
        '2012-03-13/PACMAN_OBJ_ASTRO_074_0009.fits',
        '2012-03-13/PACMAN_OBJ_ASTRO_074_0010.fits',
        '2012-03-13/PACMAN_OBJ_ASTRO_074_0011.fits',
        '2012-03-13/PACMAN_OBJ_ASTRO_074_0012.fits',
        '2012-03-13/PACMAN_OBJ_ASTRO_074_0013.fits',
        '2012-03-13/PACMAN_OBJ_ASTRO_074_0014.fits',
        '2012-03-14/PACMAN_OBJ_ASTRO_075_0001.fits',
        '2012-03-14/PACMAN_OBJ_ASTRO_075_0002.fits',
        '2012-03-14/PACMAN_OBJ_ASTRO_075_0003.fits',
        '2012-03-14/PACMAN_OBJ_ASTRO_075_0004.fits',
        '2012-03-14/PACMAN_OBJ_ASTRO_075_0005.fits',
        '2012-03-14/PACMAN_OBJ_ASTRO_075_0006.fits',
        '2012-03-14/PACMAN_OBJ_ASTRO_075_0007.fits',
        '2012-03-14/PACMAN_OBJ_ASTRO_075_0008.fits',
        '2012-03-14/PACMAN_OBJ_ASTRO_075_0009.fits',
        '2012-03-14/PACMAN_OBJ_ASTRO_075_0010.fits',
        '2012-03-14/PACMAN_OBJ_ASTRO_075_0011.fits',
        '2012-03-14/PACMAN_OBJ_ASTRO_075_0012.fits',
        '2012-03-14/PACMAN_OBJ_ASTRO_075_0013.fits',
        '2012-03-14/PACMAN_OBJ_ASTRO_075_0014.fits',
        '2012-03-14/PACMAN_OBJ_ASTRO_075_0015.fits',
        '2012-03-14/PACMAN_OBJ_ASTRO_075_0016.fits',
        '2012-03-14/PACMAN_OBJ_ASTRO_075_0017.fits',
        '2012-03-14/PACMAN_OBJ_ASTRO_075_0018.fits',
        '2012-03-14/PACMAN_OBJ_ASTRO_075_0019.fits',
        '2012-03-14/PACMAN_OBJ_ASTRO_075_0020.fits',
        '2012-03-14/PACMAN_OBJ_ASTRO_075_0021.fits',
        '2012-03-14/PACMAN_OBJ_ASTRO_075_0022.fits',
        '2012-03-14/PACMAN_OBJ_ASTRO_075_0023.fits',
        #'2012-03-14/PACMAN_OBJ_ASTRO_075_0024.fits'
        ]
    
    fg = {'SEP':35, 'PA':134,
          'M0 MJD 56000.0 56000.07':0.00,
          'M0 MJD 56000.07 56000.99':0.00,
          'M0 MJD 56001.0 56001.05':0.00,
          'M0 MJD 56001.05 56001.99':0.00,
          }
    
    doNotFit=['LIN','AT3', 'AT4']
    if runOffCorrection=='blind':
        doNotFit.extend('scale')
    
    ### Jan 2012 parameters:
    #for k in ATsPUP.keys(): # add 
    #    fg[k] = ATsPUP[k]
    
    ### March 2012 parameters:
    if runOffCorrection:
        for k in Bonnet1.keys(): # add 
            fg[k] = Bonnet1[k]
    
    a = astromNew.fitListOfFiles(files_hd66958,reduceData=False,
                                 directory=data_directory,
                                 firstGuess=fg, plot=True, verbose=1,
                                 doNotFit=doNotFit, allX=allX,
                                 maxResiduals=10,
                                 exportAscii='hd66598.txt'
                                 )
    return 

def test_targetsBegRun(runOffCorrection=False, allX=False):
    files_hd66958 = [
        ###### March 2012 data ##############
        '2012-03-13/PACMAN_OBJ_ASTRO_074_0001.fits',
        '2012-03-13/PACMAN_OBJ_ASTRO_074_0002.fits',
        '2012-03-13/PACMAN_OBJ_ASTRO_074_0003.fits',
        '2012-03-13/PACMAN_OBJ_ASTRO_074_0004.fits',
        '2012-03-13/PACMAN_OBJ_ASTRO_074_0005.fits',
        '2012-03-13/PACMAN_OBJ_ASTRO_074_0006.fits',
        '2012-03-13/PACMAN_OBJ_ASTRO_074_0007.fits',
        '2012-03-13/PACMAN_OBJ_ASTRO_074_0008.fits',
        '2012-03-13/PACMAN_OBJ_ASTRO_074_0009.fits',
        '2012-03-13/PACMAN_OBJ_ASTRO_074_0010.fits',
        '2012-03-13/PACMAN_OBJ_ASTRO_074_0011.fits',
        '2012-03-13/PACMAN_OBJ_ASTRO_074_0012.fits',
        '2012-03-13/PACMAN_OBJ_ASTRO_074_0013.fits',
        '2012-03-13/PACMAN_OBJ_ASTRO_074_0014.fits',
        '2012-03-14/PACMAN_OBJ_ASTRO_075_0001.fits',
        '2012-03-14/PACMAN_OBJ_ASTRO_075_0002.fits',
        '2012-03-14/PACMAN_OBJ_ASTRO_075_0003.fits',
        '2012-03-14/PACMAN_OBJ_ASTRO_075_0004.fits',
        '2012-03-14/PACMAN_OBJ_ASTRO_075_0005.fits',
        '2012-03-14/PACMAN_OBJ_ASTRO_075_0006.fits',
        '2012-03-14/PACMAN_OBJ_ASTRO_075_0007.fits',
        '2012-03-14/PACMAN_OBJ_ASTRO_075_0008.fits',
        '2012-03-14/PACMAN_OBJ_ASTRO_075_0009.fits',
        '2012-03-14/PACMAN_OBJ_ASTRO_075_0010.fits',
        '2012-03-14/PACMAN_OBJ_ASTRO_075_0011.fits',
        '2012-03-14/PACMAN_OBJ_ASTRO_075_0012.fits',
        '2012-03-14/PACMAN_OBJ_ASTRO_075_0013.fits',
        '2012-03-14/PACMAN_OBJ_ASTRO_075_0014.fits',
        '2012-03-14/PACMAN_OBJ_ASTRO_075_0015.fits',
        '2012-03-14/PACMAN_OBJ_ASTRO_075_0016.fits',
        '2012-03-14/PACMAN_OBJ_ASTRO_075_0017.fits',
        '2012-03-14/PACMAN_OBJ_ASTRO_075_0018.fits',
        '2012-03-14/PACMAN_OBJ_ASTRO_075_0019.fits',
        '2012-03-14/PACMAN_OBJ_ASTRO_075_0020.fits',
        '2012-03-14/PACMAN_OBJ_ASTRO_075_0021.fits',
        '2012-03-14/PACMAN_OBJ_ASTRO_075_0022.fits',
        '2012-03-14/PACMAN_OBJ_ASTRO_075_0023.fits',
        '2012-03-15/PACMAN_OBJ_ASTRO_076_0002.fits',
        '2012-03-15/PACMAN_OBJ_ASTRO_076_0003.fits',
        '2012-03-15/PACMAN_OBJ_ASTRO_076_0004.fits',
        '2012-03-15/PACMAN_OBJ_ASTRO_076_0005.fits',
        '2012-03-15/PACMAN_OBJ_ASTRO_076_0006.fits',
        '2012-03-15/PACMAN_OBJ_ASTRO_076_0007.fits',
        '2012-03-15/PACMAN_OBJ_ASTRO_076_0008.fits',
        '2012-03-15/PACMAN_OBJ_ASTRO_076_0009.fits',
        '2012-03-15/PACMAN_OBJ_ASTRO_076_0010.fits',
        '2012-03-15/PACMAN_OBJ_ASTRO_076_0011.fits',
        '2012-03-15/PACMAN_OBJ_ASTRO_076_0012.fits',
        '2012-03-15/PACMAN_OBJ_ASTRO_076_0013.fits',
        '2012-03-15/PACMAN_OBJ_ASTRO_076_0014.fits',
        '2012-03-15/PACMAN_OBJ_ASTRO_076_0015.fits',
        '2012-03-15/PACMAN_OBJ_ASTRO_076_0016.fits',
        '2012-03-15/PACMAN_OBJ_ASTRO_076_0017.fits',
        '2012-03-15/PACMAN_OBJ_ASTRO_076_0018.fits',
        '2012-03-15/PACMAN_OBJ_ASTRO_076_0019.fits',
        '2012-03-15/PACMAN_OBJ_ASTRO_076_0020.fits',
        '2012-03-15/PACMAN_OBJ_ASTRO_076_0021.fits',
        '2012-03-15/PACMAN_OBJ_ASTRO_076_0022.fits',
        '2012-03-15/PACMAN_OBJ_ASTRO_076_0023.fits',
        '2012-03-15/PACMAN_OBJ_ASTRO_076_0024.fits',
        '2012-03-15/PACMAN_OBJ_ASTRO_076_0025.fits',
        '2012-03-15/PACMAN_OBJ_ASTRO_076_0026.fits',]
    
    fg = {'T:HD66598 SEP':35,
          'T:HD66598 PA':134,
          'T:HD66598 M0 MJD 56000.00 56000.07':0.00,
          'T:HD66598 M0 MJD 56000.07 56000.99':0.00,
          'T:HD66598 M0 MJD 56001.00 56001.05':0.00,
          'T:HD66598 M0 MJD 56001.05 56001.99':0.00,
          'T:HD134481 SEP': 26,
          'T:HD134481 PA': 144,
          'T:HD134481 M0 MJD 56002.00 56002.32':0.00,
          'T:HD134481 M0 MJD 56002.32 56002.35':0.00,
          'T:HD134481 M0 MJD 56002.35 56002.99':0.00,
          }
    
    doNotFit=['LIN']
 
    ### Jan 2012 parameters:
    #for k in ATsPUP.keys(): # add 
    #    fg[k] = ATsPUP[k]
    
    ### March 2012 parameters:
    if runOffCorrection:
        for k in Bonnet0.keys(): # add 
            fg[k] = Bonnet0[k]
        doNotFit.extend(['scale','AT3', 'AT4'])
    
    a = astromNew.fitListOfFiles(files_hd66958,reduceData=False,
                                 directory=data_directory,
                                 firstGuess=fg, plot=True, verbose=2,
                                 doNotFit=doNotFit, allX=allX,
                                 maxResiduals=20,
                                 #exportAscii='March12_Phase1.txt'
                                 )
    
    deltaDec = a['BEST']['T:HD134481 SEP']*np.cos(a['BEST']['T:HD134481 PA']*np.pi/180)
    deltaRA_cosDec = a['BEST']['T:HD134481 SEP']*np.sin(a['BEST']['T:HD134481 PA']*np.pi/180)
    deltaRA = deltaRA_cosDec/np.cos(a['DEC']['HD134481']*np.pi/180)
    print 'HD134481: dRA', deltaRA, 'dDEC=', deltaDec
    return 

def test_targetsRun2(runOffCorrection=False, allX=False):
    """
    data collected after realignement of M4/AT3 by Stephane Guisard
    to minimize pupil runout.
    """
    
    files = [
        ###### March 2012 data ##############
        # HD 134481
        '2012-03-16/PACMAN_OBJ_ASTRO_077_0001.fits',
        '2012-03-16/PACMAN_OBJ_ASTRO_077_0002.fits',
        '2012-03-16/PACMAN_OBJ_ASTRO_077_0003.fits',
        '2012-03-16/PACMAN_OBJ_ASTRO_077_0004.fits',
        '2012-03-16/PACMAN_OBJ_ASTRO_077_0006.fits',
        '2012-03-16/PACMAN_OBJ_ASTRO_077_0007.fits',
        '2012-03-16/PACMAN_OBJ_ASTRO_077_0008.fits',
        '2012-03-16/PACMAN_OBJ_ASTRO_077_0009.fits',
        '2012-03-16/PACMAN_OBJ_ASTRO_077_0010.fits',
        '2012-03-16/PACMAN_OBJ_ASTRO_077_0011.fits',
        '2012-03-16/PACMAN_OBJ_ASTRO_077_0012.fits',
        '2012-03-16/PACMAN_OBJ_ASTRO_077_0013.fits',
        '2012-03-16/PACMAN_OBJ_ASTRO_077_0014.fits',
        '2012-03-16/PACMAN_OBJ_ASTRO_077_0015.fits',
        '2012-03-16/PACMAN_OBJ_ASTRO_077_0016.fits',
        '2012-03-16/PACMAN_OBJ_ASTRO_077_0017.fits',
        '2012-03-16/PACMAN_OBJ_ASTRO_077_0018.fits',
        '2012-03-16/PACMAN_OBJ_ASTRO_077_0019.fits',
        '2012-03-16/PACMAN_OBJ_ASTRO_077_0020.fits',
        '2012-03-16/PACMAN_OBJ_ASTRO_077_0021.fits',
        '2012-03-16/PACMAN_OBJ_ASTRO_077_0022.fits',
        '2012-03-16/PACMAN_OBJ_ASTRO_077_0023.fits',
        '2012-03-16/PACMAN_OBJ_ASTRO_077_0024.fits',
        '2012-03-16/PACMAN_OBJ_ASTRO_077_0025.fits',
        '2012-03-16/PACMAN_OBJ_ASTRO_077_0026.fits',
        '2012-03-16/PACMAN_OBJ_ASTRO_077_0027.fits',
        '2012-03-16/PACMAN_OBJ_ASTRO_077_0028.fits',
        '2012-03-16/PACMAN_OBJ_ASTRO_077_0029.fits',
        '2012-03-16/PACMAN_OBJ_ASTRO_077_0030.fits',
        '2012-03-16/PACMAN_OBJ_ASTRO_077_0031.fits',
        # HD 66598
        '2012-03-17/PACMAN_OBJ_ASTRO_078_0002.fits',
        '2012-03-17/PACMAN_OBJ_ASTRO_078_0003.fits',
        '2012-03-17/PACMAN_OBJ_ASTRO_078_0004.fits',
        '2012-03-17/PACMAN_OBJ_ASTRO_078_0005.fits',
        '2012-03-17/PACMAN_OBJ_ASTRO_078_0006.fits',
        '2012-03-17/PACMAN_OBJ_ASTRO_078_0007.fits',
        '2012-03-17/PACMAN_OBJ_ASTRO_078_0008.fits',
        '2012-03-17/PACMAN_OBJ_ASTRO_078_0009.fits',  
        '2012-03-17/PACMAN_OBJ_ASTRO_078_0010.fits',
        '2012-03-17/PACMAN_OBJ_ASTRO_078_0011.fits',  
        '2012-03-17/PACMAN_OBJ_ASTRO_078_0012.fits',
        '2012-03-17/PACMAN_OBJ_ASTRO_078_0013.fits',
        '2012-03-17/PACMAN_OBJ_ASTRO_078_0014.fits',
        '2012-03-17/PACMAN_OBJ_ASTRO_078_0015.fits',
        '2012-03-17/PACMAN_OBJ_ASTRO_078_0016.fits',
        '2012-03-17/PACMAN_OBJ_ASTRO_078_0017.fits',
        '2012-03-17/PACMAN_OBJ_ASTRO_078_0018.fits',
        '2012-03-17/PACMAN_OBJ_ASTRO_078_0019.fits',
        # HD 129926
        '2012-03-17/PACMAN_OBJ_ASTRO_078_0025.fits',
        '2012-03-17/PACMAN_OBJ_ASTRO_078_0026.fits',
        '2012-03-17/PACMAN_OBJ_ASTRO_078_0027.fits',
        '2012-03-17/PACMAN_OBJ_ASTRO_078_0028.fits',
        '2012-03-17/PACMAN_OBJ_ASTRO_078_0029.fits',
        '2012-03-17/PACMAN_OBJ_ASTRO_078_0030.fits',
        '2012-03-17/PACMAN_OBJ_ASTRO_078_0031.fits',
        '2012-03-17/PACMAN_OBJ_ASTRO_078_0032.fits',
        '2012-03-17/PACMAN_OBJ_ASTRO_078_0033.fits',
        '2012-03-17/PACMAN_OBJ_ASTRO_078_0034.fits',
        '2012-03-17/PACMAN_OBJ_ASTRO_078_0035.fits',
        '2012-03-17/PACMAN_OBJ_ASTRO_078_0036.fits',
        #'2012-03-17/PACMAN_OBJ_ASTRO_078_0037.fits',# BAD
        #'2012-03-17/PACMAN_OBJ_ASTRO_078_0038.fits',# BAD
        #'2012-03-17/PACMAN_OBJ_ASTRO_078_0039.fits',# BAD
        '2012-03-17/PACMAN_OBJ_ASTRO_078_0040.fits',
        '2012-03-17/PACMAN_OBJ_ASTRO_078_0041.fits',
        '2012-03-17/PACMAN_OBJ_ASTRO_078_0042.fits',
        '2012-03-17/PACMAN_OBJ_ASTRO_078_0043.fits',
        '2012-03-17/PACMAN_OBJ_ASTRO_078_0044.fits',
        '2012-03-17/PACMAN_OBJ_ASTRO_078_0045.fits',
        ]
    
    # parameters for the fit:
    fg = {'T:HD129926 SEP':8,
          'T:HD129926 PA':124,
          'T:HD129926 M0 MJD 56004.00 56004.35':0.00,
          'T:HD129926 M0 MJD 56004.35 56004.99':0.00,
          'T:HD66598 SEP':35,
          'T:HD66598 PA':134,
          'T:HD66598 M0 MJD 56004.00 56004.99':0.00,
          'T:HD134481 SEP': 26,
          'T:HD134481 PA': 144,
          'T:HD134481 M0 MJD 56003.00 56003.30':0.00,
          'T:HD134481 M0 MJD 56003.30 56003.99':0.00,
         }
    doNotFit=[]
    ### March 2012 pupil runout parameters:
    if runOffCorrection:
        for k in Bonnet1.keys(): # add 
            fg[k] = Bonnet1[k]
        doNotFit.extend(['scale',
                         #'AT4 ct', 'AT4 st', 'AT4 cd', 'AT4 sd', # fit 2ROT
                         #'AT4 ct', 'AT4 st', 'AT4 c2d', 'AT4 s2d', # fit ROT
                         #'AT4 c2d', 'AT4 s2d', #'AT4 c2d', 'AT4 s2d', # fit THETA
                         'AT4',
                         'AT3'])
        
    a = astromNew.fitListOfFiles(files,reduceData=False,
                                 directory=data_directory,
                                 firstGuess=fg, plot=True, verbose=2,
                                 doNotFit=doNotFit, allX=allX,
                                 maxResiduals=10,
                                 #exportAscii='March12_Phase2.txt'
                                 )

    return

def test_rotation():
    """
    rotation modulation
    """
    
    files = ['2012-03-18/PACMAN_OBJ_ASTRO_079_0001.fits',
             '2012-03-18/PACMAN_OBJ_ASTRO_079_0004.fits',
             '2012-03-18/PACMAN_OBJ_ASTRO_079_0005.fits',
             '2012-03-18/PACMAN_OBJ_ASTRO_079_0006.fits',
             '2012-03-18/PACMAN_OBJ_ASTRO_079_0007.fits',
             '2012-03-18/PACMAN_OBJ_ASTRO_079_0008.fits',
             '2012-03-18/PACMAN_OBJ_ASTRO_079_0009.fits',
             '2012-03-18/PACMAN_OBJ_ASTRO_079_0010.fits',
             '2012-03-18/PACMAN_OBJ_ASTRO_079_0011.fits',
             '2012-03-18/PACMAN_OBJ_ASTRO_079_0012.fits',
             '2012-03-18/PACMAN_OBJ_ASTRO_079_0013.fits',
             '2012-03-18/PACMAN_OBJ_ASTRO_079_0014.fits',
             '2012-03-18/PACMAN_OBJ_ASTRO_079_0015.fits',
             '2012-03-18/PACMAN_OBJ_ASTRO_079_0016.fits',
             '2012-03-18/PACMAN_OBJ_ASTRO_079_0020.fits',
             '2012-03-18/PACMAN_OBJ_ASTRO_079_0021.fits',
             '2012-03-18/PACMAN_OBJ_ASTRO_079_0022.fits',
             '2012-03-18/PACMAN_OBJ_ASTRO_079_0024.fits',
             '2012-03-18/PACMAN_OBJ_ASTRO_079_0026.fits',
             '2012-03-18/PACMAN_OBJ_ASTRO_079_0027.fits',
             '2012-03-18/PACMAN_OBJ_ASTRO_079_0028.fits',
             '2012-03-18/PACMAN_OBJ_ASTRO_079_0029.fits',
             ]
    
    # parameters for the fit:
    fg = {'T:HD100286J SEP':10.0,
          'T:HD100286J PA':210.0,
          'T:HD100286J M0 MJD 56005.00 56005.14':0.0,
          'T:HD100286J M0 MJD 56005.14 56005.99':0.0, 
          'T:HD129926 SEP':8.0,
          'T:HD129926 PA':124.0,
          'T:HD129926 M0 MJD 56005.00 56005.32':0.0,
          'T:HD129926 M0 MJD 56005.32 56005.99':0.0,
         }
    doNotFit=[]
        
    a = astromNew.fitListOfFiles(files,reduceData=False,
                                 directory=data_directory,
                                 firstGuess=fg, plot=True, verbose=2,
                                 doNotFit=doNotFit, allX=False,
                                 #maxResiduals=10,
                                 #exportAscii='March12_Phase2.txt'
                                 )

    return

def test_rotation2():
    """
    rotation modulation, second night
    """
    files = ['2012-03-19/PACMAN_OBJ_ASTRO_080_0009.fits',
             '2012-03-19/PACMAN_OBJ_ASTRO_080_0012.fits',
             '2012-03-19/PACMAN_OBJ_ASTRO_080_0013.fits',
             '2012-03-19/PACMAN_OBJ_ASTRO_080_0014.fits',
             '2012-03-19/PACMAN_OBJ_ASTRO_080_0015.fits',
             '2012-03-19/PACMAN_OBJ_ASTRO_080_0016.fits',
             '2012-03-19/PACMAN_OBJ_ASTRO_080_0017.fits',
             '2012-03-19/PACMAN_OBJ_ASTRO_080_0018.fits',
             '2012-03-19/PACMAN_OBJ_ASTRO_080_0019.fits',
             '2012-03-19/PACMAN_OBJ_ASTRO_080_0020.fits',
             '2012-03-19/PACMAN_OBJ_ASTRO_080_0021.fits',
             '2012-03-19/PACMAN_OBJ_ASTRO_080_0022.fits',
             '2012-03-19/PACMAN_OBJ_ASTRO_080_0023.fits',
             '2012-03-19/PACMAN_OBJ_ASTRO_080_0024.fits',
             '2012-03-19/PACMAN_OBJ_ASTRO_080_0025.fits',
             '2012-03-19/PACMAN_OBJ_ASTRO_080_0026.fits',
             '2012-03-19/PACMAN_OBJ_ASTRO_080_0027.fits',
             '2012-03-19/PACMAN_OBJ_ASTRO_080_0028.fits',
             '2012-03-19/PACMAN_OBJ_ASTRO_080_0029.fits',
             '2012-03-19/PACMAN_OBJ_ASTRO_080_0030.fits',
             ]
    
    # parameters for the fit:
    fg = {'T:HD100286J SEP':10.0,
          'T:HD100286J PA':210.0,
          'T:HD100286J M0 MJD 56006.00 56006.99':0.0, 
         }
    doNotFit=[]
        
    a = astromNew.fitListOfFiles(files,reduceData=False,
                                 directory=data_directory,
                                 firstGuess=fg, plot=True, verbose=2,
                                 doNotFit=doNotFit, allX=False,
                                 #maxResiduals=10,
                                 #exportAscii='March12_Phase2.txt'
                                 )

    return

def reduceAllData():
    dataFiles=[]
    dataFiles.extend(['2012-03-13/PACMAN_OBJ_ASTRO_074_00'+k+'.fits' for k in
                      ['01', '02', '03', '04', '05', '06', '07', '08',
                       '09', '10', '11', '12', '13', '14']])
    dataFiles.extend(['2012-03-14/PACMAN_OBJ_ASTRO_075_00'+k+'.fits' for k in
                      ['01', '02', '03', '04', '05', '06', '07', '08',
                       '09', '10', '11', '12', '13', '14', '15', '16',
                       '17', '18', '19', '20', '21', '22', '23', '24']])
    dataFiles.extend(['2012-03-15/PACMAN_OBJ_ASTRO_076_00'+k+'.fits' for k in
                      ['01', '02', '03', '04', '05', '06', '07', '08',
                       '09', '10', '11', '12', '13', '14', '15', '16',
                       '17', '18', '19', '20', '21', '22', '23', '24',
                       '25', '26']])
    dataFiles.extend(['2012-03-16/PACMAN_OBJ_ASTRO_077_00'+k+'.fits' for k in
                      ['01', '02', '03', '04', '05', '06', '07', '08',
                       '09', '10', '11', '12', '13', '14', '15', '16',
                       '17', '18', '19', '20', '21', '22', '23', '24',
                       '25', '26', '27', '28', '29', '30', '31']])
    dataFiles.extend(['2012-03-17/PACMAN_OBJ_ASTRO_078_00'+k+'.fits' for k in
                      ['01', '02', '03', '04', '05', '06', '07', '08',
                       '09', '10', '11', '12', '13', '14', '15', '16',
                       '17', '18', '19', '20', '21', '22', '23', '24',
                       '25', '26', '27', '28', '29', '30', '31', '32',
                       '33', '34', '35', '36', '37', '38', '39', '40',
                       '41', '42', '43', '44', '45']])
    dataFiles.extend(['2012-03-18/PACMAN_OBJ_ASTRO_079_00'+k+'.fits' for k in
                      ['01', '02', '03', '04', '05', '06', '07', '08',
                       '09', '10', '11', '12', '13', '14', '15', '16',
                       '17', '18', '19', '20', '21', '22', '23', '24',
                       '25', '26', '27', '28', '29']])
    dataFiles.extend(['2012-03-19/PACMAN_OBJ_ASTRO_080_00'+k+'.fits' for k in
                      ['01', '02', '03', '04', '05', '06', '07', '08',
                       '09', '10', '11', '12', '13', '14', '15', '16',
                       '17', '18', '19', '20', '21', '22', '23', '24',
                       '25', '26', '27', '28', '29', '30']])
    print 'reducing', len(dataFiles), 'files'

    for f in dataFiles:
        print '='*10, f, '='*10
        tmp = './reducePrimaFile.py '+data_directory+f+\
              ' pssguiRecorderDir='+data_directory+'PSS/'
        os.system(tmp)

    return

def plotAllFtk():
    tmp = os.popen('dfits '+data_directory+'/2012-03*/PACMAN*RED.fits |'+
                   ' fitsort INS.MODE OCS.PS.ID RED.PRO.LR_OPDC RED.PRO.LR_DOPDC').readlines()[2:]
    swapped = np.array([x.split()[1]=='SWAPPED' for x in tmp])
    normal = np.array([x.split()[1]=='NORMAL' for x in tmp])
    FTKopdc =  np.array([float(x.split()[3]) for x in tmp])
    FTKdopdc = np.array([float(x.split()[4]) for x in tmp])
    FTKfsuA = np.array([float(x.split()[3]) if x.split()[1]=='SWAPPED' else
                        float(x.split()[4]) for x in tmp])
    FTKfsuB = np.array([float(x.split()[3]) if x.split()[1]=='NORMAL' else
                        float(x.split()[4]) for x in tmp])

    pyplot.figure(0, figsize=(9,9))
    pyplot.clf()
    pyplot.subplot(221)
    pyplot.hist(FTKfsuA[np.where(swapped)], bins=20, color='r',
                alpha=0.5, label='FSUA, OPDC', normed=True, linewidth=1)
    pyplot.hist(FTKfsuB[np.where(normal)], bins=20, color='b',
                alpha=0.5, label='FSUB, OPDC', normed=True)
    pyplot.legend(loc='upper left')
    pyplot.xlabel('locking ratio')
    pyplot.title('OPDC')

    pyplot.subplot(222)
    pyplot.hist(FTKfsuA[np.where(normal)], bins=20, color='y',
                alpha=0.5, label='FSUA, DOPDC', normed=True, linewidth=1)
    pyplot.hist(FTKfsuB[np.where(swapped)], bins=20, color='c',
                alpha=0.5, label='FSUB, DOPDC', normed=True)
    pyplot.legend(loc='upper left')
    pyplot.xlabel('locking ratio')
    pyplot.title('DOPDC')

    pyplot.subplot(223)
    pyplot.hist(FTKfsuA[np.where(swapped)], bins=20, color='r',
                alpha=0.5, label='FSUA, OPDC', normed=True, linewidth=1)
    pyplot.hist(FTKfsuA[np.where(normal)], bins=20, color='y',
                alpha=0.5, label='FSUA, DOPDC', normed=True)
    pyplot.legend(loc='upper left')
    pyplot.xlabel('locking ratio')
    pyplot.title('FSUA')
    
    pyplot.subplot(224)
    pyplot.hist(FTKfsuB[np.where(normal)], bins=20, color='b',
                alpha=0.5, label='FSUB, OPDC', normed=True, linewidth=1)
    pyplot.hist(FTKfsuB[np.where(swapped)], bins=20, color='c',
                alpha=0.5, label='FSUB, DOPDC', normed=True)
    pyplot.legend(loc='upper left')
    pyplot.xlabel('locking ratio')
    pyplot.title('FSUB')

    pyplot.figure(1, figsize=(6,9))
    pyplot.clf()
    pyplot.hist((FTKfsuB/FTKfsuA)[np.where(swapped)], bins=20, color='g',
                alpha=0.5, label='SWAPPED: FSUB / FSUA', normed=True, linewidth=1)
    pyplot.hist((FTKfsuA/FTKfsuB)[np.where(normal)], bins=20, color='m',
                alpha=0.5, label='NORMAL: FSUA / FSUB', normed=True)
    pyplot.legend(loc='upper left')
    pyplot.xlabel('locking ratio: DOPDC/OPDC')
    pyplot.xlim(0,1.2)
    return
