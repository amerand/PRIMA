import os
import numpy
import scipy
import pyfits
from matplotlib import pyplot

import prima # low level data reduction
import astrom # astrometric data reduction

"""
here the astrometry dataset that we agreed on using for the separation
fitting / residual analysis.  For completeness, I also included the
corresponding fast scanning files.
"""

# external HD:
data_directory = '/Volumes/DATA/PRIMA/COMM15/'

files_HD10360_199 = ['20110717/PACMAN_OBJ_ASTRO_199_0012.fits',
                     '20110717/PACMAN_OBJ_ASTRO_199_0013.fits',
                     '20110717/PACMAN_OBJ_ASTRO_199_0017.fits',
                     '20110717/PACMAN_OBJ_ASTRO_199_0018.fits',
                     '20110717/PACMAN_OBJ_ASTRO_199_0019.fits']

files_195 =['20110713/PACMAN_OBJ_ASTRO_195_0049.fits', 
            '20110713/PACMAN_OBJ_ASTRO_195_0050.fits']

files_200_HD156274 =['20110718/PACMAN_OBJ_ASTRO_200_0001.fits', # 0
                     '20110718/PACMAN_OBJ_ASTRO_200_0002.fits', # 1
                     '20110718/PACMAN_OBJ_ASTRO_200_0003.fits', # 2 SW
                     '20110718/PACMAN_OBJ_ASTRO_200_0004.fits', # 3 SW
                     '20110718/PACMAN_OBJ_ASTRO_200_0005.fits', # 4
                     '20110718/PACMAN_OBJ_ASTRO_200_0006.fits', # 5 rotation -20 deg
                     #'20110718/PACMAN_OBJ_ASTRO_200_0007_final_merged.fits',
                     '20110718/PACMAN_OBJ_ASTRO_200_0008.fits', # 6 nominal position
                     '20110718/PACMAN_OBJ_ASTRO_200_0009.fits', # 7 clean
                     #'20110718/PACMAN_OBJ_ASTRO_200_0010_final_merged.fits',
                     '20110718/PACMAN_OBJ_ASTRO_200_0001_0001.fits', #  8 
                     '20110718/PACMAN_OBJ_ASTRO_200_0002_0001.fits', #  9: SW
                     '20110718/PACMAN_OBJ_ASTRO_200_0003_0001.fits', # 10: SW
                     '20110718/PACMAN_OBJ_ASTRO_200_0004_0001.fits', # 11: SW, -20 deg
                     '20110718/PACMAN_OBJ_ASTRO_200_0005_0001.fits', # 12: SW, +20 deg
                     '20110718/PACMAN_OBJ_ASTRO_200_0006_0001.fits', # 13: SW, 
                     '20110718/PACMAN_OBJ_ASTRO_200_0007.fits', # 14: SW
                     '20110718/PACMAN_OBJ_ASTRO_200_0009_0001.fits', # 15
                     '20110718/PACMAN_OBJ_ASTRO_200_0010.fits', #16
                     '20110718/PACMAN_OBJ_ASTRO_200_0011.fits',
                     #'20110718/PACMAN_OBJ_ASTRO_200_0012.fits',
                     #'20110718/PACMAN_OBJ_ASTRO_200_0013.fits',
                     '20110718/PACMAN_OBJ_ASTRO_200_0014.fits',
                     '20110718/PACMAN_OBJ_ASTRO_200_0015.fits',
                     '20110718/PACMAN_OBJ_ASTRO_200_0016.fits',
                     '20110718/PACMAN_OBJ_ASTRO_200_0017.fits',
                     '20110718/PACMAN_OBJ_ASTRO_200_0018.fits',
                     #'20110718/PACMAN_OBJ_ASTRO_200_0019.fits',
                     #'20110718/PACMAN_OBJ_ASTRO_200_0020.fits',
                     ]

#what = 'PREVIOUS_RED'
what = 'CORRECTED_RED'

files_201_HD156274 =['20110719/'+what+'/PACMAN_OBJ_ASTRO_201_0001.fits', 
                     '20110719/'+what+'/PACMAN_OBJ_ASTRO_201_0002.fits', 
                     '20110719/'+what+'/PACMAN_OBJ_ASTRO_201_0003.fits',
                     '20110719/'+what+'/PACMAN_OBJ_ASTRO_201_0004.fits',
                     '20110719/'+what+'/PACMAN_OBJ_ASTRO_201_0005.fits',
                     '20110719/'+what+'/PACMAN_OBJ_ASTRO_201_0006.fits',
                     '20110719/'+what+'/PACMAN_OBJ_ASTRO_201_0008.fits',
                     '20110719/'+what+'/PACMAN_OBJ_ASTRO_201_0009.fits',
                     '20110719/'+what+'/PACMAN_OBJ_ASTRO_201_0010.fits',
                     '20110719/'+what+'/PACMAN_OBJ_ASTRO_201_0011.fits',
                     '20110719/'+what+'/PACMAN_OBJ_ASTRO_201_0012.fits',
                     '20110719/'+what+'/PACMAN_OBJ_ASTRO_201_0013.fits',
                     '20110719/'+what+'/PACMAN_OBJ_ASTRO_201_0014.fits',
                     '20110719/'+what+'/PACMAN_OBJ_ASTRO_201_0015.fits',
                     '20110719/'+what+'/PACMAN_OBJ_ASTRO_201_0016.fits',
                     '20110719/'+what+'/PACMAN_OBJ_ASTRO_201_0017.fits',
                     '20110719/'+what+'/PACMAN_OBJ_ASTRO_201_0018.fits',
                     '20110719/'+what+'/PACMAN_OBJ_ASTRO_201_0019.fits',
                     '20110719/'+what+'/PACMAN_OBJ_ASTRO_201_0020.fits',
                     '20110719/'+what+'/PACMAN_OBJ_ASTRO_201_0021.fits',
                     '20110719/'+what+'/PACMAN_OBJ_ASTRO_201_0022.fits',
                     '20110719/'+what+'/PACMAN_OBJ_ASTRO_201_0023.fits',
                     '20110719/'+what+'/PACMAN_OBJ_ASTRO_201_0024.fits',
                     '20110719/'+what+'/PACMAN_OBJ_ASTRO_201_0025.fits']

def analyse195():
    """
    astrometric fit
    """
    files = files_195
    selection = [0,1]
    astrom.interpListOfFiles(files, data_directory,
                             plot=True,
                             fit_param  =[1,    1,    1],
                             first_guess=[8.0, -90.0,-1],
                             fit_only_files=selection)
    return


def analyse200_HD156274(reduction=False):
    """
    astrometric fit
    """

    files = files_200_HD156274
    print len(files)

    if reduction:
        for f in files:
            a = prima.drs(os.path.join(data_directory,f))
            a.astrometryFtk(writeOut=True, max_err_um=4.0, max_GD_um=3.0, 
                            correctA_B=True, overwrite=True, sigma_clipping=6.0)
            del a

    selection = [0,1,4,5,6,7,8,9,10,11,17,18,19] # normal
    
    selection = range(len(files)) # ALL files
    #selection = filter(lambda x: x!=6, selection) # merged
    #selection = filter(lambda x: x!=9, selection) # merged
    #selection = filter(lambda x: x!=20, selection) # merged

    print len(selection)
    astrom.interpListOfFiles(files, data_directory,
                             plot=True,
                             fit_param  =[1, 1, 1,      0,0,0],
                             first_guess=[8,256,5.15523,0,0,0],
                             fit_only_files=selection,
                             maxResiduals=10)
    return

def analyse201_HD156274(reReduce=False):
    """
    astrometric fit
    STS-AT4 rotation test
    """

    files = files_201_HD156274
    print len(files)

    if reReduce:
        for f in files:
            a = prima.drs(os.path.join(data_directory,f))
            a.astrometryFtk(writeOut=True, max_GD_um = 2.0,
                            overwrite=True, max_err_um=10, sigma_clipping=5.0)
            del a
    
    selection = range(len(files)) # ALL files
    selection = filter(lambda x: x!=17, selection) # 17 has A-B overflow
    selection = filter(lambda x: x!=1, selection) # 17 has A-B overflow
    selection = filter(lambda x: x!=21, selection) # 17 has A-B overflow
    #selection = [0,1,2,3,4,5,6,7,8,9,19,20,21] # normal
    #selection = [0,1,5,9,19,20,21] # normal, no rotation
    #selection = [10,11,12,13,14,15,16,18, 22, 23] # swapped, 17 has A-B overflow
    #selection = [10,18,22,23] # swapped, no rotation
    #selection = [0,1,5,9,19,20,21,10,18,22,23]
    astrom.interpListOfFiles(files, data_directory,
                             plot=True,
                             fit_param  =[1, 1, 1,      0,0,0],
                             first_guess=[10.08,258.02,5.15523,0,0,0],
                             fit_only_files=selection,
                             maxResiduals=35)
    return
    at4 = prima.pssRecorder('/Volumes/DATA/PRIMA/COMM15/pssRecorder/pssguiRecorder_lat4fsm_2011-07-19_23-49-45.dat')
    at3 = prima.pssRecorder('/Volumes/DATA/PRIMA/COMM15/pssRecorder/pssguiRecorder_lat3fsm_2011-07-19_23-49-45.dat')
    pyplot.figure(1)
    pyplot.clf()
    pyplot.plot(at4.mjd[::10], at4.data['Dermec.[deg]'][::10],
                color='orange', linewidth=5, label='AT4 derot', alpha=0.9)
    pyplot.plot(at3.mjd[::10], at3.data['Dermec.[deg]'][::10],
                linestyle='dashed',
                color='green', linewidth=2, label='AT3 derot')
    pyplot.legend()
    pyplot.xlim(5.5762e4+0.01, 5.5762e4+0.17)
    pyplot.annotate('modulation\ncommand', xy=(5.5762e4+0.025, 108),
                    xycoords='data', size=12,
                    xytext=(5.5762e4+0.04, 120), textcoords='data',
                    arrowprops=dict(arrowstyle="->"))
    pyplot.annotate('tracking\nwrapping', xy=(5.5762e4+0.1, 160),
                    xycoords='data', size=12,
                    xytext=(5.5762e4+0.06, 150), textcoords='data',
                    arrowprops=dict(arrowstyle="->"))
    pyplot.annotate('swapping', xy=(5.5762e4+0.067, 60),
                    xycoords='data', size=12,
                    xytext=(5.5762e4+0.03, 50), textcoords='data',
                    arrowprops=dict(arrowstyle="->"))
    pyplot.xlabel('MJD')
    pyplot.ylabel('mechanical position (degrees)')
                  
    return
