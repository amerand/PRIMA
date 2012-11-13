import os
import numpy as np
import scipy
import pyfits
import time
from matplotlib import pylab

import prima # low level data reduction
import astrom # astrometric data reduction
import myfit
import astro
"""
here the astrometry dataset that we agreed on using for the separation
fitting / residual analysis.


comm16.doAll(comm16.files_241_HD202730, maxResiduals=5)
comm16.doAll(comm16.files_241_HD202730, maxResiduals=2, bootstrap=True)

# HD202730 is supposed to be a P=1500y, which leads to ~0.1 mas per day
astro.EstimateApparentOrbitalSeparation(M1='A5V',M2='A7V',plx=.033,P=1500*365)

# HD10360 -> 0.25 mas per day (P~483yrs, semi_a~10.2 arcsec)
"""

data_directory = '/Volumes/DATA/PRIMA/COMM16/' # external hardrive
data_directory = '/Volumes/DATA500/PRIMA/COMM16/'

files_236 =['2011-08-23/PACMAN_OBJ_ASTRO_236_0004.fits',
            '2011-08-23/PACMAN_OBJ_ASTRO_236_0005.fits',
            '2011-08-23/PACMAN_OBJ_ASTRO_236_0006.fits',
            '2011-08-23/PACMAN_OBJ_ASTRO_236_0007.fits',
            '2011-08-23/PACMAN_OBJ_ASTRO_236_0008.fits',
            '2011-08-23/PACMAN_OBJ_ASTRO_236_0009.fits',
            '2011-08-23/PACMAN_OBJ_ASTRO_236_0010.fits',
            '2011-08-23/PACMAN_OBJ_ASTRO_236_0011.fits',
            '2011-08-23/PACMAN_OBJ_ASTRO_236_0012.fits',
            '2011-08-23/PACMAN_OBJ_ASTRO_236_0013.fits']

files_238 =['2011-08-25/PACMAN_OBJ_ASTRO_238_0002.fits',
            '2011-08-25/PACMAN_OBJ_ASTRO_238_0003.fits',
            '2011-08-25/PACMAN_OBJ_ASTRO_238_0004.fits',
            '2011-08-25/PACMAN_OBJ_ASTRO_238_0005.fits',
            '2011-08-25/PACMAN_OBJ_ASTRO_238_0006.fits',
            '2011-08-25/PACMAN_OBJ_ASTRO_238_0007.fits',
            '2011-08-25/PACMAN_OBJ_ASTRO_238_0008.fits',
            '2011-08-25/PACMAN_OBJ_ASTRO_238_0009.fits',
            ]

files_238_HD10360 = [
    '2011-08-25/PACMAN_OBJ_ASTRO_238_0011.fits',
    '2011-08-25/PACMAN_OBJ_ASTRO_238_0012.fits',
    '2011-08-25/PACMAN_OBJ_ASTRO_238_0013.fits',
    '2011-08-25/PACMAN_OBJ_ASTRO_238_0014.fits',
    '2011-08-25/PACMAN_OBJ_ASTRO_238_0015.fits',
    '2011-08-25/PACMAN_OBJ_ASTRO_238_0016.fits',
    ]


files_241_HD202730 = ['2011-08-28/PACMAN_OBJ_ASTRO_241_0002.fits',
                      '2011-08-28/PACMAN_OBJ_ASTRO_241_0003.fits',
                      '2011-08-28/PACMAN_OBJ_ASTRO_241_0004.fits',
                      '2011-08-28/PACMAN_OBJ_ASTRO_241_0005.fits',
                      '2011-08-28/PACMAN_OBJ_ASTRO_241_0006.fits',# A-B overflow
                      '2011-08-28/PACMAN_OBJ_ASTRO_241_0007.fits',
                      '2011-08-28/PACMAN_OBJ_ASTRO_241_0008.fits'
                      ]

files_241_HD10268 = ['2011-08-28/PACMAN_OBJ_ASTRO_241_0009.fits',
                     '2011-08-28/PACMAN_OBJ_ASTRO_241_0010.fits',
                     '2011-08-28/PACMAN_OBJ_ASTRO_241_0011.fits',# A-B overflow
                     '2011-08-28/PACMAN_OBJ_ASTRO_241_0012.fits',
                     '2011-08-28/PACMAN_OBJ_ASTRO_241_0013.fits',
                     '2011-08-28/PACMAN_OBJ_ASTRO_241_0014.fits',
                     '2011-08-28/PACMAN_OBJ_ASTRO_241_0015.fits',
                     ]

def doAll(files, directory=data_directory, reduceData=False, bootstrap=False,
          selection=None, maxResiduals=None, firstGuess=[10.,90., 0.0],
          fittedParam=[1,1,1], res2fits=False):
    """
    astrometric fit
    """
    # -- data reduction
    if reduceData:
        for f in files:
            print '#'*5, f, '#'*12
            a = prima.drs(data_directory+f)
            a.astrometryFtk(max_err_um=3, max_GD_um=5.0, sigma_clipping=4.0,
                            writeOut=True, overwrite=True, max_length_s=1.0)
            a.raw.close()
            del a
            # correct a bug in this particular file
            if f == '2011-08-23/PACMAN_OBJ_ASTRO_236_0007.fits':
                tmp=pyfits.open(data_directory+\
                               '/2011-08-23/PACMAN_OBJ_ASTRO_236_0007_RED.fits',
                                mode='update')
                tmp[0].header.update('HIERARCH ESO INS MODE', 'SWAPPED')
                tmp.flush()
                tmp.close()

    # -- data fitting
    if selection is None:
        selection = range(len(files))
    if bootstrap:
        files = [files[k] for k in selection]
        t0 = time.time()
        res = astrom.bootstrapListOfFiles(files, data_directory, N=500,
                                          fit_param  =fittedParam, plot=True,
                                          first_guess=firstGuess,
                                          maxResidualsMas=maxResiduals,
                                          multi=True)
        print 'bootstraping performed in', round(time.time()-t0, 1), 's'
    else:
        astrom.interpListOfFiles(files, data_directory,
                                 plot=True, quiet=False,
                                 fit_param  = fittedParam,
                                 first_guess= firstGuess,
                                 fit_only_files=selection,
                                 maxResiduals=maxResiduals, res2fits=res2fits)
        res = []
    return

def compare238_241_HD202730(reduceData=False, bootstrap=False):
    global res241, res238
    if bootstrap or res241 is None or res238 is None:
        print 'bootstrapping day 241...'
        res241 = analyse241_HD202730(bootstrap=True, reduceData=reduceData)
        print 'bootstrapping day 238...'
        res238 = analyse238_HD202730(bootstrap=True, reduceData=reduceData)
    pylab.figure(4)
    pylab.clf()

    pylab.plot([x['param'][0]*np.sin(x['param'][1]*np.pi/180) for x in res238],
               [x['param'][0]*np.cos(x['param'][1]*np.pi/180) for x in res238],
               'or', alpha=0.5, label='238')
    pylab.plot([x['param'][0]*np.sin(x['param'][1]*np.pi/180) for x in res241],
               [x['param'][0]*np.cos(x['param'][1]*np.pi/180) for x in res241],
               'og', alpha=0.5, label='241')
    pylab.legend()
    pylab.xlabel('$\Delta$ RA (\")')
    pylab.ylabel('$\Delta$ dec (\")')
    pylab.axes().set_aspect('equal', 'datalim')
    return

def primetFsmBias(x, param, quiet=True):
    """
    primet model as a function of FSM position:
    x = [MJD, FSM1X, FSM1Y, FSM2X, FSM2Y]
    param = [offset, amjd, a1x, a1y, ...]

    primet = offset + amjd*MJD + a1x*FSM1X + a1y*FSM1Y + ...
    """
    res = param[0] + (np.array(x)*np.array(param)[1:,np.newaxis]).sum(axis=0)
    return res

def interpByStep(x, x0, y0):
    """
    for each x returns the value y0, for the closest x0, with x0<x

    FIXME: this is WAY too slow!!!!!!!!!!!!
    """
    res = []
    for xx in x:
        w = np.where(xx<=x0)
        res.append(np.array(y0)[w[0][np.abs(xx-np.array(x0)[w[0]]).argmin()]])
    return res

def FSM_full_test(model=False, AT='AT4'):
    """
    """
    directory = '/Volumes/DATA500/PRIMA/COMM16/'
    ### AT4
    if AT=='AT4' or AT==4:
        AT = 'AT4'; DT = .6/(24*3600)
        #a = prima.drs(directory+'2011-08-29/PACMAN_OBJ_ASTRO_242_0014.fits')
        #p = prima.pssRecorder(directory+'PSSRECORDER/pssguiRecorder_lat4vcm_2011-08-30_09-09-54.dat')
        #f = open(directory+'PIEZOSCAN/pscsosfPiezoScan2_lat4fsm_2011-08-30T09_10_46.txt')
        a = prima.drs(directory+'2011-08-29/PACMAN_OBJ_ASTRO_242_0018.fits')
        p = prima.pssRecorder(directory+'PSSRECORDER/pssguiRecorder_lat4vcm_2011-08-30_09-49-23.dat')
        f = open(directory+'PIEZOSCAN/pscsosfPiezoScan2_lat4fsm_2011-08-30T09_50_11.txt')
        #a = prima.drs(directory+'2011-08-29/PACMAN_OBJ_ASTRO_242_0021.fits')
        #p = prima.pssRecorder(directory+'PSSRECORDER/pssguiRecorder_lat4vcm_2011-08-30_10-24-43.dat')
        #f = open(directory+'PIEZOSCAN/pscsosfPiezoScan2_lat4fsm_2011-08-30T10_25_14.txt')
        #a = prima.drs(directory+'2011-08-29/PACMAN_OBJ_ASTRO_242_0023.fits')
        #p = prima.pssRecorder(directory+'PSSRECORDER/pssguiRecorder_lat4vcm_2011-08-30_10-35-52.dat')
        #f = open(directory+'PIEZOSCAN/pscsosfPiezoScan2_lat4fsm_2011-08-30T10_36_31.txt')
    else:
        ### AT3
        AT = 'AT3'; DT=0.75/(24.*3600)
        #a = prima.drs(directory+'2011-08-29/PACMAN_OBJ_ASTRO_242_0016.fits')
        #p = prima.pssRecorder(directory+'PSSRECORDER/pssguiRecorder_lat3vcm_2011-08-30_09-27-37.dat')
        #f = open(directory+'PIEZOSCAN/pscsosfPiezoScan2_lat3fsm_2011-08-30T09_28_22.txt')
        #a = prima.drs(directory+'2011-08-29/PACMAN_OBJ_ASTRO_242_0020.fits')
        #p = prima.pssRecorder(directory+'PSSRECORDER/pssguiRecorder_lat3vcm_2011-08-30_10-09-40.dat')
        #f = open(directory+'PIEZOSCAN/pscsosfPiezoScan2_lat3fsm_2011-08-30T10_10_09.txt')
        #a = prima.drs(directory+'2011-08-29/PACMAN_OBJ_ASTRO_242_0024.fits')
        #p = prima.pssRecorder(directory+'PSSRECORDER/pssguiRecorder_lat3vcm_2011-08-30_10-41-53.dat')
        #f = open(directory+'PIEZOSCAN/pscsosfPiezoScan2_lat3fsm_2011-08-30T10_43_04.txt')
        #a = prima.drs(directory+'2011-08-29/PACMAN_OBJ_ASTRO_242_0026.fits')
        #p = prima.pssRecorder(directory+'PSSRECORDER/pssguiRecorder_lat3vcm_2011-08-30_10-53-32.dat')
        #f = open(directory+'PIEZOSCAN/pscsosfPiezoScan2_lat3fsm_2011-08-30T10_54_20.txt')
        a = prima.drs(directory+'2011-08-29/PACMAN_OBJ_ASTRO_242_0027.fits')
        p = prima.pssRecorder(directory+'PSSRECORDER/pssguiRecorder_lat3vcm_2011-08-30_11-00-29.dat')
        f = open(directory+'PIEZOSCAN/pscsosfPiezoScan2_lat3fsm_2011-08-30T11_00_58.txt')

    lines = f.readlines()
    f.close()
    lines = filter(lambda x: not '#' in x and len(x)>10, lines)
    mjd =  [astro.tag2mjd('2011-08-30T'+x.split()[1])+DT for x in lines]
    xmjd = np.linspace(min(mjd), max(mjd), 200)

    FSM1X = [float(x.split()[2]) for x in lines]
    FSM1Y = [float(x.split()[3]) for x in lines]
    FSM2X = [float(x.split()[4]) for x in lines]
    FSM2Y = [float(x.split()[5]) for x in lines]

    print 'PCR START:', a.raw[0].header['ESO PCR ACQ START']
    print 'PCR END  :', a.raw[0].header['ESO PCR ACQ END']

    min_mjd = min(mjd)
    max_mjd = max(mjd)
    if model:
        param = np.array([0,0,.1,.1,.1,.1])
        mjd_primet =  1e-6*a.raw['METROLOGY_DATA'].data.field('TIME')/\
                   (24*3600)+astro.tag2mjd(a.raw[0].header['ESO PCR ACQ START'])
        primet =  (a.raw['METROLOGY_DATA'].data.field('DELTAL')-
                   a.raw['METROLOGY_DATA'].data.field('DELTAL').mean())*1e6
        print 'w_fit:'
        w_fit = np.where((mjd_primet>min_mjd)*(mjd_primet<max_mjd))
        #w_fit = np.where((mjd_primet>min_mjd))
        w_fit = (w_fit[0][::200],)
        print 'x_fit:'
        X_fit = [mjd_primet[w_fit]-mjd_primet[w_fit].mean(),
                 interpByStep(mjd_primet[w_fit], mjd, np.array(FSM1X)-FSM1X[0]),
                 interpByStep(mjd_primet[w_fit], mjd, np.array(FSM1Y)-FSM1Y[0]),
                 interpByStep(mjd_primet[w_fit], mjd, np.array(FSM2X)-FSM2X[0]),
                 interpByStep(mjd_primet[w_fit], mjd, np.array(FSM2Y)-FSM2Y[0])]
        Y_fit = primet[w_fit]
        print 'fit:'
        fit = myfit.fit(primetFsmBias, X_fit, param, Y_fit)
        fit.leastsqfit()
        print 'dOPD_um/(FSM1X_um - %5.3f) = %6.4f' %\
              (FSM1X[0], fit.leastsq_best_param[2])
        print 'dOPD_um/(FSM1Y_um - %5.3f) = %6.4f' %\
              (FSM1Y[0], fit.leastsq_best_param[3])
        print 'dOPD_um/(FSM2X_um - %5.3f) = %6.4f' %\
              (FSM2X[0], fit.leastsq_best_param[4])
        print 'dOPD_um/(FSM2Y_um - %5.3f) = %6.4f' %\
              (FSM2Y[0], fit.leastsq_best_param[5])

        pylab.figure(4, figsize=(17,3))
        pylab.clf()
        pylab.subplots_adjust(left=0.06, bottom=0.15, right=0.96, top=0.85,
                              wspace=0.15, hspace=0.01)
        pylab.title(AT+' FSM test:'+a.filename)
        pylab.plot(mjd_primet,  primet, 'b-', label='PRIMET A-B')
        pylab.plot(mjd_primet[w_fit[0]],
                   primetFsmBias(X_fit,fit.leastsq_best_param),
                   'r-', alpha=0.5, linewidth=3, label='linear model FSM')
        pylab.plot(mjd_primet[w_fit[0]],
                   Y_fit-primetFsmBias(X_fit,fit.leastsq_best_param),
                   '-', color='g', alpha=0.5, linewidth=3, label='residuals')
        pylab.hlines([0], min(mjd), max(mjd), color='k', linestyle='dashed',
                     linewidth=2)
        pylab.legend(ncol=3, loc=('upper left' if AT=='AT4' else 'upper right'))
        pylab.xlim(min_mjd, max_mjd)
        pylab.ylabel('PRIMET A-B ($\mu$m)')

    pylab.figure(3, figsize=(17,9))
    pylab.subplots_adjust(left=0.06, bottom=0.07, right=0.96, top=0.96,
                          wspace=0.15, hspace=0.01)
    pylab.clf()

    ax1 = pylab.subplot(5,2,1)
    pylab.title(AT+' FSM test:'+a.filename)
    pylab.plot(1e-6*a.raw['METROLOGY_DATA'].data.field('TIME')/(24*3600)+
               astro.tag2mjd(a.raw[0].header['ESO PCR ACQ START']),
               (a.raw['METROLOGY_DATA'].data.field('DELTAL')-
                a.raw['METROLOGY_DATA'].data.field('DELTAL').mean())*1e6,
               'b-')
    pylab.ylabel('PRIMET A-B ($\mu$m)')

    pylab.subplot(5,2,3, sharex=ax1)
    #pylab.plot(mjd, FSM1X, '-k', markersize=8, linestyle='steps')
    pylab.plot(xmjd, interpByStep(xmjd, mjd, FSM1X), 'k-')
    pylab.ylabel('FSM1 X ($\mu$m)')

    pylab.subplot(5,2,5, sharex=ax1)
    #pylab.plot(mjd, FSM1Y, '-k', markersize=8, linestyle='steps')
    pylab.plot(xmjd, interpByStep(xmjd, mjd, FSM1Y), 'k-')
    pylab.ylabel('FSM1 Y ($\mu$m)')

    pylab.subplot(5,2,7, sharex=ax1)
    #pylab.plot(mjd, FSM2X, '-k', markersize=8, linestyle='steps')
    pylab.plot(xmjd, interpByStep(xmjd, mjd, FSM2X), 'k-')
    pylab.ylabel('FSM2 X ($\mu$m)')

    pylab.subplot(5,2,9, sharex=ax1)
    #pylab.plot(mjd, FSM2Y, '-k', markersize=8, linestyle='steps')
    pylab.plot(xmjd, interpByStep(xmjd, mjd, FSM2Y), 'k-')
    pylab.ylabel('FSM2 Y ($\mu$m)')
    pylab.xlabel('MJD')

    pylab.subplot(5,2,2, sharex = ax1, sharey= ax1)
    pylab.title(AT+' FSM test:'+a.filename)
    pylab.plot(1e-6*a.raw['METROLOGY_DATA'].data.field('TIME')/(24*3600)+
               astro.tag2mjd(a.raw[0].header['ESO PCR ACQ START']),
               (a.raw['METROLOGY_DATA'].data.field('DELTAL')-
                a.raw['METROLOGY_DATA'].data.field('DELTAL').mean())*1e6,
               'b-')

    pylab.subplot(5,2,4, sharex=ax1)
    pylab.plot(p.mjd, p.data['VCM1X[um]'], '-k')
    pylab.ylabel('VCM1 X ($\mu$m)')

    pylab.subplot(5,2,6, sharex=ax1)
    pylab.plot(p.mjd, p.data['VCM1Y[um]'], '-k')
    pylab.ylabel('VCM1 Y ($\mu$m)')

    pylab.subplot(5,2,8, sharex=ax1)
    pylab.plot(p.mjd, p.data['VCM2X[um]'], '-k')
    pylab.ylabel('VCM2 X ($\mu$m)')

    pylab.subplot(5,2,10, sharex=ax1)
    pylab.plot(p.mjd, p.data['VCM2Y[um]'], '-k')
    pylab.ylabel('VCM2 Y ($\mu$m)')
    pylab.xlim(min_mjd, max_mjd)
    del a

    return

########################### OBSOLETE ######################################

def analyse236_HD202730(reduceData=False, bootstrap=False):
    """
    astrometric fit
    """
    files = files_236
    if reduceData:
        for f in files:
            print '#'*5, f, '#'*12
            a = prima.drs(data_directory+f)
            a.astrometryFtk(max_err_um=5, max_GD_um=3.0, sigma_clipping=3.0,
                            writeOut=True, overwrite=True, max_length_s=1.0)
            del a
            if f == '2011-08-23/PACMAN_OBJ_ASTRO_236_0007.fits':
                tmp=pyfits.open(data_directory+\
                               '/2011-08-23/PACMAN_OBJ_ASTRO_236_0007_RED.fits',
                               mode='update')
                tmp[0].header.update('HIERARCH ESO INS MODE', 'SWAPPED')
                tmp.flush()
                tmp.close()

    selection = range(len(files))
    #selection = [0,1,4,5,8,9]
    if bootstrap:
        files = [files[k] for k in selection]
        astrom.bootstrapListOfFiles(files, data_directory, N=500,
                                    fit_param  =[1,    1,    1], plot=True,
                                    first_guess=[6.0, 200.0,-3.92104],
                                    maxResidualsMas=20, multi=True)
    else:
        astrom.interpListOfFiles(files, data_directory,
                                 plot=True,
                                 fit_param  =[1,    1,    1],
                                 first_guess=[6.0, 200.0,1.0],
                                 fit_only_files=selection, maxResiduals=20 )
    return

def analyse238_HD202730(reduceData=False, bootstrap=False):
    """
    astrometric fit
    """
    files = files_238
    if reduceData:
        for f in files:
            print '#'*5, f, '#'*12
            a = prima.drs(data_directory+f)
            a.astrometryFtk(max_err_um=5, max_GD_um=3.0, sigma_clipping=4.0,
                            writeOut=True, overwrite=True, max_length_s=1.0)
            del a

    selection = range(len(files))
    #selection = [0,2,3,4,5,6,7]
    if bootstrap:
        files = [files[k] for k in selection]
        res = astrom.bootstrapListOfFiles(files, data_directory, N=500,
                                    fit_param  =[1,    1,    1], plot=True,
                                    first_guess=[6.0, 200.0,-3.92104],
                                          maxResidualsMas=30)
    else:
        astrom.interpListOfFiles(files, data_directory,
                                 plot=True,
                                 fit_param  =[1,    1,    1],
                                 first_guess=[6.0, 200.0,1.0],
                                 fit_only_files=selection,maxResiduals=20)
    return res

def analyse238_HD10360(reduceData=False, bootstrap=False):
    """
    astrometric fit
    """
    files = files_238_10360
    if reduceData:
        for f in files:
            a = prima.drs(data_directory+f)
            print '#'*5, f, '#'*12
            a.astrometryFtk(max_err_um=2.0, max_GD_um=3.0, sigma_clipping=4.0,
                            writeOut=True, overwrite=True, max_length_s=1.0)
            del a
            a=[]

    selection = range(len(files))
    #selection = [0,1,2,3,5]
    #selection = [1,3,5]
    if bootstrap:
        files = [files[k] for k in selection]
        astrom.bootstrapListOfFiles(files, data_directory, N=500,
                                    fit_param  =[1,    1,    1], plot=True,
                                    first_guess=[6.0, 200.0,-3.92104])
    else:
        res = astrom.interpListOfFiles(files, data_directory,
                                       plot=True,quiet=False,
                                       fit_param  =[1,    1,    1],
                                       first_guess=[6.0, 200.0,-3.92104],
                                       fit_only_files=selection,
                                       maxResiduals=5)
    return

def analyse241_HD202730(reduceData=False, bootstrap=False):
    """
    astrometric fit
    """
    files = files_241_HD202730
    if reduceData:
        for f in files:
            print '#'*5, f, '#'*12
            a = prima.drs(data_directory+f)
            a.astrometryFtk(max_err_um=5, max_GD_um=3.0, sigma_clipping=4.0,
                            writeOut=True, overwrite=True, max_length_s=1.0)
            del a

    selection = range(len(files))
    selection = [0,1,2,3,5,6]
    if bootstrap:
        files = [files[k] for k in selection]
        res = astrom.bootstrapListOfFiles(files, data_directory, N=500,
                                    fit_param  =[1,    1,    1], plot=True,
                                    first_guess=[6.0, 200.0,-3.92104],
                                          maxResidualsMas=30)

    else:
        astrom.interpListOfFiles(files, data_directory,
                                 plot=True,
                                 fit_param  =[1,    1,    1],
                                 first_guess=[6.0, 200.0,1.0],
                                 fit_only_files=selection,maxResiduals=20)
        res = []
    return res

def analyse241_HD10268(reduceData=False, bootstrap=False):
    """
    astrometric fit
    """
    files = files_241_HD10268
    if reduceData:
        for f in files:
            print '#'*5, f, '#'*12
            a = prima.drs(data_directory+f)
            a.astrometryFtk(max_err_um=5, max_GD_um=3.0, sigma_clipping=4.0,
                            writeOut=True, overwrite=True, max_length_s=1.0)
            del a
    selection = range(len(files))
    selection=[0,1,2,4,5,6]
    if bootstrap:
        files = [files[k] for k in selection]
        res = astrom.bootstrapListOfFiles(files, data_directory, N=500,
                                          fit_param  =[1,    1,    1],
                                          plot=True,
                                          first_guess=[6.0, 200.0,-3.92104],
                                          maxResidualsMas=30, multi=True)
    else:
        astrom.interpListOfFiles(files, data_directory,
                                 plot=True,
                                 fit_param  =[1,    1,    1],
                                 first_guess=[6.0, 200.0,1.0],
                                 fit_only_files=selection,maxResiduals=20)
        res = []
    return res
