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
astro_files034 =['PACMAN_OBJ_ASTRO_034_0001.fits', 
                 'PACMAN_OBJ_ASTRO_034_0002.fits', 
                 'PACMAN_OBJ_ASTRO_034_0003.fits', 
                 'PACMAN_OBJ_ASTRO_034_0004.fits', 
                 'PACMAN_OBJ_ASTRO_034_0005.fits', 
                 'PACMAN_OBJ_ASTRO_034_0006.fits', 
                 'PACMAN_OBJ_ASTRO_034_0007.fits',
                 'PACMAN_OBJ_ASTRO_034_0008.fits']
# external HD:
data_directory034 = '/Volumes/DATA/PRIMA/COMM14/2011-02-02/'

scan_files030 =['PACMAN_OBJ_SCAN_030_0015.fits',  
                'PACMAN_OBJ_SCAN_030_0018.fits',  
                'PACMAN_OBJ_SCAN_030_0020.fits',  
                'PACMAN_OBJ_SCAN_030_0024.fits',  
                'PACMAN_OBJ_SCAN_030_0026.fits',  
                'PACMAN_OBJ_SCAN_030_0029.fits',  
                'PACMAN_OBJ_SCAN_030_0031.fits',  
                'PACMAN_OBJ_SCAN_030_0033.fits',  
                'PACMAN_OBJ_SCAN_030_0035.fits']                 
astro_files030 =['PACMAN_OBJ_ASTRO_030_0004.fits', 
                 'PACMAN_OBJ_ASTRO_030_0005.fits', 
                 'PACMAN_OBJ_ASTRO_030_0006.fits', 
                 'PACMAN_OBJ_ASTRO_030_0007.fits', 
                 'PACMAN_OBJ_ASTRO_030_0008.fits', 
                 'PACMAN_OBJ_ASTRO_030_0009.fits', 
                 'PACMAN_OBJ_ASTRO_030_0010.fits',
                 'PACMAN_OBJ_ASTRO_030_0011.fits', 
                 'PACMAN_OBJ_ASTRO_030_0012.fits',
                 'PACMAN_OBJ_ASTRO_030_0013.fits']
all_files030 = ['PACMAN_OBJ_SCAN_030_0015.fits',  
                'PACMAN_OBJ_SCAN_030_0018.fits',  
                'PACMAN_OBJ_SCAN_030_0020.fits',  
                'PACMAN_OBJ_SCAN_030_0024.fits',  
                'PACMAN_OBJ_SCAN_030_0026.fits',  
                'PACMAN_OBJ_SCAN_030_0029.fits',  
                'PACMAN_OBJ_SCAN_030_0031.fits',  
                'PACMAN_OBJ_SCAN_030_0033.fits',  
                'PACMAN_OBJ_SCAN_030_0035.fits',
                'PACMAN_OBJ_ASTRO_030_0004.fits',
                'PACMAN_OBJ_ASTRO_030_0005.fits', 
                'PACMAN_OBJ_ASTRO_030_0006.fits', 
                'PACMAN_OBJ_ASTRO_030_0007.fits', 
                'PACMAN_OBJ_ASTRO_030_0008.fits', 
                'PACMAN_OBJ_ASTRO_030_0009.fits', 
                'PACMAN_OBJ_ASTRO_030_0010.fits',
                'PACMAN_OBJ_ASTRO_030_0011.fits', 
                'PACMAN_OBJ_ASTRO_030_0012.fits',
                'PACMAN_OBJ_ASTRO_030_0013.fits']

env_file030 = '/Volumes/DATA/PRIMA/COMM14/ENV/PACMA.2011-01-29T12:00:00.000.fits'

# external HD, raw DATA:
data_directory030 = '/Volumes/DATA/PRIMA/COMM14/DAY030/'
# internal HD, local copy of reduced files:
red_directory030 = '/Users/amerand/Codes/PRIMA/REDUCED'

def analyse030(onlyAstro = True, T0=290, P0=734):
    """
    astrometric fit of the data of day 030 of 2011.

    run 'reduce030' first
    """
    global all_files030, red_directory030, env_file030, astro_files030

    if onlyAstro:
        files = astro_files030
        selection = range(len(files)) # all files
        #selection=[0,1,3,5,7,9] # ALL SWAPPED
        #selection=[2,4,6,8]     # ALL NORMAL    
    else:
        files = all_files030
        #selection = range(len(files))              # ALL FILES
        #selection = [0,1,2,3,4,5,6,7,8]            # SCAN
        #selection = [9,10,11,12,13,14,15,16,17,18] # ASTRO
        selection = [1,3,5,7,9,10,11,12,13,14,15,16,17,18] # HIGH SNR
        # --- SWAPPED ---------------------------------------
        #selection=[0,2,4,6,8,9,10,12,14,16,18] # ALL SWAPPED
        #selection=[9,10,12,14,16,18]           # ASTRO SWAPPED
        #selection=[0,2,4,6,8]                  # SCAN SWAPPED
        # --- NORMAL ----------------------------------------
        #selection=[1,3,5,7,11,13,15,17]        # ALL NORMAL
        #selection=[11,13,15,17]                # ASTRO NORMAL
        #selection=[1,3,5,7]                    # SCAN NORMAL
        
    astrom.interpListOfFiles(files, data_directory030,
                             plot=True,
                             fit_param  =[1, 1,   1,    0,1,1],
                             first_guess=[35,40.0,14.0, 0,0,0],
                             fit_only_files=selection
                             )
    return

def reduce030(demo=False, max_GD_um=1.0, max_err_um=0.35, max_length_s=1.0,
              sigma_clipping=4.0, onlyScans=False, min_scan_snr=3.0):
    global astro_files030, data_directory030, scan_files030
    if not onlyScans:
        for f in astro_files030:        
            print '### reducing ', os.path.join(data_directory030,f) , '######' 
            # typical reduction
            a = prima.drs(os.path.join(data_directory030,f))
            a.astrometryFtk(writeOut=(not demo), overwrite=True,
                            max_length_s=max_length_s,plot=demo,
                            max_GD_um=max_GD_um, max_err_um=max_err_um,
                            sigma_clipping=sigma_clipping)
            if demo:
                break
            a.raw.close()

    for f in scan_files030:        
        print '### reducing ', os.path.join(data_directory030,f) , '########' 
        # typical reduction
        a = prima.drs(os.path.join(data_directory030,f))
        a.astrometryScan(writeOut=(not demo), overwrite=True,
                         plot=demo, min_snr=min_scan_snr,
                         sigma_clipping=sigma_clipping)
        if demo:
            break
        a.raw.close()
    print "#########################################################"
    print " remember to copy reduced files to internal hard drive !"
    print "#########################################################"
    return

def reduce034(demo=False, max_GD_um=1.0, max_err_um=0.3, max_length_s=1.0,
              sigma_clipping=5.0):
    global astro_files034, data_directory034
    for f in astro_files034:        
        if f.split('_')[2] == 'ASTRO':            
            print '### reducing ', f , '###################################' 
            # typical reduction
            a = prima.drs(os.path.join(data_directory034,f))
            a.astrometry(writeOut=(not demo), overwrite=True,
                         max_length_s=max_length_s,plot=demo,
                         max_GD_um=max_GD_um, max_err_um=max_err_um,
                         sigma_clipping=sigma_clipping)
            if demo:
                break
            a.raw.close()
    return

def analyseDark(sub_dir):
    """
    sub routine for 'checkIfDarksAreContaminated'    
    """
    global data_directory
    files = os.listdir(data_directory+sub_dir)
    
    darksA = []
    darksB = []
    flatsA  = []
    flatsB  = []
    mjd_obs = []
    ditA = []
    ditB = []
    for f in files:
        if f[:16]=='PACMAN_OBJ_ASTRO':
            print f
            a = prima.drs(data_directory+sub_dir+f)
            try:
                darksA.append(a.fsu_calib[('FSUA', 'DARK')])
                darksB.append(a.fsu_calib[('FSUB', 'DARK')])
                flatsA.append(a.fsu_calib[('FSUA', 'FLAT')])
                flatsB.append(a.fsu_calib[('FSUB', 'FLAT')])
                mjd_obs.append(a.getKeyword('DATE-OBS'))
                ditA.append(a.getKeyword('ISS PRI FSU1 DIT'))
                ditB.append(a.getKeyword('ISS PRI FSU2 DIT'))
            except:
                print 'no calibs?'
            a.__del__()
    #--
    darksA = numpy.array(darksA)
    darksB = numpy.array(darksB)
    flatsA = numpy.array(flatsA)
    flatsB = numpy.array(flatsB)
    ditA = numpy.array(ditA)
    ditB = numpy.array(ditB)
    # end analyseDark
    return mjd_obs, darksA, darksB, flatsA, flatsB, ditA, ditB

def checkIfDarksAreContaminated():
    """
    check if darks are contaminated by object's flux. see subroutine
    for 'analyseDark'
    """
    subdir = 'DAY027/'
    mjd_obs, darksA, darksB, flatsA, flatsB, ditA,\
             ditB = analyseDark(subdir)
    
    pyplot.figure(0, figsize=(12,10))
    pyplot.clf()
    w = numpy.where(ditA<=0.001)
    wl = ['W','1','2','3','4','5'] 
    for k in range(4):
        for l in range(6):
            pyplot.subplot(6,2,1+2*l)
            pyplot.ylabel('dark ['+wl[l]+']')
            if l==0:
                pyplot.title(subdir+' FSUA')
            elif l==5:
                pyplot.xlabel('flat-dark')
            pyplot.plot(flatsA[w[0],l,k]-darksA[w[0],l,k],\
                        darksA[w[0],l,k], 'o')
            coef = scipy.polyfit(flatsA[w[0],l,k]-darksA[w[0],l,k],\
                                 darksA[w[0],l,k], 1)
            pyplot.plot(flatsA[w[0],l,k]- darksA[w[0],l,k],\
                        scipy.polyval(coef, flatsA[w[0],l,k]-
                                      darksA[w[0],l,k]))
            pyplot.subplot(6,2,2+2*l)
            if l==0:
                 pyplot.title(subdir+' FSUB')
            elif l==5:
                 pyplot.xlabel('flat-dark')
            pyplot.plot(flatsB[w[0],l,k]-darksB[w[0],l,k],\
                        darksB[w[0],l,k], 'o')
            coef = scipy.polyfit(flatsB[w[0],l,k]-darksB[w[0],l,k],\
                                 darksB[w[0],l,k], 1)
            pyplot.plot(flatsB[w[0],l,k]-darksB[w[0],l,k],\
                        scipy.polyval(coef, flatsB[w[0],l,k]-
                                      darksB[w[0],l,k]))

def testTemperature():
    ind = ['3','4','5','6','7','8']
    #ind = ['3']
    d = '/Volumes/DATA/PRIMA/COMM14/'
    files = '2011-02-02/PACMAN_OBJ_ASTRO_034_000'
    mjd_all = []
    mfsub_all = []
    for i in ind:
        f = pyfits.open(d+files+i+'.fits')
        mjd0 = f[0].header['MJD-OBS']
        print d+files+i+'.fits', mjd0
        mjd = mjd0+f['METROLOGY_DATA_FSUB'].data.field('TIME')/(1e6*3600*24)
        m = f['METROLOGY_DATA_FSUB'].data.field('DELTAL')
        mjd_all.extend(list(mjd))
        mfsub_all.extend(list(m))
        f.close()
    f=pyplot.figure(1, figsize=(8,5))
    f.subplots_adjust(hspace=0.01, top=0.97, left=0.1,
                      right=0.98, bottom=0.15)
    pyplot.clf()
    pyplot.plot(numpy.array(mjd_all)[::100]-numpy.array(mjd_all).min(),
                (mfsub_all[::100]-numpy.array(mfsub_all).mean())*1e6,
                'k-')
    pyplot.ylim(-15,15)
    pyplot.ylabel(r'PRIMET -B ($\mu$m)')
    pyplot.xlabel('MJD')

    pyplot.figure(2, figsize=(8,10))
    pyplot.clf()
    # toy model
    e = prima.env(d+'ENV/PACMA.2011-01-29T12:00:00.000.fits')
    mjd = numpy.linspace(55591.1, 55591.145, 100)

    # for AT3-G2-DL2(E)
    # approximative distances
    #     in duct,  to DL, to cart, back, to FSU 
    dist_G2 = numpy.array([13., 19., 50, 50, 20])*2
    
    T_G2   = [e.interpVarMJD('VLTItempSens6')(mjd)+272,
              e.interpVarMJD('VLTI_HSS_TEMP2')(mjd)+272,
              e.interpVarMJD('VLTI_HSS_TEMP4')(mjd)+272,
              e.interpVarMJD('VLTI_HSS_TEMP2')(mjd)+272,
              e.interpVarMJD('VLTItempSens5')(mjd)+272]
    opl_G2 = prima.n_air_P_T(1.0, T=T_G2)*\
             numpy.array(dist_G2)[:,numpy.newaxis]
    opl_G2 = opl_G2.sum(axis=0)
    
    # for AT4-J2-DL4(W)
    dist_J2 = numpy.array([57., 45, 35, 35, 20])*2
    T_J2   = [e.interpVarMJD('VLTItempSens16')(mjd)+272,
              e.interpVarMJD('VLTI_HSS_TEMP2')(mjd)+272, 
              e.interpVarMJD('VLTI_HSS_TEMP1')(mjd)+272,
              e.interpVarMJD('VLTI_HSS_TEMP2')(mjd)+272,
              e.interpVarMJD('VLTItempSens5')(mjd)+272]
    opl_J2 = prima.n_air_P_T(1.0, T=T_J2)*\
             numpy.array(dist_J2)[:,numpy.newaxis]
    opl_J2 = opl_J2.sum(axis=0)
    pyplot.subplot(211)
    #  pyplot.plot(mjd-mjd.min(), (opl_G2-opl_G2.mean())*1.e6, 'or')
    pyplot.plot(mjd-mjd.min(), (opl_J2-opl_J2.mean())*1.e6, 'ob')
    pyplot.ylabel(r'$\delta$ OPL ($\mu$m)')
    pyplot.ylim(-15,15)
    
    pyplot.subplot(212)
    for k in range(5):
        #pyplot.plot(mjd-mjd.min(), T_G2[k]-T_G2[k].mean(), 'r-')
        pyplot.plot(mjd-mjd.min(), T_J2[k]-T_J2[k].mean(), 'b-')
    pyplot.ylabel(r'$\delta$ T (K)')
    pyplot.xlabel('time (d)')
    
    return

