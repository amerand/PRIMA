import os
import numpy as np
from scipy.optimize import leastsq
import pyfits
import time
from matplotlib import pyplot

import prima # low level data reduction
import astrom # astrometric data analysis
import myfit

"""
here the astrometry dataset that we agreed on using for the separation
fitting / residual analysis. 

# HD202730 is supposed to be a P=1500y, which leads to ~0.1 mas per day   
astro.EstimateApparentOrbitalSeparation(M1='A5V', M2='A7V', plx=.033, P=1500*365)

# HD10360 -> 0.25 mas per day (P~483yrs, semi-a~10.2 arcsec)

"""

data_directory = '/Volumes/DATA/PRIMA/COMM17/' # external hardrive
#data_directory = '/Users/amerand/DATA/PRIMA/' # internal hardrive
# -----------------------------------------------------------------------

files_hd10360_324 = ['2011-11-19/PACMAN_OBJ_ASTRO_324_0001.fits',
                    '2011-11-19/PACMAN_OBJ_ASTRO_324_0002.fits']
# -----------------------------------------------------------------------

files_hd66598_324 = ['2011-11-19/PACMAN_OBJ_ASTRO_324_0003.fits',
                     '2011-11-19/PACMAN_OBJ_SCAN_324_0024.fits',
                     '2011-11-19/PACMAN_OBJ_SCAN_324_0021.fits']
# -----------------------------------------------------------------------

# max_err_um=2, max_GD_um=5.
files_hd10360_325 = ['2011-11-20/PACMAN_OBJ_ASTRO_325_0001.fits',
                     '2011-11-20/PACMAN_OBJ_ASTRO_325_0002.fits',
                     '2011-11-20/PACMAN_OBJ_ASTRO_325_0003.fits',
                     '2011-11-20/PACMAN_OBJ_ASTRO_325_0004.fits',
                     '2011-11-20/PACMAN_OBJ_ASTRO_325_0005.fits',
                     '2011-11-20/PACMAN_OBJ_ASTRO_325_0006.fits', # sigma_clipping = 2.5
                     '2011-11-20/PACMAN_OBJ_ASTRO_325_0007.fits',
                     '2011-11-20/PACMAN_OBJ_ASTRO_325_0008.fits',
                     '2011-11-20/PACMAN_OBJ_ASTRO_325_0009.fits',
                     #'2011-11-20/PACMAN_OBJ_ASTRO_325_0010.fits', #BAD
                     ]
# -----------------------------------------------------------------------

# long sequence without swapping in the middle
files_hd10360_328 = ['2011-11-23/PACMAN_OBJ_ASTRO_328_00'+k+'.fits'
                     for k in ('01','02','03','04','05','06','07',
                               '08','09','10','11','13','14','15',
                               '16','17','18','19','20','21','22',
                               '24','25','26','27','28','29','30') ]
#astrom.doAll(comm17.files_hd10360_328, directory=comm17.data_directory, firstGuess=[11,187,-1.3,3.3,0], fittedParam=[1,1,1,0,1])

# -----------------------------------------------------------------------

files_hd66598_328 = ['2011-11-23/PACMAN_OBJ_ASTRO_328_00'+k+'.fits'
                     for k in ('31','32','33','34','35','36','37',
                               '38','39','40','41','42','43','44',
                               '45','46','47','48','49','50','51',
                               '53','54','55','56','57')]
#pssRec_hd66598_328 = {'lat3':prima.pssRecorder(data_directory+
#        'pssguiRecorder/pssguiRecorder_lat3fsm_2011-11-24_01-25-01.dat'),
#                      'lat4':prima.pssRecorder(data_directory+
#        'pssguiRecorder/pssguiRecorder_lat4fsm_2011-11-24_01-25-01.dat')}
#astrom.doAll(comm17.files_hd66598_328, directory=comm17.data_directory,reduceData=1, max_GD_um=5, max_err_um=2, sigma_clipping=3.5,pssRec=comm17.pssRec_hd66598_328)
#astrom.doAll(comm17.files_hd66598_328, directory=comm17.data_directory,firstGuess=[35,134,-10,3.5,2.5], fittedParam=[1,1,1,1,1], maxResiduals=20)

# -----------------------------------------------------------------------
# 03 has no data; 16 and 21 have bad tracking 
files_hd10360_329 = ['2011-11-24/PACMAN_OBJ_ASTRO_329_00'+k+'.fits'
                     for k in ('01','02','04','05','06','07',
                               '08','09','10','11','13','14','15',
                               '17','18','19','20')]
files_hd66598_329 = ['2011-11-24/PACMAN_OBJ_ASTRO_329_00'+k+'.fits'
                     for k in ('22','24','25','26','27','28','29','30',
                               '31','32','33','34','35','36','37',
                               '38','39','40','41','42','43','44',
                               '45','46','47','48',
                               #'49',
                               '50','51')]

# -----------------------------------------------------------------------
# 05, 06, 07, 13, 14 are taken with AT3 dome vignetting half of the pupil
files_hd10360_330 = ['2011-11-25/PACMAN_OBJ_ASTRO_330_00'+k+'.fits'
                     for k in ('01','02','03','04','05','06','07',
                               '08','09','10','11','12','13','14',
                               '15','16')]

files_hd10360_330bis = ['2011-11-25/PACMAN_OBJ_ASTRO_330_00'+k+'.fits'
                        for k in ('17','18','19','20','21','22','23',
                                  '24','25','26','27','28','29','30',
                                  '31','32','33','34','35','36','37',
                                  '38','39','40','41')]

files_hd66598 = files_hd66598_328
files_hd66598.extend(files_hd66598_329)

"""
astromNew.fitListOfFiles(comm17.files_hd66598, directory=comm17.data_directory,
                         firstGuess={'M0 MJD 55889.0 55889.999':-0.009,
                                     'M0 MJD 55890.0 55890.999':-0.010, 
                                     'SEP':35, 'PA':134.8,
                                     'PUPSCALE':1, 'PUPANGLE':0
                                     })#, exportAscii='HD66598.txt')

astromNew.fitListOfFiles(comm17.files_hd66598, directory=comm17.data_directory,
                         firstGuess={'T:HD66598 M0 MJD 55889.0 55889.999':-0.009,
                                     'T:HD66598 M0 MJD 55890.0 55890.999':-0.010, 
                                     'T:HD66598 SEP':35,
                                     'T:HD66598 PA':134.8})                                     
"""

files_hd10360 = files_hd10360_325
files_hd10360.extend(files_hd10360_328)
files_hd10360.extend(files_hd10360_329)
files_hd10360.extend(files_hd10360_330)
files_hd10360.extend(files_hd10360_330bis)

"""
# astrometric fit, including bias
astromNew.fitListOfFiles(comm17.files_hd10360, directory=comm17.data_directory,
                         maxResiduals=5,
                         firstGuess={'M0 MJD 55886.0 55886.999':0.0,
                                     'M0 MJD 55889.0 55889.999':0.0,
                                     'M0 MJD 55890.0 55890.999':0.0,
                                     'M0 MJD 55891.0 55891.150':0.0,
                                     'M0 MJD 55891.15 55891.999':0.0,
                                     'SEP':11, 'PA':190, 
                                     '2ROT AMP':1.06, '2ROT PHI':247.43})

astromNew.fitListOfFiles(comm17.files_hd10360, directory=comm17.data_directory,
                         firstGuess={'T:HD10360 M0 MJD 55886.0 55886.999':0.0,
                                     'T:HD10360 M0 MJD 55889.0 55889.999':0.0,
                                     'T:HD10360 M0 MJD 55890.0 55890.999':0.0,
                                     'T:HD10360 M0 MJD 55891.0 55891.150':0.0,
                                     'T:HD10360 M0 MJD 55891.15 55891.999':0.0,
                                     'T:HD10360 SEP':11, 'T:HD10360 PA':190})
                                  



# astrometric fit, with fixed bias
astromNew.fitListOfFiles(comm17.files_hd10360, directory=comm17.data_directory,
                         maxResiduals=5,
                         firstGuess={'M0 MJD 55886.0 55886.999':0.0,
                                     'M0 MJD 55889.0 55889.999':0.0,
                                     'M0 MJD 55890.0 55890.999':0.0,
                                     'M0 MJD 55891.0 55891.150':0.0,
                                     'M0 MJD 55891.15 55891.999':0.0,
                                     'SEP':11, 'PA':90,
                                     '2ROT AMP':1.06, '2ROT PHI':47.43}, # from HD66598
                                     fitOnly=['M0 MJD 55886.0 55886.999',
                                     'M0 MJD 55889.0 55889.999',
                                     'M0 MJD 55890.0 55890.999',
                                     'M0 MJD 55891.0 55891.150',
                                     'M0 MJD 55891.15 55891.999',
                                     'SEP','PA'])

"""

files_all = list(files_hd66598)
files_all.extend(files_hd10360)
firstGuess = {'T:HD10360 M0 MJD 55886.0 55886.999':-0.003,
              'T:HD10360 M0 MJD 55889.0 55889.999':-0.0013,
              'T:HD10360 M0 MJD 55890.0 55890.999':0.0,
              'T:HD10360 M0 MJD 55891.0 55891.150':-0.003,
              'T:HD10360 M0 MJD 55891.15 55891.999':-0.019,
              'T:HD10360 SEP':11.4,
              'T:HD10360 PA':188.0,
              'T:HD66598 M0 MJD 55889.0 55889.999':-0.009,
              'T:HD66598 M0 MJD 55890.0 55890.999':-0.010, 
              'T:HD66598 SEP':35.0,
              'T:HD66598 PA':134.8,
              #'PUP4 ANG':90, 'PUP4 AMP':10.0
              }
             
"""
astromNew.fitListOfFiles(comm17.files_all, directory=comm17.data_directory, firstGuess=comm17.firstGuess)
"""

def corkScrew(t, param):
    """
    [t0, X0, Y0, dX/dt, dY/dt, P, phi, Xcos, Ycos]
    """
    X = (t-param[0])*param[3]+param[1]
    Y = (t-param[0])*param[4]+param[2]
    X += param[7]*np.cos(2*np.pi*t/param[5]+param[6])
    Y += param[8]*np.cos(2*np.pi*t/param[5]+param[9])
    return (X,Y)
    
def leastsqCorkScrew(p_fit, fit, p_fixed, obs, model=False):
    try:
        p = np.zeros(fit.size)
        p[np.where(fit)] = p_fit
        p[np.where(1-fit)] = p_fixed
    except:
        p = p_fit
    res = []
    #print list(p), obs
    for o in obs:
        if model:
            res.append(corkScrew(o[0], p))
        else:
            tmp = corkScrew(o[0], p)
            res.append(o[1]-tmp[0])
            res.append(o[2]-tmp[1])
            
    res = np.array(res)
    return np.array(res)

def HD10360():
    """
    """
    NACO_sep = 11448.3 # Damien
    #NACO_sep *=0.999
    NACO_PA  = 187.0 # Damien
    NACO_PA +=0.9
    data = [{'DEC_mas': -11327.459991981585,
             'MJD': 55799.38788483571, 'PCA': ([-0.78351315847945402,
             -0.62137519301107402],
                     [-0.78351315847945402, -0.62137519301107402]),
             'RA_mas': -1586.2834801192896, 'errs_uas': (784.90506323494469,
             56.755250740679948), 'NOTE':'ACOMM2'},
            {'DEC_mas': -11328.434517473423,
             'MJD': 55886.23025809787, 'PCA': ([-0.47704928581135631,
             -0.87887654360943956],
                     [-0.47704928581135631, -0.87887654360943956]),
             'RA_mas': -1573.9752919650998, 'errs_uas': (636.66710179387803,
             97.443812507475542) , 'NOTE':'ACOMM3'},
            {'DEC_mas': -11325.155618568057,
             'MJD': 55889.144962967606, 'PCA': ([-0.5851692703101341,
             -0.81091116966330268],
                     [-0.5851692703101341, -0.81091116966330268]),
             'RA_mas': -1572.8284251763607, 'errs_uas': (1128.5886751363144,
             95.502764113947521), 'NOTE':'ACOMM3'},
            #{'DEC_mas':NACO_sep*np.cos(NACO_PA*np.pi/180),
            # 'RA_mas':NACO_sep*np.sin(NACO_PA*np.pi/180),
            # 'MJD': 55886.2, 'errs_uas':(1000,1000),
            #    'PCA':([1,0],[0,1]), 'NOTE':'NACO'},
            #{'DEC_mas': -11321.057996449183,
            {'DEC_mas': -11322.677253993401,
             'MJD': 55891.11655027233,
             'PCA': ([-0.78812707963177309, -0.61551255580296094],
                 [-0.78812707963177309, -0.61551255580296094]),
                 'RA_mas': -1571.7671378904256,
             'errs_uas': (582.6031352344354, 38.92080197079315),'NOTE':'ACOMM3'},
                {'DEC_mas': -11325.519043937573,
                'MJD': 55891.22297829576,
                'PCA': ([-0.36045943416341675, -0.93277489048676132],
         [-0.36045943416341675, -0.93277489048676132]),
 'RA_mas': -1571.9855632046058,
 'errs_uas': (152.81098660659865, 28.057280342744438),
                    'NOTE':'ACOMM3'}    
            ]
    cork = [55800, -1586.0, -11327,-1, 1, 12, 1., 1, 1, 1]
    fit =  np.array([0,      1,    1,   1,     1,      0, 1, 1, 1,1])
    obs = [(x['MJD'], x['RA_mas'], x['DEC_mas']) for x in data]
    p_fit = np.array(cork)[np.where(np.array(fit))]
    p_fixed = np.array(cork)[np.where(1-np.array(fit))]
    
    plsq = leastsq(leastsqCorkScrew, p_fit, args=(fit,p_fixed,obs,False), epsfcn=1e-3)
    corkF = np.array(cork)
    corkF[np.where(fit)] = plsq[0]
    print corkF
    t=np.linspace(55795, 55900, 2000)
    model = leastsqCorkScrew(plsq[0], fit, p_fixed,
                             [(tau,0,0) for tau in t], model=True)
    
    # linear fit to displacement:
    f = myfit.fit(myfit.PolyN,
               [d['MJD']-55800 for d in data], [0.,1.],
               [d['RA_mas'] for d in data],
               err=[d['errs_uas'][1] for d in data])
    f.leastsqfit()
    cRA = f.leastsq_best_param
    f = myfit.fit(myfit.PolyN,
               [d['MJD']-55800 for d in data], [0.,1.],
               [d['DEC_mas'] for d in data],
               err=[d['errs_uas'][1] for d in data])
    f.leastsqfit()
    cDEC = f.leastsq_best_param
    
    pyplot.figure(10, figsize=(10,5))
    pyplot.clf()
    pyplot.axes().set_aspect('equal', 'datalim')
    th = np.linspace(0, 2*np.pi, 100)
    colors=['r', 'g', 'b', 'y', 'm', 'c']
    for k,d in enumerate(data):
        ### observation:
        pyplot.plot(d['RA_mas']/1000., d['DEC_mas']/1000., 'o'+colors[k],
                    label='MJD:%8.2f (%s)'%(d['MJD'], d['NOTE']), markersize=8)
        ### error to linear fit:
        pyplot.plot([d['RA_mas']/1000, myfit.PolyN(d['MJD']-55800, cRA)/1000],
            [d['DEC_mas']/1000, myfit.PolyN(d['MJD']-55800, cDEC)/1000],
            color=colors[k], linewidth=2, alpha=0.5, linestyle='dotted')
        ### error to corkScrew:
        #pyplot.plot([d['RA_mas']/1000, leastsqCorkScrew(plsq[0],fit,p_fixed,
        #                                                [(d['MJD'],0,0)],
        #                                                model=True)[0][0]/1000],
        #    [d['DEC_mas']/1000, leastsqCorkScrew(plsq[0],fit,p_fixed,
        #                                         [(d['MJD'],0,0)],
        #                                         model=True)[0][1]/1000],    
        #    color=colors[k], linewidth=2)
        ### error ellipse:
        pyplot.plot(d['RA_mas']/1000.+
                    d['errs_uas'][0]*d['PCA'][0][1]*np.cos(th)/1e6+
                    d['errs_uas'][1]*d['PCA'][0][0]*np.sin(th)/1e6,
                    d['DEC_mas']/1000.+
                    -d['errs_uas'][1]*d['PCA'][0][1]*np.sin(th)/1e6+
                    d['errs_uas'][0]*d['PCA'][0][0]*np.cos(th)/1e6,
                    color=colors[k], linewidth=3, alpha=0.5)
    
    #pyplot.plot([m[0]/1000. for m in model], [m[1]/1000 for m in model], '-',
    #    linewidth=3, alpha=0.5, color='0.5')
    
    pyplot.plot(myfit.PolyN(t-55800, cRA)/1000.,
                myfit.PolyN(t-55800, cDEC)/1000., 'k-',
                label='linear fit', alpha=0.3, linewidth=2, markersize=10)
    
    print 'computed linear movement: %6.2f uas/day' %\
            (np.sqrt(cRA[1]**2+cDEC[1]**2)*1000)
    
    # some best obtained astrometric results:
    #pyplot.errorbar(-1.578, -11.326, xerr=0.001, marker='o',
    #                alpha=0.5, color='0.6',
    #                label='GLAO perfo:$\pm$1.0mas\n(Meyer et al. 2011)')
    #pyplot.errorbar(-1.578, -11.3265, xerr=0.0005, marker='o',
    #                alpha=0.5, color='0.4',
    #                label='HST perfo:$\pm$0.5mas\n(McLaughlin et al. 2009)')
    
    pyplot.xlabel('$\Delta$ in E direction (arcsec)')
    pyplot.ylabel('$\Delta$ in N direction (arcsec)')
    pyplot.xlim(pyplot.xlim()[1], pyplot.xlim()[0])
    pyplot.title('HD10360 - HD10360J')
    pyplot.legend(numpoints=1, prop={'size':8})

    return
    
def pupils():
    """
    display image of the pupils taken during the commissioning run.
    """
    directory = '/Users/amerand/DATA/PRIMA/'
    dark = pyfits.open(directory+'pupils/HD10360-DARK.fits')[0].data
    files = ['pupils/HD10360-IP1-AT3-FSUA-DL2-NORMAL.fits',
             'pupils/HD10360-IP2-AT3-FSUB-DL2-NORMAL.fits',
             'pupils/HD10360-IP3-AT4-FSUA-DL4-NORMAL.fits',
             'pupils/HD10360-IP4-AT4-FSUB-DL4-NORMAL.fits']
    pyplot.figure(0, figsize=(10,10))
    pyplot.subplots_adjust(hspace=0.2, top=0.95, left=0.02,
                           wspace=0.01, bottom=0.05, right=0.98)
    pyplot.clf()
    pos = [(256.2,113.5), (215.3, 118.8),
           (243.2, 122.2), (210.4, 116.3)]
    for k,f in enumerate(files):
        a = pyfits.open(directory+f)
        pyplot.subplot(2,2,k+1)
        pyplot.title(f.split('.')[0])
        pyplot.imshow(a[0].data-dark, vmin=0, cmap='gnuplot')
        pyplot.xlim(pos[k][0]-40, pos[k][0]+40)
        pyplot.ylim(pos[k][1]-40, pos[k][1]+40)
        t = np.linspace(0,2*np.pi,100)
        pyplot.plot(20*np.cos(t)+pos[k][0],
                    20*np.sin(t)+pos[k][1], 'w-',
                    linewidth=3, alpha=0.5)
        a.close()
    files = ['pupils/HD10360-IP1-AT3-FSUA-DL2-SWAPPED.fits',
             'pupils/HD10360-IP2-AT3-FSUB-DL2-SWAPPED.fits',
             'pupils/HD10360-IP3-AT4-FSUA-DL4-SWAPPED.fits',
             'pupils/HD10360-IP4-AT4-FSUB-DL4-SWAPPED.fits']

    pyplot.figure(1, figsize=(10,10))
    pyplot.subplots_adjust(hspace=0.2, top=0.95, left=0.02,
                           wspace=0.01, bottom=0.05, right=0.98)
    pyplot.clf()
    pos = [(258.2,112.5), (216.3, 118.0),
           (245.2, 119.2), (214.4, 114.3)]
    for k,f in enumerate(files):
        a = pyfits.open(directory+f)
        pyplot.subplot(2,2,k+1)
        pyplot.title(f.split('.')[0])
        pyplot.imshow(a[0].data-dark, vmin=0, cmap='gnuplot')
        pyplot.xlim(pos[k][0]-40, pos[k][0]+40)
        pyplot.ylim(pos[k][1]-40, pos[k][1]+40)
        t = np.linspace(0,2*np.pi,100)
        pyplot.plot(20*np.cos(t)+pos[k][0],
                    20*np.sin(t)+pos[k][1], 'w-',
                    linewidth=3, alpha=0.5)
        a.close()
    
    dark = pyfits.open(directory+'pupils/HD66598-DARK.fits')[0].data
    files = ['pupils/HD66598-IP1-AT3-FSUA-DL2-SWAPPED.fits',
             'pupils/HD66598-IP2-AT3-FSUB-DL2-SWAPPED.fits',
             'pupils/HD66598-IP3-AT4-FSUA-DL4-SWAPPED.fits',
             'pupils/HD66598-IP4-AT4-FSUB-DL4-SWAPPED.fits']


    pyplot.figure(2, figsize=(10,10))
    pyplot.subplots_adjust(hspace=0.2, top=0.95, left=0.02,
                           wspace=0.01, bottom=0.05, right=0.98)
    pos = [(256.2,117.7), (214.6, 121.7),
           (248.2, 129.2), (218.5, 126.4)]    
    pyplot.clf()
    for k,f in enumerate(files):
        a = pyfits.open(directory+f)
        pyplot.subplot(2,2,k+1)
        pyplot.title(f.split('.')[0])
        pyplot.imshow(a[0].data-dark, vmin=0, cmap='gnuplot')
        pyplot.xlim(pos[k][0]-40, pos[k][0]+40)
        pyplot.ylim(pos[k][1]-40, pos[k][1]+40)
        t = np.linspace(0,2*np.pi,100)
        pyplot.plot(20*np.cos(t)+pos[k][0],
                    20*np.sin(t)+pos[k][1], 'w-',
                    linewidth=3, alpha=0.5)
        a.close()

    dark = pyfits.open(directory+'pupils/HD66598_2-DARK.fits')[0].data
    files = ['pupils/HD66598_2-IP1-AT3-FSUA-DL2-SWAPPED.fits',
             'pupils/HD66598_2-IP2-AT3-FSUB-DL2-SWAPPED.fits',
             'pupils/HD66598_2-IP3-AT4-FSUA-DL4-SWAPPED.fits',
             'pupils/HD66598_2-IP4-AT4-FSUB-DL4-SWAPPED.fits']


    pyplot.figure(3, figsize=(10,10))
    pyplot.subplots_adjust(hspace=0.2, top=0.95, left=0.02,
                           wspace=0.01, bottom=0.05, right=0.98)
    pos = [(256.2,117.7), (214.6, 121.7),
           (248.2, 129.2), (218.5, 126.4)]    
    pyplot.clf()
    for k,f in enumerate(files):
        a = pyfits.open(directory+f)
        pyplot.subplot(2,2,k+1)
        pyplot.title(f.split('.')[0])
        pyplot.imshow(a[0].data-dark, vmin=0, cmap='gnuplot')
        pyplot.xlim(pos[k][0]-40, pos[k][0]+40)
        pyplot.ylim(pos[k][1]-40, pos[k][1]+40)
        t = np.linspace(0,2*np.pi,100)
        pyplot.plot(20*np.cos(t)+pos[k][0],
                    20*np.sin(t)+pos[k][1], 'w-',
                    linewidth=3, alpha=0.5)
        a.close()
    dark = pyfits.open(directory+'pupils/HD10360_2-DARK.fits')[0].data
    files = ['pupils/HD10360_2-IP1-AT3-FSUA-DL2-NORMAL.fits',
             'pupils/HD10360_2-IP2-AT3-FSUB-DL2-NORMAL.fits',
             'pupils/HD10360_2-IP3-AT4-FSUA-DL4-NORMAL.fits',
             'pupils/HD10360_2-IP4-AT4-FSUB-DL4-NORMAL.fits']
    pyplot.figure(4, figsize=(10,10))
    pyplot.subplots_adjust(hspace=0.2, top=0.95, left=0.02,
                           wspace=0.01, bottom=0.05, right=0.98)
    pyplot.clf()
    pos = [(256.2,113.5), (215.3, 118.8),
           (243.2, 122.2), (210.4, 116.3)]
    for k,f in enumerate(files):
        a = pyfits.open(directory+f)
        pyplot.subplot(2,2,k+1)
        pyplot.title(f.split('.')[0])
        pyplot.imshow(a[0].data-dark, vmin=0, cmap='gnuplot')
        pyplot.xlim(pos[k][0]-40, pos[k][0]+40)
        pyplot.ylim(pos[k][1]-40, pos[k][1]+40)
        t = np.linspace(0,2*np.pi,100)
        pyplot.plot(20*np.cos(t)+pos[k][0],
                    20*np.sin(t)+pos[k][1], 'w-',
                    linewidth=3, alpha=0.5)
        a.close()    

def pupilsMasked():
    directory = '/Users/amerand/DATA/PRIMA/pupils/'
    pyplot.figure(0, figsize=(12,5))
    pyplot.subplots_adjust(hspace=0.01, top=0.95, left=0.05,
                           wspace=0.1, bottom=0.05, right=0.98)
    pyplot.clf()
    #pos = [(258,118), (215, 121), (245, 121), (214, 115)]
    cir = [(253,113), (211, 124), (241, 122), (208, 119)]
    for k,ip in enumerate(['1','2','3','4']):
        a = pyfits.open(directory+'HD2261_2011-11-25_IP'+ip+'_nominal.fits')
        pyplot.subplot(2,4,k+1)
        pyplot.imshow(a[0].data-np.median(a[0].data),
                      vmin=0, cmap='gnuplot')
        pyplot.xlim(cir[k][0]-40, cir[k][0]+40)
        pyplot.ylim(cir[k][1]-40, cir[k][1]+40)
        t = np.linspace(0,2*np.pi,100)
        pyplot.plot(30*np.cos(t)+cir[k][0],
                    30*np.sin(t)+cir[k][1], 'w-',
                    linewidth=3, alpha=0.5)
        a.close()
        
        a = pyfits.open(directory+'HD2261_2011-11-25_IP'+ip+'_masked.fits')
        pyplot.subplot(2,4,k+5)
        pyplot.imshow(a[0].data-np.median(a[0].data),
                      vmin=0, cmap='gnuplot')
        pyplot.xlim(cir[k][0]-40, cir[k][0]+40)
        pyplot.ylim(cir[k][1]-40, cir[k][1]+40)
        t = np.linspace(0,2*np.pi,100)
        pyplot.plot(30*np.cos(t)+cir[k][0],
                    30*np.sin(t)+cir[k][1], 'w-',
                    linewidth=3, alpha=0.5)
        a.close()
    return

def naco():
    f1 = pyfits.open('/Users/amerand/DATA/NACO/HD10360/NACO_ACQ325_0001.fits')
    f2 = pyfits.open('/Users/amerand/DATA/NACO/HD10360/NACO_ACQ325_0003.fits')
    im1 = f1[0].data - np.median(f1[0].data)
    f1.close()
    im2 = f2[0].data - np.median(f2[0].data)
    f2.close()
    
    pyplot.figure(0)
    pyplot.clf()
    pyplot.imshow(im1, cmap='gnuplot', vmax=100)
    pyplot.figure(1)
    pyplot.clf()
    pyplot.imshow(im2, cmap='gnuplot', vmax=100)
    return

def leastsqOPDModel(p_fit, fit, p_fixed, obs):
    try:
        p = np.zeros(fit.size)
        p[np.where(fit)] = p_fit
        p[np.where(1-fit)] = p_fixed
    except:
        p = p_fit
    
    res = []
    for x in obs:
        res.extend(prima.projBaseline(p,[x['RA'], x['DEC']],
                                      x['LST'])['opd']/x['AIRN']-
                   x['DL2-DL1'])
    res = np.array(res)
    print 'B=', p, 'RMS (um)', res.std()*1e6
    return np.array(res)

def OPDmodel(loadData=False, bin=10, do_fit=False):
    """
    """
    global opd_model_data
    try:
        print len(opd_model_data)
    except:
        loadData=True
    if loadData:
        files = ['2011-11-21/PACMAN_OBJ_ASTRO_326_00%s.fits'%(f) for f in
                 ['04', '05', '06', '07', '08', '09', '11', '12',
                  '13', '14', '15', '16', '17', '18', '19', '20']]
        opd_model_data = []
        for f in files:
            a = prima.drs(data_directory+f)
            opd_model_data.append(a.expectedOPD(bin=bin))
            a = []

    B = opd_model_data[0]['XYZ']
    fit = np.array([1,1,1,1])
    p_fit = np.array(B)[np.where(np.array(fit))]
    p_fixed = np.array(B)[np.where(1-np.array(fit))]

    obsN = filter(lambda x: x['INSMODE']=='NORMAL',  opd_model_data)
    obsS = filter(lambda x: x['INSMODE']=='SWAPPED', opd_model_data)
    
    plsq = leastsq(leastsqOPDModel, p_fit, args=(fit, p_fixed, obsN), epsfcn=1e-3)
    Bf = np.array(B)
    Bf[np.where(fit)] = plsq[0]

    print np.array(B)-np.array(Bf)

    for x in opd_model_data:
        x['ORIGINAL']=prima.projBaseline(x['XYZA'], [x['RA'], x['DEC']],
                                         x['LST'])['opd']/x['AIRN']-x['DL2-DL1']
        x['RESIDUALS']=prima.projBaseline(Bf, [x['RA'], x['DEC']],
                                          x['LST'])['opd']/x['AIRN']-x['DL2-DL1']
    obsN = filter(lambda x: x['INSMODE']=='NORMAL', opd_model_data)
    obsS = filter(lambda x: x['INSMODE']=='SWAPPED', opd_model_data)
    obs = opd_model_data
    
    pyplot.figure(0)
    pyplot.clf()
    pyplot.subplot(221)
    pyplot.plot([x['AZ'] for x in obsN], [x['RESIDUALS'].mean() for x in obsN], 'ob',
                label='NORMAL')
    pyplot.plot([x['AZ'] for x in obsS], [x['RESIDUALS'].mean() for x in obsS], 'or',
                label='SWAPPED')
    pyplot.plot([x['AZ'] for x in obsN], [x['ORIGINAL'].mean() for x in obsN], '+b',
                label='NORMAL')
    pyplot.plot([x['AZ'] for x in obsS], [x['ORIGINAL'].mean() for x in obsS], '+r',
                label='SWAPPED')
    #pyplot.legend() 
    pyplot.xlabel('AZ')
    pyplot.ylabel('residuals')
    pyplot.subplot(222)
    pyplot.plot([x['ALT'] for x in obsN], [x['RESIDUALS'].mean() for x in obsN], 'ob',
                label='NORMAL')
    pyplot.plot([x['ALT'] for x in obsS], [x['RESIDUALS'].mean() for x in obsS], 'or',
                label='SWAPPED')
    pyplot.plot([x['ALT'] for x in obsN], [x['ORIGINAL'].mean() for x in obsN], '+b',
                label='NORMAL')
    pyplot.plot([x['ALT'] for x in obsS], [x['ORIGINAL'].mean() for x in obsS], '+r',
                label='SWAPPED')
    #pyplot.legend() 
    pyplot.xlabel('ALT')
    pyplot.ylabel('residuals')
    pyplot.subplot(223)
    pyplot.plot(range(len(obs)), [x['RESIDUALS'].mean() for x in obs], 'ok',
                label='NORMAL')
    pyplot.plot(range(len(obs)), [x['ORIGINAL'].mean() for x in obs], '+k',
                label='NORMAL')
    #pyplot.legend() 
    pyplot.xlabel('N')
    pyplot.ylabel('residuals')
    return opd_model_data
