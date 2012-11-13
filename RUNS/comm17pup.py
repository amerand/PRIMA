import astromNew
from matplotlib import pyplot
import numpy as np
import dpfit

# combines AT3 and AT4, expressed foPUPr AT4:
AT4_a_1_combined = 0.34
AT4_phi_1_combined = -94.34
AT4_a_2_combined = 9.0
AT4_phi_2_combined = 142.72
# ----------------------------

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

data_directory = '/Volumes/DATA500/PRIMA/COMM17/' # external BIG hardrive
data_directory = '/Volumes/DATA/PRIMA/COMM17/' # external small hardrive
#data_directory = '/Users/amerand/DATA/PRIMA/' # internal hardrive

files_hd66598_329 = ['2011-11-24/PACMAN_OBJ_ASTRO_329_00'+k+'.fits'
                     for k in ('22','24','25','26','27','28','29','30',
                               '31','32','33','34','35','36','37',
                               '38','39','40','41','42','43','44',
                               '45','46','47','48',
                               #'49',
                               '50','51')]

files_hd66598 = ['2011-11-23/PACMAN_OBJ_ASTRO_328_00'+k+'.fits'
                  for k in ('31','32','33','34','35','36','37',
                            '38','39','40','41','42','43','44',
                            '45','46','47','48','49','50','51',
                            '53','54','55','56','57')]
files_hd66598.extend(list(files_hd66598_329))

guess_hd66598 = {'SEP':35, 'PA':180,
                 'M0 MJD 55889.0 55889.999':-0.009,
                 'M0 MJD 55890.0 55890.999':-0.010}
for k in ATsPUP.keys():
    guess_hd66598[k] = ATsPUP[k]

#astromNew.fitListOfFiles(comm17pup.files_hd66598, directory=comm17pup.data_directory, firstGuess=comm17pup.guess_hd66598, doNotFit=['AT3', 'AT4', 'ATPUP scale'])

### following files were taken simultaneously with NACO observations
files_hd10360 = [['2011-11-20/PACMAN_OBJ_ASTRO_325_0001.fits',
                  '2011-11-20/PACMAN_OBJ_ASTRO_325_0002.fits',
                  '2011-11-20/PACMAN_OBJ_ASTRO_325_0003.fits',
                  '2011-11-20/PACMAN_OBJ_ASTRO_325_0004.fits',
                  '2011-11-20/PACMAN_OBJ_ASTRO_325_0005.fits',
              #'2011-11-20/PACMAN_OBJ_ASTRO_325_0006.fits',# sigma_clipping=2.5
                  '2011-11-20/PACMAN_OBJ_ASTRO_325_0007.fits',
                  '2011-11-20/PACMAN_OBJ_ASTRO_325_0008.fits',
                  '2011-11-20/PACMAN_OBJ_ASTRO_325_0009.fits',
                  #'2011-11-20/PACMAN_OBJ_ASTRO_325_0010.fits', #BAD
                  ]]
files_hd10360_NACO = files_hd10360[0]


fgNACO = {'SEP':8, 'PA':180, 'M0':0.00}
#for k in ATsPUP.keys():
#    fgNACO[k] = ATsPUP[k]

"""
a = astromNew.fitListOfFiles(comm17pup.files_hd10360_NACO,
                             directory=comm17pup.data_directory,
                             firstGuess=comm17pup.fgNACO, plot=False,
                             doNotFit=['AT3', 'AT4', 'ATPUP scale'])
a['Delta DEC'] = a['BEST']['SEP']*np.cos(a['BEST']['PA']*np.pi/180)
a['Delta RA cos(DEC)'] = a['BEST']['SEP']*np.sin(a['BEST']['PA']*np.pi/180)
a['Delta RA'] = a['Delta RA cos(DEC)']/np.cos(a['DEC']['HD10360']*np.pi/180)

"""

files_hd10360.append(['2011-11-23/PACMAN_OBJ_ASTRO_328_00'+k+'.fits'
                     for k in ('01','02',#'03',
                               '04','05','06','07',
                               '08','09','10','11','13','14','15',
                               '16','17','18','19','20','21','22',
                               '24','25','26','27','28','29','30') ])
files_hd10360.append(['2011-11-24/PACMAN_OBJ_ASTRO_329_00'+k+'.fits'
                     for k in ('01','02','04','05','06',#'07',
                               '08','09','10','11','13','14','15',
                               '17','18','19','20')])
files_hd10360.append(['2011-11-25/PACMAN_OBJ_ASTRO_330_00'+k+'.fits'
                      for k in (#'01',
                                '02','03','04','05','06','07',
                                '08','09','10','11','12','13','14',
                                '15','16')])
files_hd10360.append(['2011-11-25/PACMAN_OBJ_ASTRO_330_00'+k+'.fits'
                        for k in ('17',
                                  '18','19','20','21','22','23',
                                  '24',#'25',
                                  '26','27','28','29','30',
                                  '31','32','33','34','35','36','37',
                                  '38','39','40','41')])

files_hd10360_all = []
for f in files_hd10360:
    files_hd10360_all.extend(f)
guessHD10360 = {'M0 MJD 55886 55886.3':0.0,
                'M0 MJD 55889 55889.3':0.0,
                'M0 MJD 55890 55890.3':0.0,
                'M0 MJD 55891 55891.15':0.0,
                'M0 MJD 55891.16 55891.3':0.0,
                'SEP MJD 55885': 11.45, 'PA MJD 55885': 188, 'LINDIR': 90.0, 'LINRATE':133.0,
                #'SEP': 11.45, 'PA': 133,
                }
for k in ATsPUP.keys():
    guessHD10360[k] = ATsPUP[k]


# astromNew.fitListOfFiles(comm17pup.files_hd10360_all, directory=comm17pup.data_directory, firstGuess=comm17pup.guessHD10360, doNotFit=['LINRATE', 'AT3', 'AT4', 'ATPUP scale']); os.system('say done')


files_all = list(files_hd66598)
files_all.extend(files_hd10360_all)
guessAll = {'T:HD10360 M0 MJD 55886 55886.3':-3.44,
            'T:HD10360 M0 MJD 55889 55889.3':-1.34,
            'T:HD10360 M0 MJD 55890 55890.3':0.017,
            'T:HD10360 M0 MJD 55891 55891.15':-1.94,
            'T:HD10360 M0 MJD 55891.16 55891.3':-3.52,
            'T:HD10360 SEP MJD 55885': 11.44,'T:HD10360 PA MJD 55885':188.0,'T:HD10360 LINRATE': 133.0, 'T:HD10360 LINDIR':90.,
            #'T:HD10360 SEP': 11.45,'T:HD10360 PA':187.0,
            'T:HD66598 SEP':35.8, 'T:HD66598 PA':135.,
            'T:HD66598 M0 MJD 55889.0 55889.999':-10.1,
            'T:HD66598 M0 MJD 55890.0 55890.999':-8.73}

guess2All = {'T:HD10360 M0 MJD 55886 55886.3':-3.44,
             'T:HD10360 M0 MJD 55889 55889.3':-1.34,
             'T:HD10360 M0 MJD 55890 55890.3':0.017,
             'T:HD10360 M0 MJD 55891 55891.15':-1.94,
             'T:HD10360 M0 MJD 55891.16 55891.3':-3.52,
             'T:HD10360 SEP MJD 55885': 11.44,'T:HD10360 PA MJD 55885':188.0,'T:HD10360 LINRATE': 133.0, 'T:HD10360 LINDIR':90.0,
             #'T:HD10360 SEP': 11.45,'T:HD10360 PA':187.0,
             'T:HD66598 SEP':35.8, 'T:HD66598 PA':134.,
             'T:HD66598 M0 MJD 55889.0 55889.999':-10.1,
             'T:HD66598 M0 MJD 55890.0 55890.999':-8.73,
             'AT4 a_2':AT4_a_2_combined, 'AT4 phi_2':AT4_phi_2_combined,
             'AT4 a_1':AT4_a_1_combined, 'AT4 phi_1':AT4_phi_1_combined,
             }

guess3All = {'T:HD10360 M0 MJD 55886 55886.3':-3.44,
             'T:HD10360 M0 MJD 55889 55889.3':-1.34,
             'T:HD10360 M0 MJD 55890 55890.3':0.017,
             'T:HD10360 M0 MJD 55891 55891.15':-1.94,
             'T:HD10360 M0 MJD 55891.16 55891.3':-3.52,
             'T:HD10360 SEP MJD 55885': 11.44,'T:HD10360 PA MJD 55885':188.0,'T:HD10360 LINRATE': 133.0, 'T:HD10360 LINDIR':90.0,
             #'T:HD10360 SEP': 11.45,'T:HD10360 PA':187.0,
             'T:HD66598 SEP':35.8, 'T:HD66598 PA':134.,
             'T:HD66598 M0 MJD 55889.0 55889.999':-10.1,
             'T:HD66598 M0 MJD 55890.0 55890.999':-8.73}

for k in ATsPUP.keys():
    guess3All[k] = ATsPUP[k]


# 133uas/day is from Johannes, from the known orbit
def fitAllHD10360():
    global pup
    HD10360corr = []
    HD10360 = []
    for k, f in enumerate(files_hd10360):
        print k+1, '/', len(files_hd10360)
        fg = {'SEP':8., 'PA':135, 'M0':0.0}
        for k in ATsPUP.keys():
            fg[k] = ATsPUP[k]
        HD10360corr.append(astromNew.fitListOfFiles(f, directory=data_directory,
                                                  firstGuess=fg,
                                                  doNotFit=['AT3','AT4','PUP'],
                                                  plot=False, verbose=False))
        print HD10360corr[-1]['MJD_MIN'], '-', HD10360corr[-1]['MJD_MAX']

        HD10360.append(astromNew.fitListOfFiles(f, directory=data_directory,
                                     firstGuess={'SEP':8., 'PA':135, 'M0':-0.0},
                                     plot=False, verbose=False))


    pyplot.figure(10)
    pyplot.clf()
    pyplot.axes().set_aspect('equal', 'datalim')
    th = np.linspace(0, 2*np.pi, 100)
    colors=['r', 'g', 'b', 'y', 'm', 'c']

    ### data reduced without correction: ####################
    XYT = []
    for k,d in enumerate(HD10360):
        ### observation:
        X = d['BEST']['SEP']*np.sin(d['BEST']['PA']*np.pi/180)
        Y = d['BEST']['SEP']*np.cos(d['BEST']['PA']*np.pi/180)
        XYT.append((X,Y, d['MJD_MEAN']))
        pyplot.plot(X, Y, 's'+colors[k], markersize=10, alpha=0.5,
                    label = 'no correction' if k==0 else None)
    # linear fit
    Xb, Xe, chi2, Xm =\
        dpfit.leastsqFit(dpfit.polyN, [x[2]-55885 for x in XYT],
                         {'A0':0, 'A1':1},
                         [x[0] for x in XYT])
    Yb, Ye, chi2, Ym =\
       dpfit.leastsqFit(dpfit.polyN, [x[2]-55885 for x in XYT],
                        {'A0':0, 'A1':1},
                        [x[1] for x in XYT])
    # display linear fit
    MJD = np.linspace(55885, 55890, 100)
    pyplot.plot(dpfit.polyN(MJD-55885, Xb), dpfit.polyN(MJD-55885, Yb),
                'k-', linewidth=3, alpha=0.3)
    # distance to fit
    for k,d in enumerate(HD10360):
        X = d['BEST']['SEP']*np.sin(d['BEST']['PA']*np.pi/180)
        Y = d['BEST']['SEP']*np.cos(d['BEST']['PA']*np.pi/180)
        pyplot.plot(Xm[k], Ym[k], 'x'+colors[k], markersize=6)
        pyplot.plot([X, Xm[k]], [Y, Ym[k]], '-'+colors[k],
                    linewidth=2, alpha=0.5)

    ### data reduced with correction ############################
    XYT=[]
    for k,d in enumerate(HD10360corr):
        ### observation:
        X = d['BEST']['SEP']*np.sin(d['BEST']['PA']*np.pi/180)
        Y = d['BEST']['SEP']*np.cos(d['BEST']['PA']*np.pi/180)
        XYT.append((X,Y, d['MJD_MEAN']))
        pyplot.plot(X, Y, 'o'+colors[k], markersize=10,
                    label = 'scaled correction' if k==0 else None)
    # linear fit
    Xb, Xe, chi2, Xm =\
        dpfit.leastsqFit(dpfit.polyN, [x[2]-55885 for x in XYT],
                         {'A0':0, 'A1':1},
                         [x[0] for x in XYT])
    Yb, Ye, chi2, Ym =\
       dpfit.leastsqFit(dpfit.polyN, [x[2]-55885 for x in XYT],
                        {'A0':0, 'A1':1},
                        [x[1] for x in XYT])
    print 'linrate:', np.sqrt(Xb['A1']**1+Yb['A1']**2)*1e6, 'uas/day'
    # display linear fit
    MJD = np.linspace(55885, 55892, 10)
    pyplot.plot(dpfit.polyN(MJD-55885, Xb), dpfit.polyN(MJD-55885, Yb),
                'k-', linewidth=3, alpha=0.3)

    # distance to fit
    for k,d in enumerate(HD10360corr):
        X = d['BEST']['SEP']*np.sin(d['BEST']['PA']*np.pi/180)
        Y = d['BEST']['SEP']*np.cos(d['BEST']['PA']*np.pi/180)
        pyplot.plot(Xm[k], Ym[k], 'x'+colors[k], markersize=6)
        pyplot.plot([X, Xm[k]], [Y, Ym[k]], '-'+colors[k],
                    linewidth=2, alpha=0.5)

    pyplot.legend(loc='upper left')
    pyplot.xlabel('$\delta$ E (\")')
    pyplot.ylabel('$\delta$ N (\")')
    return

def bootstrapHD103060():
    global pup
    HD10360corr = []
    HD10360 = []
    for k, f in enumerate(files_hd10360):
        print k+1, '/', len(files_hd10360)
        HD10360corr.append(astromNew.bootstrapListOfFiles(f, data_directory,
                                 firstGuess={'SEP':8., 'PA':135, 'M0':-0.03,
                                             'AT4 a_2':pup['BEST']['AT4 a_2'],
                                          'AT4 phi_2':pup['BEST']['AT4 phi_2']},
                                     doNotFit=['AT4 phi_2', 'AT4 a_2'], N=100))
        HD10360.append(astromNew.bootstrapListOfFiles(f, data_directory, N=100,
                              firstGuess={'SEP':8., 'PA':135, 'M0':-0.03}))

    pyplot.figure(10)
    pyplot.clf()
    pyplot.axes().set_aspect('equal', 'datalim')
    th = np.linspace(0, 2*np.pi, 100)
    colors=['r', 'g', 'b', 'y', 'm', 'c']
    for k,d in enumerate(HD10360corr):
        ### observation:
        pyplot.plot(d['RA_mas']/1000., d['DEC_mas']/1000., '+'+colors[k],
                markersize=12)
        pyplot.plot(d['RA_mas']/1000.+
                    d['errs_uas'][0]*d['PCA'][0][1]*np.cos(th)/1e6+
                    d['errs_uas'][1]*d['PCA'][0][0]*np.sin(th)/1e6,
                    d['DEC_mas']/1000.+
                    -d['errs_uas'][1]*d['PCA'][0][1]*np.sin(th)/1e6+
                    d['errs_uas'][0]*d['PCA'][0][0]*np.cos(th)/1e6,
                    color=colors[k], linewidth=3, alpha=0.5)
    for k,d in enumerate(HD10360):
        ### observation:
        pyplot.plot(d['RA_mas']/1000., d['DEC_mas']/1000., '.'+colors[k],
                markersize=8)
        pyplot.plot(d['RA_mas']/1000.+
                    d['errs_uas'][0]*d['PCA'][0][1]*np.cos(th)/1e6+
                    d['errs_uas'][1]*d['PCA'][0][0]*np.sin(th)/1e6,
                    d['DEC_mas']/1000.+
                    -d['errs_uas'][1]*d['PCA'][0][1]*np.sin(th)/1e6+
                    d['errs_uas'][0]*d['PCA'][0][0]*np.cos(th)/1e6,
                    color=colors[k], linewidth=3, alpha=0.5)

    return
