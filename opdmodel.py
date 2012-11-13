import prima
import cPickle
import os
import dpfit
import numpy as np
from matplotlib import pyplot

def exctractOpdFromFiles(files):
    """
    run the .expectedOPD methon from the PRIMA drs and pickled it in a
    '.opd' file.
    """
    for f in files:
        a = prima.drs(f)
        opd = a.expectedOPD(plot=False)
        g = open(f+'.opd', 'wb')
        print 'saving:', f+'.opd' 
        cPickle.dump(opd, g, 2)
        g.close
        a.raw.close()

def reduceModel(directory):
    """
    open all .opd files in a directory and fit an OPD model
    """
    # load all .opd files:
    files = os.listdir(directory)
    files = filter(lambda x: os.path.splitext(x)[1]=='.opd', files)
    res = []
    for f in files:
        #print f
        g = open(os.path.join(directory, f), 'r')
        res.append(cPickle.load(g))
        g.close
    print [o['TARGET'] for o in res]
    # -- keep only normal
    #res = filter(lambda x: x['INSMODE']=='NORMAL' and x['TARGET']=='HD66598', res)
    res = filter(lambda x: x['INSMODE']=='NORMAL' and x['TARGET']=='HD65297', res)
    #res = filter(lambda x: x['INSMODE']=='SWAPPED' and x['TARGET']=='HD66598', res)
    DL2_DL1 = []
    DDL = []
    ERR = []
    OPD = []
    for o in res:    
        DL2_DL1.extend(o['DL2-DL1'])
        DDL.extend(o['DDL'])
        ERR.extend(o['ERR'])
        OPD.extend(o['computed opd'])
    DL2_DL1 = np.array(DL2_DL1)
    DDL = np.array(DDL)
    ERR = np.array(ERR)
    OPD = np.array(OPD)
    guess = {'X':res[0]['XYZA'][0],
             'Y':res[0]['XYZA'][1],
             'Z':res[0]['XYZA'][2],
             'A':res[0]['XYZA'][3],
             'LAT':-24.62743941}
    best, uncer, chi2, model =\
          dpfit.leastsqFit(opdFunc, res, guess, DL2_DL1, err=ERR,
                               doNotFit=['Z', 'LAT'])
    print 'CHI2:', chi2
    print 'RESIDUAL:', (DL2_DL1-model).std(), 'm'
    ke = best.keys(); ke.sort()
    for k in ke:
        if uncer[k]>0:
            print '| %3s | %10.6f | %10.6f +- %8.6f |' %\
                  (k,guess[k], best[k], uncer[k] )
        else:
            print '| %3s | %10.6f | %10.6f             |' %\
                  (k,guess[k], best[k] )

    pyplot.figure(0)
    pyplot.clf()

    ax = pyplot.subplot(221)
    pyplot.plot(DL2_DL1, 'sb', label='OPD in file')
    pyplot.plot(model, '-b', label='this model')
    pyplot.xlabel('index')
    pyplot.ylabel('m')
    pyplot.legend()

    pyplot.subplot(222, sharex=ax)
    pyplot.plot((DL2_DL1-model)*1e3, '-b', label='residuals')
    #pyplot.plot((DL2_DL1-OPD)*1e3, '-g', label='original OPD model', linestyle='dashed')
    pyplot.xlabel('index')
    pyplot.ylabel('mm')
    pyplot.legend()

    px = pyplot.subplot(223, polar=True)
    #pyplot.plot([o['AZ'] for o in res], [90-o['ALT'] for o in res], '*')
    pyplot.plot([(270-o['AZ'])*np.pi/180 for o in res], [90-o['ALT'] for o in res], '*')
    pyplot.text(np.pi/2, 70, 'N', size='x-large', color='k',
                horizontalalignment='center',
                verticalalignment='center',)
    pyplot.text(0.0, 70, 'E', size='x-large', color='k',
                horizontalalignment='center',
                verticalalignment='center')
    pyplot.text(-np.pi/2, 70, 'S', size='x-large', color='k',
                horizontalalignment='center',
                verticalalignment='center')
    pyplot.text(np.pi, 70, 'W', size='x-large', color='k',
                horizontalalignment='center',
                verticalalignment='center')
    px.get_xaxis().set_visible(False)
    pyplot.ylim(0, 70)
    return 

def opdFunc(obs, params):
    res = []
    for o in obs:
        res.extend(prima.projBaseline([params['X'],
                                       params['Y'],
                                       params['Z'],
                                       params['A']],
                                      [o['RA'], o['DEC']],
                                      o['LST'],
                                      latitude = params['LAT'])['opd']),
    return np.array(res)
    
directory = os.path.join(os.environ['HOME'], 'DATA/PRIMA/COMM18/')

files_019_1 = ['2012-01-18/PACMAN_OBJ_ASTRO_019_0001.fits',
               '2012-01-18/PACMAN_OBJ_ASTRO_019_0002.fits',
               '2012-01-18/PACMAN_OBJ_ASTRO_019_0003.fits',
               '2012-01-18/PACMAN_OBJ_ASTRO_019_0004.fits',
               '2012-01-18/PACMAN_OBJ_ASTRO_019_0005.fits',
               '2012-01-18/PACMAN_OBJ_ASTRO_019_0006.fits',
               '2012-01-18/PACMAN_OBJ_ASTRO_019_0007.fits', 
               '2012-01-18/PACMAN_OBJ_ASTRO_019_0008.fits', 
               '2012-01-18/PACMAN_OBJ_ASTRO_019_0009.fits',
               '2012-01-18/PACMAN_OBJ_ASTRO_019_0010.fits',
               '2012-01-18/PACMAN_OBJ_ASTRO_019_0011.fits',
               '2012-01-18/PACMAN_OBJ_ASTRO_019_0012.fits',
               '2012-01-18/PACMAN_OBJ_ASTRO_019_0013.fits',
               '2012-01-18/PACMAN_OBJ_ASTRO_019_0014.fits',
               '2012-01-18/PACMAN_OBJ_ASTRO_019_0015.fits',
               '2012-01-18/PACMAN_OBJ_ASTRO_019_0016.fits']

files_019_2 = ['2012-01-18/PACMAN_OBJ_ASTRO_019_0017.fits',
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
               '2012-01-18/PACMAN_OBJ_ASTRO_019_0031.fits',]

#exctractOpdFromFiles([os.path.join(directory, f) for f in files_019_2])
