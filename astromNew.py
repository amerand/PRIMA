import pyfits
import numpy as np
import dpfit
import os
from matplotlib import pyplot
import prima
import pca
from multiprocessing import Pool
import astro
import time

"""
data interpretation of PRIMA astrometric measurements

also has some codes to simulate stuff.
"""


def ipbreak():
    """
    http://wiki.ipython.org/Cookbook/Interrupting_threads
    """
    import IPython.Shell
    if IPython.Shell.KBINT:
        IPython.Shell.KBINT = False
        raise SystemExit

def Pfuhl_pupilPxPy(rot, az, prms):
    """
    rot az in degrees 
    """
    d2r = np.pi/180.0
    a0, a1, phi0, phi1, phi2, x0, y0 = (prms['a0'],
                                        prms['a1'],
                                        prms['phi0']*d2r,
                                        prms['phi1']*d2r,
                                        prms['phi2']*d2r,
                                        prms['x0'],
                                        prms['y0'])
    
    px = a0*np.cos(az*d2r - phi0 - phi2 - 2*rot*d2r) + a1*np.cos(phi1 + rot*d2r) + \
         x0*np.cos(phi2 + 2*rot*d2r) - y0*np.sin(phi2 + 2*rot*d2r)
    py =-a0*np.sin(az*d2r - phi0 - phi2 - 2*rot*d2r) + a1*np.sin(phi1 + rot*d2r) + \
         x0*np.sin(phi2 + 2*rot*d2r) + y0*np.cos(phi2 + 2*rot*d2r)
    return (px, py)

def Bonnet_pupilPxPy(rot, alt, az, prms):
    """
    see email 15-03-2012

    prms is a dict with keys: ['ctX', 'stX', 'cdX', 'sdX', 'c2dX',
    's2dX', 'ctY', 'stY', 'cdY', 'sdY', 'c2dY', 's2dY']
     
    returns px,py: pupil movement in IRIS pixels (Note: 23.5mm/pix for
    ATs)
    """
    theta = az-alt-2*rot 

    d2r = np.pi/180.0
    ct, st, cd, sd, c2d, s2d = (prms['ctX'],
                                prms['stX'],
                                prms['cdX'],
                                prms['sdX'],
                                prms['c2dX'],
                                prms['s2dX'])
    pU = (ct*np.cos(theta*d2r)  + st*np.sin(theta*d2r) + 
          cd*np.cos(rot*d2r)    + sd*np.sin(rot*d2r) + 
          c2d*np.cos(2*rot*d2r) + s2d*np.sin(2*rot*d2r))

    ct, st, cd, sd, c2d, s2d = (prms['ctY'],
                                prms['stY'],
                                prms['cdY'],
                                prms['sdY'],
                                prms['c2dY'],
                                prms['s2dY'])
    pW = (ct*np.cos(theta*d2r)  + st*np.sin(theta*d2r) + 
          cd*np.cos(rot*d2r)    + sd*np.sin(rot*d2r) + 
          c2d*np.cos(2*rot*d2r) + s2d*np.sin(2*rot*d2r))

    return (pU, pW)

def dOPDfuncMultiTargets(X, params, addIntermediate=False):
    """
    mutliple target version of dOPDfunc (still only one baseline)
    X: same as in dOPDfunc

    params: same as dOPDfunc, but each target specific parameter must
    start with 'T:target' where target in the name of the target in
    X. Note also that targets' names should not contain any spaces!
    All parameters without 'T:' will be considered as common to all
    targets.
    """
    targets = list(set([x['TARGET'] for x in X]))

    setX = []
    setP = []
    if addIntermediate:
        residuals = np.array([{} for k in range(len(X))])
    else:
        residuals = np.zeros(len(X))
        
    for t in targets:
        setX.append(filter(lambda x: x['TARGET']==t, X))
        setK = filter(lambda x: 'T:'+t in x or not 'T:' in x, params.keys())
        tmp = {}
        for k in setK:
            if 'T:' in k:
                tmp[reduce(lambda x,y: x+' '+y, k.split()[1:])] = params[k]
            else:
                tmp[k] = params[k]
        setP.append(tmp)

        residuals[np.where([x['TARGET']==t for x in  X])] =\
                                           dOPDfunc(setX[-1], setP[-1],
                                                    addIntermediate=addIntermediate)
    return residuals

def dOPDfunc(X, prms, addIntermediate=False):
    """
    only works with one target, and one baselines

    X = [{'TARGET':, 'RA':, 'DEC':, 'B_XYZA':, 'lst', 'SWAPPED':, 'MJD'}]
    ---------------------------------------------------------------------

    - RA, DEC: the coordinates in decimal hour and degrees
    - B_XYZA is the baseline vector in meters (length 4)
    - LST in decimal hour
    - SWAPPED: 1, 0, or True, False

    possible parameters:
    --------------------

    angles in degrees, separation in arcsec, M0 in mm

    - params: {'M0':, 'SEP':, 'PA':)

    - params: {'M0':, 'SEP':, 'PA':, 'M0S':} where M0S is the zero
      point in SWAPPED

    - params: {'M0 MJD xxxxxx.xxx xxxxxx.xxx', 'M0 MJD yyyyyy.yyyy
      yyyyyyy.yyy':, 'SEP':, 'PA':} with 2 different zero point, based
      on the MJD (the 2 MJD and the min and max range). Should be
      floats, no ints

    linear drift of separation:
    
    - 'SEP MJD xxxxxx.xx':arcsec, 'PA MJD xxxxxxxxx.xx':degree,
      'LINDIR':degrees, 'LINRATE':uas/day 'SEP ...' and 'PA ...'
      defines the separation and PA at a given date. from there, the
      separation vector evolves at the given rate, in the given
      direction.
      
    - addIntermediate: stores intermediate calculations, such as
      projected baseline, separation, M0 etc. PRIMETphase counts when
      PRIMET has been initialized. Only point with same PRIMETphase
      can be combined.

    """
    global dOPDfuncNiter
    c_ = np.pi/(180*3600.0) # rad / arcsec

    if len(X)<1: # case nothing is passed
        return []

    # start with 0 dOPD
    dOPD = np.zeros(len(X))

    # we will need (-1)**swap a few time:
    swap = (-1)**np.array([x['SWAPPED'] for x in X])

    # compute projected baseline:
    projB = prima.projBaseline(X[0]['B_XYZA'],(X[0]['RA'], X[0]['DEC']),
                               [x['LST'] for x in X])
    b_p = projB['B']
    b_pa = projB['PA']
    if addIntermediate:
        for k in range(len(X)):
            X[k]['fit Bp'] = b_p[k]
            X[k]['fit Bpa'] = b_pa[k]

    # single metrology zero point
    if prms.has_key('M0'):
        dOPD += prms['M0']*1e-3
        if addIntermediate:
            for k in range(len(X)):
                X[k]['fit M0'] = prms['M0']*1e-3
                X[k]['fit PRIMETphase'] = 0

    # different zero point for SWAPPED and NORMAL
    if prms.has_key('M0S'):
        dOPD = np.array([prms['M0S']*1e-3 if x['SWAPPED'] else prms['M0'] for x in X])
        if addIntermediate:
            for k in range(len(X)):
                X[k]['fit M0'] = prms['M0S']*1e-3 if X[k]['SWAPPED'] else prms['M0']
                X[k]['fit PRIMETphase'] = 0 # still considered a single PRIMET phase

    # different M0 by MJD range
    if any([k[:6]=='M0 MJD' for k in prms.keys()]):
        mjd = np.array([x['MJD'] for x in X])
        PRIMETphase = 0
        for k in prms.keys():
            if k[:6]=='M0 MJD':
                MJD_min = float(k.split()[2])
                MJD_max = float(k.split()[3])
                w = np.where((mjd<MJD_max)*(mjd>=MJD_min))
                dOPD[w] = prms[k]*1e-3
                if addIntermediate:
                    for k in range(len(w[0])):
                        X[w[0][k]]['fit M0'] = dOPD[w][k]
                        X[w[0][k]]['fit PRIMETphase'] = PRIMETphase
                PRIMETphase += 1

    # astrometric modulation: static
    if prms.has_key('SEP') and prms.has_key('PA'):
        dOPD += swap*b_p*prms['SEP']*c_*\
                np.cos((b_pa-prms['PA'])*np.pi/180)
        if addIntermediate:
            for k in range(len(X)):
                X[k]['fit SEP'] = prms['SEP']
                X[k]['fit PA'] = prms['PA']

    # astrometric modulation: linear
    if any(['SEP MJD' in k for k in prms.keys()]) and \
       any(['PA MJD' in k for k in prms.keys()]) and \
       prms.has_key('LINDIR') and prms.has_key('LINRATE'):
        # get MJD0:
        for k in prms.keys():
            if k[:7]=='SEP MJD':
                MJD0 = float(k.split()[2])
                SEP0 = prms[k]
            if k[:6]=='PA MJD':
                PA0 = prms[k]
        # cartesian separation vector:
        X_ = SEP0*np.sin(PA0*np.pi/180)
        Y_ = SEP0*np.cos(PA0*np.pi/180)
        X_ += 1e-6*(np.array([x['MJD'] for x in X])-MJD0)*prms['LINRATE']*\
              np.sin(prms['LINDIR']*np.pi/180)
        Y_ += 1e-6*(np.array([x['MJD'] for x in X])-MJD0)*prms['LINRATE']*\
              np.cos(prms['LINDIR']*np.pi/180)
        # compute separation/PA as a function of time:
        SEPt = np.sqrt(X_**2+Y_**2)
        PAt  = np.arctan2(X_,Y_)*180/np.pi
        if addIntermediate:
            for k in range(len(X)):
                X[k]['fit SEP'] = SEPt[k]
                X[k]['fit PA'] = PAt[k]
        dOPD += swap*b_p*SEPt*c_*np.cos((b_pa-PAt)*np.pi/180)

    if prms.has_key('SEP'):
        # angle between M10 edge and the binary:
        addAngle = np.arcsin(10./prms['SEP'])*180/np.pi if prms['SEP'] >10\
                   else 90.0
        addAngle += 90 # relative to x
        sx = prms['SEP']*c_*np.cos(addAngle/180.*np.pi) # in rad
        sy = prms['SEP']*c_*np.sin(addAngle/180.*np.pi) # in rad
    else:
        # angle between M10 edge and the binary:
        addAngle = np.arcsin(10./SEP0)*180/np.pi if SEP0 >10 else 90.0
        addAngle += 90 # relative to x
        sx = SEP0*c_*np.cos(addAngle/180.*np.pi) # in rad
        sy = SEP0*c_*np.sin(addAngle/180.*np.pi) # in rad
    if addIntermediate:
        for k in range(len(X)):
            X[k]['fit angleIRIS'] = addAngle

    if prms.has_key('IRIS scale'):
        scale = prms['IRIS scale']
    else:
        scale =  23.5e-3 #  scale of pupil on IRIS, in m/pix
    if prms.has_key('ATPUP scale'):
        scale *= prms['ATPUP scale']

    # pupil runout model
    ####################
    px = np.zeros(len(X))
    py = np.zeros(len(X))

    # -- parameters used in the formula
    Pfuhl = ['a0', 'a1', 'phi0', 'phi1', 'phi2', 'x0', 'y0']
    Bonnet = ['ctX', 'stX', 'cdX', 'sdX', 'c2dX', 's2dX',
              'ctY', 'stY', 'cdY', 'sdY', 'c2dY', 's2dY' ]
    Ts = ['3', '4']
    for t in Ts:
        if all([prms.has_key('AT'+t+' '+p) for p in Pfuhl]):
            az  = np.array([o['AZ'+t] for o in X])
            rot = np.array([o['ROT'+t] for o in X])
            # built dict of parameters
            tmp = {}
            for z in Pfuhl:
                tmp[z] = prms['AT'+t+' '+z]
            # compute error
            tmpX, tmpY = Pfuhl_pupilPxPy(rot, az, tmp)
            px = -tmpX # check sign!
            py = -tmpY # check sign!
        if all([prms.has_key('AT'+t+' '+p) for p in Bonnet]):
            az  = np.array([o['AZ'+t]  for o in X])
            alt = np.array([o['ALT'+t] for o in X])
            rot = np.array([o['ROT'+t] for o in X])
            # built dict of parameters
            tmp = {}
            for z in Bonnet:
                tmp[z] = prms['AT'+t+' '+z]
            # compute error
            tmpX, tmpY = Bonnet_pupilPxPy(rot, alt, az, tmp)
            px += tmpX
            py += tmpY
            
    correctionPup = (px*sx + py*sy)*scale

    if addIntermediate:
        for k in range(len(X)):
            X[k]['fit PUPbias'] = correctionPup[k]
    dOPD += correctionPup
    #### done with pupil ##############
    
    ### finished ###
    dOPDfuncNiter +=1
    if addIntermediate:
        for k in range(len(X)):
            X[k]['fit DOPD'] = dOPD[k]
        return X
    else:
        return dOPD

def fitListOfFiles(files=[], directory='', fitOnlyFiles=None,
                   firstGuess={'M0':0.0, 'SEP':10.0, 'PA':90.0},
                   fitOnly=None, doNotFit=None, maxResiduals=None,
                   verbose=1, reduceData=False, pssRecDir=None,
                   max_err_um=1.0, max_GD_um=2.5, sigma_clipping=3.5,
                   plot=True, randomize=False, exportFits=None,
                   exportAscii=None,
                   rectified=True, allX=False):

    """ files, list of files

    directory='', where the data are

    fitOnlyFiles=None, list of index of files to fit

    firstGuess={'M0':0.0, 'SEP':10.0, 'PA':90.0}, see dOPDfunc



    fitOnly=None, doNotFit=None, list of names of variable to fit or not fit

    verbose=1, 2 for max verbosity (list of files)

    reduceData=False, : to forece reducing the data: relevant parameters:
       pssRecDir=None,
       max_err_um=2.0,
       max_GD_um=4.0,
       sigma_clipping=5.0,

    exportFits=None, exportAscii=None : give a filename to activate
 
    plot=True,
       rectified=True, plot 'rectified' residuals,
       maxResiduals=None, for the plot, for the MAX/min to this value (in um)
       
    randomize=False : to be used for bootstrapping
    """
    global dOPDfuncNiter, obs, obsFit

    if len(files)==0: # no files given means all of them!
        files = os.listdir(directory)
        files = filter(lambda x: '.fits' in x and not 'RED' in x, files)

    if reduceData:
        for f in files:
            print '#'*3, 'reducing:', f, '#'*5
            a = prima.drs(directory+f, pssRec=pssRecDir)
            if 'OBJ_ASTRO' in f: # reduce dual fringe tracking data
                a.astrometryFtk(max_err_um=max_err_um, max_GD_um=max_GD_um,
                                sigma_clipping=sigma_clipping,
                                max_length_s=1.0, writeOut=True,
                                overwrite=True)
            else: # reduce scanning data:
                a.astrometryScan(max_err_um=max_err_um, max_GD_um=max_GD_um,
                                 sigma_clipping=sigma_clipping,
                                 writeOut=True,
                                 overwrite=True)
            a = []
    red = [] # tables of atom
    # load all reduced files
    if fitOnlyFiles is None:
        fitOnlyFiles = range(len(files))

    for k, f in enumerate(files):
        # reduced file:
        rf = f.split('.')[0]+'_RED.fits'
        #  load reduced file
        a = atom(os.path.join(directory,rf))
        # print the table of loaded files:
        if not fitOnlyFiles==None:
            if k in fitOnlyFiles:
                fit='X'
            else:
                fit='_'
        else:
            fit = 'X'
        if verbose>1: # pretty printing of files, in a table:
            if k==0:
                N = len(rf)
                print '|idx |fit|INS.MODE | TARGET     | file'+' '*20+\
                      '| UT '+' '*16+' | MJD-OBS'
            print '| %2d | %1s | %7s | %10s | %s | %s | %10.6f' % \
                  (k,  fit, a.insmode, a.PS_ID[:10],
                   rf.split('OBJ_')[1], a.date_obs[:19], a.mjd_obs)

        red.append(a) # list of reduced files
    if verbose:
        print ''

    # prepare observation vector for the FIT
    obs = []
    for k in range(len(red)):
        for i in range(len(red[k].d_al)):
            obs.append({'TARGET':red[k].data[0].header['ESO OCS PS ID'],
                        'FILENAME':red[k].filename.split('/')[-1],
                        'DOPD':red[k].d_al[i],
                        'eDOPD':red[k].d_al_err[i],
                        'LST':red[k].lst[i],
                        'MJD':red[k].mjd[i].min(),
                        'RA':red[k].radec[0],
                        'DEC':red[k].radec[1],
                        'SWAPPED':int(red[k].swapped[i]),
                        'FIT':k in fitOnlyFiles and red[k].d_al_err[i]>0,
                        'B_XYZA':[red[k].data[0].header['ESO ISS CONF T1X']-
                                  red[k].data[0].header['ESO ISS CONF T2X'],
                                  red[k].data[0].header['ESO ISS CONF T1Y']-
                                  red[k].data[0].header['ESO ISS CONF T2Y'],
                                  red[k].data[0].header['ESO ISS CONF T1Z']-
                                  red[k].data[0].header['ESO ISS CONF T2Z'],
                                  red[k].data[0].header['ESO ISS CONF A1L']-
                                  red[k].data[0].header['ESO ISS CONF A2L']],
                        'ROT3':red[k].rot3[i], 'ROT4':red[k].rot4[i],
                        'AZ3':red[k].az3[i], 'AZ4':red[k].az4[i],
                        'ALT3':red[k].alt3[i], 'ALT4':red[k].alt4[i]})
    obsFit = filter(lambda x: x['FIT'], obs)
    obsNoFit = filter(lambda x: not x['FIT'], obs)
    t0 = time.time()

    if not doNotFit is None:
        #fitOnly = filter(lambda k: not k in doNotFit, firstGuess.keys())
        fitOnly = filter(lambda k: not any([x in k for x in  doNotFit]),
                         firstGuess.keys())
    dOPDfuncNiter = 0

    if randomize>0: # to be used by bootstrapping
        if randomize!=True:
            np.random.seed(randomize)
        w = np.random.randint(len(obsFit), size=len(obsFit))
        tmp = []
        for k in w:
            tmp.append(obsFit[k])
        obsFit=tmp

    if 'T:' in [x[:2] for x in firstGuess.keys()]:
        ### multiple targets case:
        fit = dpfit.leastsqFit(
            dOPDfuncMultiTargets, obsFit, firstGuess,
            [o['DOPD'] for o in obsFit],
            err=[o['eDOPD'] for o in obsFit],
            fitOnly = fitOnly, verbose=0, epsfcn=1e-5)
        best = fit['best']
        uncer = fit['uncer']
        chi2 = fit['chi2']
        model = fit['model']
        
        obs = dOPDfuncMultiTargets(obs, best, addIntermediate=True)
        obsFit = dOPDfuncMultiTargets(obsFit, best, addIntermediate=True)
        obsNoFit = dOPDfuncMultiTargets(obsNoFit, best, addIntermediate=True)
        ### compute a chi2 for each targets:
        targets = list(set([x['TARGET'] for x in obsFit]))
        chi2T = {}
        rmsT = {}
        for t in targets:
            obsT = filter(lambda x: x['TARGET']==t, obsFit)
            tmp = (dOPDfuncMultiTargets(obsT, best)-
                   np.array([x['DOPD'] for x in obsT]))
            rmsT[t] = round(tmp.std()*1e6,2)
            chi2T[t] = round((tmp**2/np.array([x['eDOPD'] for x in obsT])**2).mean(),2)
        if plot:
            print 'RMS (um):', rmsT
            print 'CHI2    :', chi2T
    else:
        fit = dpfit.leastsqFit(dOPDfunc, obsFit,
              firstGuess, [o['DOPD'] for o in obsFit], err=[o['eDOPD']
              for o in obsFit], fitOnly = fitOnly, verbose=0,
              epsfcn=1e-5)
        best = fit['best']
        uncer = fit['uncer']
        chi2 = fit['chi2']
        model = fit['model']
        
        obs = dOPDfunc(obs, best, addIntermediate=True)
        obsFit = dOPDfunc(obsFit, best, addIntermediate=True)
        obsNoFit = dOPDfunc(obsNoFit, best, addIntermediate=True)

    RMSum = (np.array([o['DOPD'] for o in obsFit])-model).std()*1e6
    if plot:
        print 'residuals= %4.2f (um)'%(RMSum)
        print 'CHI2= %4.2f'%(chi2)

    # print result
    tmp = best.keys()
    tmp.sort()
    if best.has_key('AT scale') and  best['AT scale']==0:
        tmp = filter(lambda x
                     : not 'AT4' in x and 'AT3' not in x, tmp)
    units = {'M0':'mm', 'SEP':'arcsec', 'PA': 'deg',
             'AT3 phi': 'deg', 'AT4 phi': 'deg', 'LINRATE':'uas/day',
             'LINDIR': 'deg'}
    if plot:
        for k in tmp:
            if uncer[k]>0:
                nd = int(-np.log10(uncer[k])+2)
                form = '%'+str(max(nd,0)+2)+'.'+str(max(nd,0))+'f'
                form = k+' = '+form+' +/- '+form
                print form%(round(best[k], nd), round(uncer[k], nd)), 
            else:
                print k+' = '+' %f'%(best[k]),
            u = ''
            for z in units.keys():
                if z in k:
                    u = '('+units[z]+')'
            print u

    # export to Ascii:
    if not exportAscii is None:
        f = open(exportAscii, 'w')
        tmp = best.keys()
        tmp.sort()
        for k in tmp:
            f.write('# '+k+' ='+str(best[k])+'\n')
            if uncer[k]!=0:
                f.write('# ERR '+k+' ='+str(uncer[k])+'\n')
        tmp = obs[0].keys()
        tmp.sort()
        f.write('# '+'; '.join([t.replace(' ', '_') for t in tmp])+'\n')
        for o in obs:
            f.write(''+'; '.join([str(o[k]) for k in tmp])+'\n')
        f.close()
    # export to FITS:
    if not exportFits is None:
        # create fits object
        hdu0 = pyfits.PrimaryHDU(None)
        tmp = best.keys()
        tmp.sort()
        for k in tmp:
            hdu0.header.update('HIERARCH '+k, best[k])
            if uncer[k]!=0:
                hdu0.header.update('HIERARCH ERR '+k, uncer[k])
        tmp = obs[0].keys()
        tmp.sort()
        cols=[]
        for k in tmp:
            if k=='MJD':
                form = 'F12.5'
            elif type(obs[0][k])==float or \
                   type(obs[0][k])==np.float or \
                   type(obs[0][k])==np.float32 or \
                   type(obs[0][k])==np.float64:
                form = 'F12.9'
            elif type(obs[0][k])==str:
                form = 'A'+str(max([len(o[k]) for o in obs]))
            elif type(obs[0][k])==bool or \
                     type(obs[0][k])==np.bool or \
                     type(obs[0][k])==np.bool_ or \
                     type(obs[0][k])==int:
                form = 'I'
            else:
                print 'DEBUG:', k, obs[0][k], type(obs[0][k])
                form=''
                
            if k!='B_XYZA':   
                cols.append(pyfits.Column(name=k.replace(' ', '_'),
                                          format=form,
                                          array=[o[k] for o in obs]))
            else:
                cols.append(pyfits.Column(name='B_X', format='F12.8',
                                          array=[o[k][0] for o in obs]))
                cols.append(pyfits.Column(name='B_Y', format='F12.8',
                                          array=[o[k][1] for o in obs]))
                cols.append(pyfits.Column(name='B_Z', format='F12.8',
                                          array=[o[k][2] for o in obs]))
                cols.append(pyfits.Column(name='B_A', format='F12.8',
                                          array=[o[k][3] for o in obs]))

        hducols = pyfits.ColDefs(cols)
        hdub = pyfits.new_table(hducols)
        hdub.header.update('EXTNAME', 'ASTROMETRY REDUCED', '')
        thdulist = pyfits.HDUList([hdu0, hdub])
        outfile = exportFits
        if os.path.exists(outfile):
            os.remove(outfile)
        print 'writting ->', outfile
        thdulist.writeto(outfile)
        return
    
    # return fitting result
    res = {'BEST':best, 'UNCER':uncer, 'CHI2':chi2, 'RMS':RMSum,
           'MJD_MIN':min([o['MJD'] for o in obsFit]),
           'MJD_MAX':max([o['MJD'] for o in obsFit]),
           'MJD_MEAN':np.array([o['MJD'] for o in obsFit]).mean()}
    res['RA']={}
    res['DEC']={}
    for o in obs:
        if not res['RA'].has_key(o['TARGET']):
            res['RA'][o['TARGET']] = o['RA']
            res['DEC'][o['TARGET']] = o['DEC']

    if not plot:
        return res

    # =============== PLOT =============================
    if 'T:' in [x[:2] for x in best.keys()]:
        modelNoFit = dOPDfuncMultiTargets(obsNoFit, best)
    else:
        modelNoFit = dOPDfunc(obsNoFit, best)
        
    pyplot.figure(0, figsize=(18,6))
    pyplot.clf()
    pyplot.subplots_adjust(left=0.05, bottom=0.07, right=0.99,
                    top=0.95, wspace=0.01)

    if allX:
        # display all sorts of X axis
        Xs = [lambda o: (o['LST']-o['RA']+12)%24-12,
              lambda o: o['ROT4'],
              lambda o: (o['AZ4']  - o['ALT4'] - 2*o['ROT4'])%360,
              lambda o: o['AZ4'],
              lambda o: o['MJD'],]
        
        Xlabels = ['hour angle (h)',
                   'ROT4 (degrees)',
                   'AZ4 - ALT4 - 2*ROT4 (degrees)',
                   'AZ4 (degrees)',
                   'MJD',]
    else:
        # display only Hour Angle
        Xs = [lambda o: (o['LST']-o['RA']+12)%24-12]
        Xlabels = ['hour angle (h)']

    for i in range(len(Xs)):
        X = Xs[i]
        Xlabel=Xlabels[i]
        if i==0:
            ax0 = pyplot.subplot(1,len(Xs), i+1)
        else:
            pyplot.subplot(1,len(Xs), i+1, sharey=ax0)
        
        if not rectified:
            ### plot data used for the fit
            pyplot.plot([X(o) if not o['SWAPPED'] else None for o in obsFit],
                        [(obsFit[k]['DOPD']-model[k])*1e6
                         for k in range(len(obsFit))],
                        'ob', alpha=0.3, label='NORMAL')
            pyplot.plot([X(o) if o['SWAPPED'] else None for o in obsFit],
                        [(obsFit[k]['DOPD']-model[k])*1e6
                         for k in range(len(obsFit))],
                        'or', alpha=0.3, label='SWAPPED')
            if len(obsNoFit)>1:
                ### plot data NOT used for the fit
                pyplot.plot([X(o) if not o['SWAPPED'] else None for o in obsNoFit],
                            [(obsNoFit[k]['DOPD']-modelNoFit[k])*1e6
                             for k in range(len(obsNoFit))],
                            '+c', alpha=0.2, label='NORMAL, not fitted')
                pyplot.plot([X(o) if o['SWAPPED'] else None for o in obsNoFit],
                            [(obsNoFit[k]['DOPD']-modelNoFit[k])*1e6
                             for k in range(len(obsNoFit))],
                        '+y', alpha=0.2, label='SWAPPED, not fitted')
            pyplot.ylabel('$O-C$ ($\mu$m)')
        else:
            ### plot data used for the fit
            pyplot.plot([X(o) if not o['SWAPPED'] else None for o in obsFit],
                        [(-1)**obsFit[k]['SWAPPED']*(obsFit[k]['DOPD'] -
                                        model[k])*1e6 for k in range(len(obsFit))],
                        'ob', alpha=0.3, label='NORMAL')
            pyplot.plot([X(o) if o['SWAPPED'] else None for o in obsFit],
                        [(-1)**obsFit[k]['SWAPPED']*(obsFit[k]['DOPD'] -
                                        model[k])*1e6 for k in range(len(obsFit))],
                        'or', alpha=0.3, label='SWAPPED')
            if len(obsNoFit)>1:
                ### plot data NOT used for the fit
                pyplot.plot([X(o) if not o['SWAPPED'] else None for o in obsNoFit],
                            [(-1)**obsNoFit[k]['SWAPPED']*(obsNoFit[k]['DOPD'] -
                                modelNoFit[k])*1e6 for k in range(len(obsNoFit))],
                                '+c' , alpha=0.2, label='NORMAL, not fitted')
                pyplot.plot([X(o) if o['SWAPPED'] else None for o in obsNoFit],
                            [(-1)**obsNoFit[k]['SWAPPED']*(obsNoFit[k]['DOPD'] -
                                modelNoFit[k])*1e6 for k in range(len(obsNoFit))],
                            '+y' , alpha=0.2, label='SWAPPED, not fitted')
                
            # -- plot pupil bias correction
            #pyplot.plot([X(o) for o in obs],
            #            [(-1)**o['SWAPPED']*o['fit PUPbias']*1e6 for o in obs],
            #            '.k', label='pup runout correction')

        if i==0:
            pyplot.ylabel('$(-1)^\mathrm{swapped}(O-C)$ ($\mu$m)')
            pyplot.text(min([X(o) for o in obs]), 0,
                        'RMS='+str(round(RMSum, 2))+'$\mu$m',
                        va='center', ha='left',
                        bbox = dict(boxstyle="round",
                                    ec=(.4, 0.2, 0.0), fc=(1., 0.8, 0.0),
                                    alpha=0.2))
            pyplot.legend(ncol=4, loc='upper left')
        pyplot.xlabel(Xlabel)
        if not maxResiduals is None:
            pyplot.ylim(-maxResiduals, maxResiduals)
    
    return res

def bootstrapListOfFiles(files, directory='', firstGuess={'M0':0.0,
                         'SEP':10.0, 'PA':90.0}, maxResiduals=None,
                         doNotFit=None, N=50, plot=False):
    """
    bootstraped version of fitListOfFiles.

    LIMITATIONS: Only works with one Target, i.e. PA and SEP

    result in arcseconds. 'PCA' is the principal component reduction
    of the error ellipse
    """
    res = []
    p = Pool()
    cb_boot(None, init=True)
    for k in range(N):
        seed = np.random.randint(1e9)
        p.apply_async(f_boot, (files, directory,
                                firstGuess, doNotFit,seed,),
                      callback=cb_boot)
    p.close()
    p.join()
    res = cb_boot(None, ret=True)
    # delta DEC
    X = np.array([z['BEST']['SEP']*np.cos(z['BEST']['PA']*np.pi/180)
                  for z in res])
    # delta RA cos(DEC)
    Y = np.array([z['BEST']['SEP']*np.sin(z['BEST']['PA']*np.pi/180)
                  for z in res])
    p = pca.pca(np.transpose(np.array([(X-X.mean()),
                                       (Y-Y.mean())])))
    err0 = p.coef[:,0].std()
    err1 = p.coef[:,1].std()
    if plot:
        pyplot.figure(10)
        pyplot.clf()
        pyplot.axes().set_aspect('equal', 'datalim')
        pyplot.plot(Y, X, '.k', label='bootstrapped positions', alpha=0.5)
        pyplot.legend()
        pyplot.ylabel(r'$\Delta$ dec [arcsec]')
        pyplot.xlabel(r'$\Delta$ RA $\cos$(dec) [arcsec]')

    # results in usual coordinate frame.
    # units: arcseconds
    
    result={'Delta RA cos(DEC)':Y, 'Delta DEC':X,
            'AVG Delta RA cos(DEC)':Y.mean(), 'AVG Delta DEC':X.mean(),
            'PCA':(list(p.base[0]),list(p.base[1])) ,
            'errs':(err0, err1)}
    return result

def cb_boot(x, init=False, ret=False):
    """
    """
    global res_boot
    if init>0:
        res_boot = []
        return
    if ret:
        return res_boot
    res_boot.append(x)
    return

def f_boot(files, directory, firstGuess, doNotFit, seed=0):
    return fitListOfFiles(files, directory=directory,
                          firstGuess=firstGuess, doNotFit=doNotFit,
                          verbose=False, plot=False,
                          randomize=seed)

def computeRotatorAngle(obs, SEP, PA, debug=False):
    if debug:
        print 'RA, DEC:', obs[0]['RA'], obs[0]['DEC']
        print np.arcsin(10./SEP)*180/np.pi if SEP>10 else 90
        rot3 = [(prima.projBaseline(o['B_XYZA'],
                                    (o['RA'], o['DEC']), o['LST'])['parang'] +
                 PA + (np.arcsin(10./SEP)*180/np.pi if SEP>10 else 90) +
                 180*o['SWAPPED'] + o['AZ3'])/2. for o in obs]
        rot3 = np.array(rot3)%180
        az3 = [prima.projBaseline(o['B_XYZA'],
                                  (o['RA'], o['DEC']),
                                  o['LST'])['az'] for o in obs]

        pyplot.figure(0)
        pyplot.clf()
        pyplot.plot(np.array([o['ROT3'] for o in obs])-rot3, '.r')
        pyplot.xlabel('index')
        pyplot.ylabel('ROT3 - computed (degrees)')
        pyplot.figure(1)
        pyplot.clf()
        pyplot.plot((np.array([o['AZ3'] for o in obs])-az3)%360, '.r')
        pyplot.xlabel('index')
        pyplot.ylabel('AZ - computed (degrees)')
    return

class atom():
    """
    atomic observation (one file). reads and load one reduced fits
    file. nothing else so far
    """
    def __init__(self, filename, debug=False):
        he = 'HIERARCH ESO '
        self.data = pyfits.open(filename)
        self.filename = filename
        self.date_obs = self.data[0].header['DATE-OBS']
        self.targname = self.data[0].header['HIERARCH ESO OBS TARG NAME']
        self.PS_ID = self.data[0].header['HIERARCH ESO OCS PS ID']
        self.SS_ID = self.data[0].header['HIERARCH ESO OCS SS ID']
        self.mjd_obs = self.data[0].header['MJD-OBS']
        self.lst_obs = self.data[0].header['LST']/3600.
        self.mjd = self.data['ASTROMETRY_BINNED'].data.field('MJD')
        try:
            self.lst= self.data['ASTROMETRY_BINNED'].data.field('LST')
        except:
            print 'WARNING: USING BAD LST!'
            self.lst= self.lst_obs + (self.mjd - self.mjd_obs)*(23.+56/60.+4/3600.)
       
        self.d_al= self.data['ASTROMETRY_BINNED'].data.field('D_AL')
        if self.PS_ID=='HD129926' and self.mjd_obs<56005.0:
            #problem with these observations
            print 'WARNING: KLUDGING D_AL -> -DA_L!!!'
            self.d_al *=-1
        self.d_al_err= self.data['ASTROMETRY_BINNED'].data.field('D_AL_err')
        try:
            self.rot3= self.data['ASTROMETRY_BINNED'].data.field('ROT3')
            self.rot4= self.data['ASTROMETRY_BINNED'].data.field('ROT4')
            self.az3 = self.data['ASTROMETRY_BINNED'].data.field('AZ3')
            self.az4 = self.data['ASTROMETRY_BINNED'].data.field('AZ4')
            self.alt3= self.data['ASTROMETRY_BINNED'].data.field('ALT3')
            self.alt4= self.data['ASTROMETRY_BINNED'].data.field('ALT4')
        except:
            self.rot3= np.zeros(len(self.d_al))
            self.rot4= np.zeros(len(self.d_al))
            self.az3= np.zeros(len(self.d_al))
            self.az4= np.zeros(len(self.d_al))
            self.alt3= np.zeros(len(self.d_al))
            self.alt4= np.zeros(len(self.d_al))

        self.mjd_PCR_start = astro.tag2mjd(self.data[0].header[he+'PCR ACQ START'])
        self.mjd_PCR_end = astro.tag2mjd(self.data[0].header[he+'PCR ACQ END'])
        self.lst_PCR_start = astro.tag2lst(self.data[0].header['ESO PCR ACQ START'],
                                           longitude=self.data[0].header['ESO ISS GEOLON'])
        self.A1L = self.data[0].header[he+'ISS CONF A1L']
        self.A2L = self.data[0].header[he+'ISS CONF A2L']
        try:
            self.insmode = self.data[0].header[he+'ISS PRI STS3 GUIDE_MODE']
        except:
            self.insmode = self.data[0].header[he+'INS MODE']
        
        if not self.insmode in ['NORMAL', 'SWAPPED']:
            #print 'WARNING: unknown INSMODE=', self.insmode
            pass
        if self.insmode == 'INCONSISTENT':
            # -- try to guess
            if self.data[0].header[he+'DEL FT SENSOR'] == 'FSUB':
                self.insmode = 'NORMAL'
            elif self.data[0].header[he+'DEL FT SENSOR'] == 'FSUA':
                self.insmode = 'SWAPPED'
            #print '  -> guessing: INSMODE=', self.insmode

        cP  = self.data[0].header[he+'ISS PRI MET C'] # in m/s
        dnuP = self.data[0].header[he+'ISS PRI MET F_SHIFT']*1e6 # in Hz
        nuP  = self.data[0].header[he+'ISS PRI MET LASER_F']
        #self.jump =  (cP*dnuP/2/(nuP**2))*(2**24-1) # PRIMET jump in m, COMM14
        self.jump =  (cP*dnuP/2/(nuP**2))*(2**31-1) # PRIMET jump in m, COMM15

        if 'SWAP' in self.insmode: # legacy: was SWAP at some point
            self.swapped = np.ones(len(self.mjd))
        else:
            astro.tag2lst(self.data[0].header['ESO PCR ACQ START'],
                          longitude=self.data[0].header['ESO ISS GEOLON'])
            self.swapped = np.zeros(len(self.mjd))

        # dictionnary of variables(mjd) START
        names = ['base', 'PA', 'OPL1', 'OPL2', 'airmass',
                 'tau0','seeing', 'parang']
        keyw = ['ISS PBL12', 'ISS PBLA12', 'DEL DLT1 OPL',
                'DEL DLT2 OPL', 'ISS AIRM', 'ISS AMBI TAU0',
                'ISS AMBI FWHM', 'ISS PARANG']
        self.var_start_end = {}
        for k in range(len(names)):
            self.var_start_end[names[k]]=(
                self.data[0].header[he+keyw[k]+' START'],
                self.data[0].header[he+keyw[k]+' END'])
        self.var_mjd={}

        # linear interpolation
        for k in self.var_start_end.keys():
            if k != 'base' and k != 'PA':
                self.var_mjd[k] = self.var_start_end[k][0]+\
                                  (self.mjd - self.mjd_PCR_start)/\
                                  (self.mjd_PCR_end - self.mjd_PCR_start)*\
                                  (self.var_start_end[k][1]-
                                   self.var_start_end[k][0])
            else:
                # compute baselines
                self.baseXYZ = [self.data[0].header[he+'ISS CONF T1X']-
                                self.data[0].header[he+'ISS CONF T2X'],
                                self.data[0].header[he+'ISS CONF T1Y']-
                                self.data[0].header[he+'ISS CONF T2Y'],
                                self.data[0].header[he+'ISS CONF T1Z']-
                                self.data[0].header[he+'ISS CONF T2Z'],
                                self.data[0].header[he+'ISS CONF A1L']-
                                self.data[0].header[he+'ISS CONF A2L']]
                # coordinates, corrected for proper motion
                try:
                    self.radec = [
                        astro.ESOcoord2decimal(self.data[0].header['ESO OCS TARG1 ALPHAPMC']),
                        astro.ESOcoord2decimal(self.data[0].header['ESO OCS TARG1 DELTAPMC'])]
                except:
                    print 'WARNING: could not find precessed coordinates in header!'
                    self.radec = [astro.ESOcoord2decimal(self.data[0].header['ESO ISS REF RA']),
                                  astro.ESOcoord2decimal(self.data[0].header['ESO ISS REF DEC'])]
                self.radec = [self.data[0].header['RA']/15.,
                              self.data[0].header['DEC']]
                bb = prima.projBaseline(self.baseXYZ, self.radec, self.lst)
                if k == 'base':
                    self.var_mjd[k] = bb['B']
                    if debug:
                        print 'DEBUG: PBL12 START: computed',\
                              self.var_mjd[k][0],\
                              'in header:',\
                              self.data[0].header[he+'ISS PBL12 START'], \
                              'DELTA=', self.var_mjd[k][0]-\
                              self.data[0].header[he+'ISS PBL12 START']
                        print 'DEBUG: PBL12 END  : computed',\
                              self.var_mjd[k][-1],\
                              'in header:',\
                              self.data[0].header[he+'ISS PBL12 END'], \
                              'DELTA=', self.var_mjd[k][-1]-\
                              self.data[0].header[he+'ISS PBL12 END']
                if k == 'PA':
                    self.var_mjd[k] = bb['PA']
                    if debug:
                        print 'DEBUG: PBLA12 START: computed',\
                              self.var_mjd[k][0],\
                              'in header:',\
                              self.data[0].header[he+'ISS PBLA12 START'],\
                              'DELTA=', self.var_mjd[k][0]-\
                              self.data[0].header[he+'ISS PBLA12 START']
                        print 'DEBUG: PBLA12 END  : computed',\
                              self.var_mjd[k][-1],\
                              'in header:',\
                              self.data[0].header[he+'ISS PBLA12 END'],\
                              'DELTA=', self.var_mjd[k][-1]-\
                              self.data[0].header[he+'ISS PBLA12 END']
        self.data.close()
        return
