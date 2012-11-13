import pyfits
import numpy as np
import scipy
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

def leastsqAstromSepSimple(p_fit, fit, p_fixed, obs):
    """
    p_fit: values of fitted parameters -> [0.3]
    fit = [0,1,0] -> fit only second parameter
    p_fized = [1,2] -> fixed parameters

    parameter vector is [1,0.3,2]

    obs = [Bproj, Bpa, swap, parang, dal, e_dal]

    """
    if fit==None:
        p = p_fit
    elif len(p_fit)==len(fit):
        p = p_fit
    else:
        p = np.zeros(len(fit))
        p[np.where(np.array(fit))]   = p_fit
        p[np.where(1-np.array(fit))] = p_fixed

    # residual, weighted by the error bar
    res = (obs[-2]-astromSepSimple(obs[:4], p))/obs[-1]
    return res

def astromSepSimple(bam, param, add_type=None):
    """
    bam = [proj_base_m, proj_PA_deg, swapped_bool, (rot1), (rot2)]

    if only rot1 is given, assumes it is the parallactic angle;
    otherwise rot1 and rot2 are assumed to the the rotator positions
    of each telescope.

    param = [sep_r_arcsec, sep_pa_deg, met_0_m, C1, p1, C2, p2]

    result in m

    """
    c_ = np.pi/(180*3600.0) # rad / arcsec
    res = param[2]*1e-3 + bam[0]*param[0]*c_*\
          np.cos((bam[1]-param[1])*np.pi/180)
    # swapped case
    w = np.where(bam[2])
    if not w is ():
        res[w] = param[2]*1e-3 - bam[0][w]*param[0]*c_*\
                 np.cos((bam[1][w]-param[1])*np.pi/180)

    if len(param)>3:
        # correction in mechanical angle
        res += 1e-6*(-1)**bam[2]*param[0]*param[3]*\
                np.cos((bam[3]-param[1])/2*np.pi/180+param[4])
    if len(param)>5:
        res += 1e-6*(-1)**bam[2]*param[0]*param[3]*\
                np.cos(bam[3]*np.pi/180+param[4])
        res += 1e-6*(-1)**bam[2]*param[0]*param[5]*\
                np.cos(bam[3]*np.pi/180+param[6])

    return res

def correctForPrimetJumps_JSa(mjd, jump=3.8e-6):
    """
    remove jumps in metrology based on MJD of occurences, given in an
    email from Johannes (14-04-2011)
    """
    g_MJD=[55591.104916,       # measured
           55591.131294862906, # predicted
           55591.157811967489, # predicted
           55591.185016,       # measured
           55591.214023341265] # predicted
    res = np.zeros(len(mjd))
    for m in g_MJD:
        w = np.where(mjd>m)
        res[w] += jump
    return res

def interpListOfFiles(list_of_files, data_directory=None,
                      first_guess=None, fit_param=None, shuffle=False,
                      fit_only_files=None, env_file=None, T0=290.0,
                      P0=734.0, plot=False, FSMplots=False,
                      arbitrary_offsets=None, maxPolyOrder=None,
                      maxResiduals=None, quiet=True,
                      sigmaClipping=None, res2fits=False,
                      saveCalibrated=False):
    """
    - read many reduced files

    - re-interpolate baselines and PA using polynomial fit

    - fit separation

    - maxResiduals=4.0: plots residuals betwwen -4 and 4 microns

    - res2fits=True: save residuals to the fit

    - saveCalibrated=True: save calibrated dOPD, i.e. with removed
      zero metrology
    """
    red = [] # tables of atom
    # load all reduced files
    for k, f in enumerate(list_of_files):
        # reduced file:
        rf = f.split('.')[0]+'_RED.fits'
        #  load reduced file
        a = atom(os.path.join(data_directory,rf))

        # print the table of loaded files:
        if not fit_only_files==None:
            if k in fit_only_files:
                fit='X'
            else:
                fit='_'
        else:
            fit = 'X'
        if not quiet: # pretty printing of data
            if k==0:
                N = len(rf)
                print '|idx |fit|INS.MODE | file'+' '*(len(rf)-3)+'| date '\
                      +' '*(len(a.date_obs)-4)+'|'
            print '| %2d | %1s | %7s | %s | %s |' % (k,  fit, a.insmode, rf, a.date_obs)

        red.append(a) # table of reduced files

    # function fo convert MJD to LST: simple linear extrapolation
    mjd2lst = lambda x: (x-red[0].mjd[0])*(23.+56/60.+4/3600.) + red[0].lst[0]

    # re-interpolate all vars(mjd) with start and end, except OPL
    var_poly_c = {}
    for k in red[0].var_start_end.keys():
        if not k[:3]=='OPL2':
            if k=='base' or k=='PA' or k=='airmass':
                nPoly = np.minimum(2*len(red)-2, 10)
            else:
                nPoly = len(red)//2+1
            if not maxPolyOrder is None:
                nPoly = min(nPoly, maxPolyOrder)
            X = [a.mjd_PCR_start for a in red]
            X.extend([a.mjd_PCR_end for a in red])
            Y = [a.var_start_end[k][0] for a in red]
            Y.extend([a.var_start_end[k][1] for a in red])
            X = np.array(X)
            Y = np.array(Y)
            # points to fit
            if k=='seeing' or k=='tau0':
                w = np.where(Y>0)
            else:
                w = range(len(X))
            # save polynomial coef
            var_poly_c[k] = np.polyfit(X[w]-X.mean(), Y[w], nPoly)
            # update each observation
            if k!= 'base' and k!='PA':
                for a in red:
                    a.var_mjd[k] = np.polyval(var_poly_c[k] ,a.mjd-X.mean())

        else:
            var_poly_c[k]= [0]

    if not env_file==None: # load Also environmental data.
        e = prima.env(env_file)
        dopl1 = []
        dopl2 = []
        for a in red:
            P = e.interpVarMJD('SitePressure')(a.mjd)
            # for AT3-G2-DL2(E)
            dist_G2 = a.A1L + a.var_mjd['OPL1']#.mean()

            T_G2   = (e.interpVarMJD('SiteTemp2m')(a.mjd)+
                      e.interpVarMJD('VLTItempSens6')(a.mjd)+
                      e.interpVarMJD('VLTI_HSS_TEMP2')(a.mjd)+
                      e.interpVarMJD('VLTI_HSS_TEMP4')(a.mjd)+
                      e.interpVarMJD('VLTI_HSS_TEMP2')(a.mjd)+
                      e.interpVarMJD('VLTItempSens5')(a.mjd))/6.+272

            dopl_G2   = prima.n_air_P_T(1.0, P, T=T_G2)*dist_G2
            dopl_G2_0 = prima.n_air_P_T(1.0, P0, T0)*dist_G2

            # for AT4-J2-DL4(W)
            dist_J2 = a.A2L + a.var_mjd['OPL2']#.mean()
            T_J2   = (e.interpVarMJD('SiteTemp2m')(a.mjd)+
                      e.interpVarMJD('VLTItempSens16')(a.mjd)+
                      e.interpVarMJD('VLTI_HSS_TEMP2')(a.mjd)+
                      e.interpVarMJD('VLTI_HSS_TEMP1')(a.mjd)+
                      e.interpVarMJD('VLTI_HSS_TEMP2')(a.mjd)+
                      e.interpVarMJD('VLTItempSens5')(a.mjd))/6.+272
            dopl_J2  = prima.n_air_P_T(1.0, P, T=T_J2)*dist_J2
            dopl_J2_0 = prima.n_air_P_T(1.0, P0, T0)*dist_J2

            dopl1.append(dopl_G2 - dopl_G2_0)
            dopl2.append(dopl_J2 - dopl_J2_0)

    # build observations tables for fitting
    base=[]
    PA  =[]
    swap=[]
    d_al=[]
    d_al_err=[]
    parang = []
    rot1 = []
    rot2 = []
    # correct for PRIMET glitches: OBSOLETE!
    #for a in red:
    #    a.d_al -= correctForPrimetJumps_JSa(a.mjd, a.jump)

    # add offset
    if not arbitrary_offsets==None:
        for k in  range(len(red)):
            red[k].d_al += arbitrary_offsets[k]*1e-6

    # default: use all observations
    if fit_only_files==None:
        fit_only_files = range(len(red))

    # all the variables, in separate lists... not very elegant
    for k in fit_only_files:
        base.extend(list(red[k].var_mjd['base']))
        PA.extend(list(red[k].var_mjd['PA']))
        swap.extend(list(red[k].swapped))
        d_al.extend(list(red[k].d_al))
        d_al_err.extend(list(red[k].d_al_err))
        parang.extend(list(red[k].var_mjd['parang']))
        rot1.extend(list(red[k].rot1))
        rot2.extend(list(red[k].rot2))

    base     = np.array(base)
    PA       = np.array(PA)
    swap     = np.array(swap)
    d_al     = np.array(d_al)
    d_al_err = np.array(d_al_err)
    parang   = np.array(parang)
    rot1     = np.array(rot1)
    rot2     = np.array(rot2)
    d_al_err = np.maximum(d_al_err, 1e-7)

    if shuffle:
        if isinstance(shuffle, int) and shuffle>1:
            np.random.seed(shuffle)
        wfit = np.random.randint(len(base), size=len(base))
    else:
        wfit = range(len(base))
    rot1 = 0 # force using parallactic
    if np.any(rot1): # we use rotators positions:
        #print 'USING rotator recordings'
        obs = [base[wfit], PA[wfit], swap[wfit], rot1[wfit], rot2[wfit],
               d_al[wfit], d_al_err[wfit]]
        if first_guess == None:
            param    = [35, 40.0, .015,0,0,0,0]
            fit_param= [1,  1,    1   ,1,1,1,1]
        else:
            param = first_guess
    else: # we use parallactic angle:
        #print 'USING parallactic angle'
        obs = [base[wfit], PA[wfit], swap[wfit], parang[wfit],
               d_al[wfit], d_al_err[wfit]]
        if first_guess == None:
            param    = [35, 40.0, .015,0,0]
            fit_param= [1,  1,    1   ,1,1]
        else:
            param = first_guess
    #print 'PARAMS:', param
    #print 'FIT_PARAM:',fit_param
    # --------- least square fit of the separation vector ---------------
    # prepare the fit
    w_fit  = np.where(np.array(fit_param))
    w_nfit = np.where(1-np.array(fit_param))

    p_fit = np.array(param)[w_fit]
    if not len(p_fit)==len(param):
        p_fixed = np.array(param)[w_nfit]
    else:
        p_fixed = []

    # fit
    if np.sum(fit_param)>0:
        plsq, cov, info, mesg, ier = \
              scipy.optimize.leastsq(leastsqAstromSepSimple, p_fit,
                                     args=(fit_param,p_fixed,obs,),
                                     epsfcn=1e-5, ftol=0.005,
                                     full_output=True)
    else:
        plsq = []

    # rebuild the complete parameter vector
    best = 1.0*np.array(param)
    best[w_fit] = plsq
    if best[0]<0:
        best[0] = abs(best[0])
        best[1] += 180

    chi2 = leastsqAstromSepSimple(best,None,None,obs)**2
    chi2_red = np.sum(chi2)/len(chi2)
    s = chi2.argsort()
    chi2_99pc = np.sum(chi2[s[:int(0.99*len(s))]])/(0.99*len(chi2))
    chi2_95pc = np.sum(chi2[s[:int(0.95*len(s))]])/(0.95*len(chi2))
    chi2_90pc = np.sum(chi2[s[:int(0.90*len(s))]])/(0.90*len(chi2))

    # build error vector
    err = np.zeros(len(param))
    if np.sum(p_fit)>0:
        err[w_fit] = np.sqrt(np.abs(np.diag(cov)*np.sum(chi2)))

    vars_  = ['sep', 'PA ', 'M0 ', 'cos', 'phi', 'cos', 'phi']
    units_ = ['\"',  'deg', 'mm ',  'um/\"', 'rad', 'um/\"', 'rad']
    if not quiet:
        print '+------------+-----------------------+'
        for k in range(len(err)):
            print '| %3s (%4s) | %10.5f +- %7.5f |' %\
                  (vars_[k], units_[k], best[k], err[k])
        print '+------------+-----------------------+'
        print 'PA baseline:', min([x.var_start_end['PA'][0] for x in red]), '->',\
              max([x.var_start_end['PA'][1] for x in red])
        print 'proj baseline:', min([x.var_start_end['base'][0] for x in red]), '->',\
              max([x.var_start_end['base'][1] for x in red])
        print 'CHI2_RED  =', round(chi2_red,2)
        print 'CHI2_[99, 95, 90]%%= [%5.2f, %5.2f, %5.2f]' % \
              (round(chi2_99pc,2), round(chi2_95pc,2) , round(chi2_90pc,2))
        print 'correlation between PA and sep=', cov[0,1]/np.sqrt(cov[0,0]*cov[1,1])
    else:
        return {'param':best, 'err':err, 'vars':vars_[:len(err)],
                'chi2':chi2_red, 'units':units_[:len(err)],
                'B_length':(min([x.var_start_end['base'][0] for x in red]),
                            max([x.var_start_end['base'][0] for x in red])),
                'B_PA':(min([x.var_start_end['PA'][0] for x in red]),
                        max([x.var_start_end['PA'][0] for x in red])),
                'target':red[0].PS_ID+'-'+red[0].SS_ID,
                'MJD-OBS':(min([a.mjd_PCR_start for a in red]), max([a.mjd_PCR_end for a in red]))}

    if res2fits:
        ### export the residuals to the fit as FITS file
        ### compute the residuals:
        mjd_min = []
        mjd_max = []
        all_res = []
        all_mjd = []
        all_lst = []
        all_proj = []
        all_pa = []
        all_parang=[]
        fitted_res = []
        flip_sign = 1.0 # put to -1 to flip sign, 1 otherwise
        for k, a in enumerate(red): # for each files (stored in red):
            mjd_min.append(a.mjd.min())
            mjd_max.append(a.mjd.max())
            all_lst.extend(list(a.lst))
            all_mjd.extend(list(a.mjd))
            UV = prima.projBaseline(red[0].baseXYZ, red[0].radec, a.lst)
            baseB = UV['B']
            all_proj.extend(list(baseB))
            basePA = UV['PA']
            all_pa.extend(list(basePA))
            parang=UV['parang']
            all_parang.extend(list(parang))
            all_res.extend(1e6*(flip_sign)**a.swapped*
                                        (a.d_al-
                                         astromSepSimple([baseB, basePA,
                                                          a.swapped, parang],
                                                         best)))
        # create fits files
        hdu = pyfits.PrimaryHDU(None)
        # list of keywords to propagate from first element of red
        list_of_kw = ['OBJECT', 'RA', 'DEC', 'EQUINOX']
        for k in list_of_kw:
            hdu.header.update(k, red[0].data[0].header[k])
        hdu.header.update('MJDSTART', min(mjd_min))
        hdu.header.update('MJDEND', min(mjd_max))
        print best
        hdu.header.update('FIT_SEP', best[0], 'binary separation in arcsec')
        hdu.header.update('FIT_PA', best[1], 'binary angle in deg')
        hdu.header.update('FIT_ZERO', best[2], 'metrology zero point, in mm')
        for k, a in enumerate(red):
            hdu.header.update('ORIF'+str(k), red[k].data[0].header['ORIGFILE'])
            hdu.header.update('ARCF'+str(k), red[k].data[0].header['ARCFILE'])
        cols = []
        cols.append(pyfits.Column(name='MJD', format='F12.5',
                                  array=all_mjd))
        cols.append(pyfits.Column(name='LST', format='F11.8',
                                  array=all_lst, unit='h'))
        cols.append(pyfits.Column(name='PROJ_BASE', format='F8.4',
                                  array=all_proj, unit='m'))
        cols.append(pyfits.Column(name='PA_BASE', format='F8.4',
                                  array=all_pa, unit='deg'))
        cols.append(pyfits.Column(name='residuals', format='F8.3',
                                  array=all_res, unit='um'))
        hducols = pyfits.ColDefs(cols)
        hdub = pyfits.new_table(hducols)
        hdub.header.update('EXTNAME', 'ASTROM_FIT_RESIDUALS', '')
        thdulist = pyfits.HDUList([hdu, hdub])
        print '--- saving residuals.fits ---'
        thdulist.writeto('residuals.fits')
        # done!

    if saveCalibrated:
        for a in red:
            #Create new file
            cols = [a.data['ASTROMETRY_BINNED'].columns[k] for
                    k in len(a.data['ASTROMETRY_BINNED'].columns)]
            cols.append(pyfits.Column(name='D_AL_ZEROED', format='F12.9',
                                      array=a.data['ASTROMETRY_BINNED'].data.field['D_AL']-
                                      best[2]*1e-3))

    if plot:
        ### x in MJD
        x = np.linspace(min([a.mjd_PCR_start for a in red])-0.003,
                           max([a.mjd_PCR_end for a in red])+0.003,
                           300)
        f = pyplot.figure(0, figsize=(10,7))
        pyplot.clf()
        f.subplots_adjust(hspace=0.05, top=0.95, left=0.13,
                          right=0.98, bottom=0.1)

        ### metrology measurements ####################
        ax1 = pyplot.subplot(211)
        pyplot.title(red[0].PS_ID+'-'+red[0].SS_ID+' (MJD '+
                     str(round(min([a.mjd_PCR_start for a in red]), 3))+
                     ' -> '+
                     str(round(max([a.mjd_PCR_end for a in red]), 3))+')')
        pyplot.setp(ax1.get_xticklabels(), visible=False)
        pyplot.ylabel('PRIMET swapping (mm)')
        # zero point:
        pyplot.hlines(best[2],
                      np.array([a.mjd_PCR_start for a in red]).min()-0.003,
                      np.array([a.mjd_PCR_end for a in red]).max()+0.003,
                      color='g', linestyle='dotted', linewidth=2)
        # model
        UV     = prima.projBaseline(red[0].baseXYZ, red[0].radec, mjd2lst(x))
        baseB  = UV['B']
        basePA = UV['PA']
        parang = UV['parang']
        pyplot.plot(x, 1e3*astromSepSimple([baseB, basePA,
                                            np.zeros(len(x)), parang], best),
                    linestyle='-', linewidth=2, color='y')
        pyplot.plot(x, 1e3*astromSepSimple([baseB, basePA,
                                            np.ones(len(x)), parang], best),
                    linestyle='--', linewidth=2, color='c')
        # measurements
        for k, a in enumerate(red):
            if (k==np.array(fit_only_files)).max():
                style = 'pg' # fitted
            else:
                style = '+r' # not fitted
            #pyplot.errorbar(a.mjd, a.d_al, yerr=a.d_al_err,
            #                color='k', marker=None)
            pyplot.plot(a.mjd, 1e3*a.d_al, style, alpha=0.1)

        ### residuals*(-1)**swapped to model ###################
        flip_sign = -1.0 # put to -1 to flip sign, 1 otherwise
        #flip_sign = 1
        pyplot.subplot(212, sharex=ax1)
        yl = r'residuals '
        if flip_sign == -1:
            yl = yl+' flipped '
        yl = yl+'($\mu$m)'
        pyplot.ylabel(yl)
        # 0-line
        pyplot.hlines([0],
                      np.array([a.mjd_PCR_start for a in red]).min()-0.003,
                      np.array([a.mjd_PCR_end for a in red]).max()+0.003,
                      color='0.5', linewidth=2, linestyle='-')

        ########## compute residuals: #####################
        # data points
        mjd_avg = []
        res_avg = []
        b_lab_f = True
        b_lab_u = True
        # for each reduced file:
        all_res = []
        fitted_res = []


        for k, a in enumerate(red):
            # for each reduced obsering block
            mjd_avg.append(a.mjd.mean())
            if np.any(rot1):
                bam = [a.var_mjd['base'], a.var_mjd['PA'],
                       a.swapped, a.rot1, a.rot2]
            else:
                bam = [a.var_mjd['base'], a.var_mjd['PA'],
                        a.swapped, a.var_mjd['parang']]
            all_res.extend(1e6*(flip_sign)**a.swapped*
                                        (a.d_al-astromSepSimple(bam,best)))
            res_avg.append(np.median(1e6*(flip_sign)**a.swapped*
                                     (a.d_al-astromSepSimple(bam,best))))
            if (k==np.array(fit_only_files)).max():
                style = 'pg' # fitted
                if b_lab_f:
                    label='fitted'
                    b_lab_f=False
                else:
                    label=None
                fitted_res.extend(1e6*(flip_sign)**a.swapped*
                                  (a.d_al-astromSepSimple(bam,best)))
            else:
                style = '+r' # not fitted
                if b_lab_u:
                    label='not fitted'
                    b_lab_u=False
                else:
                    label=None
            # plot residuals for this observing block
            pyplot.plot(a.mjd, 1e6*(flip_sign)**a.swapped*
                (a.d_al-astromSepSimple(bam, best)),
                style, label=label, alpha=0.5)

        sor = np.array(fitted_res)[np.array(fitted_res).argsort()]
        pseudo_std = (sor[int(0.84*len(sor))]-sor[int(0.15*len(sor))])/2
        print 'residuals RMS=', np.array(fitted_res).std(), 'microns'
        print 'residuals pseudo RMS=', pseudo_std , 'microns'
        mjd_avg = np.array(mjd_avg)
        res_avg = np.array(res_avg)
        # 1-position per cluster
        w = np.where(np.array([a.swapped.mean()==0 for a in red]))
        s = mjd_avg[w].argsort()
        w = w[0][s]
        pyplot.plot(mjd_avg[w], res_avg[w], 'sy', linewidth=3,
                    linestyle='-', label='NORMAL', markersize=8, alpha=0.8)
        w = np.where(np.array([a.swapped.mean()==1 for a in red]))
        s = mjd_avg[w].argsort()
        w = w[0][s]
        pyplot.plot(mjd_avg[w], res_avg[w], 'dc', linewidth=3, alpha=0.8,
                    linestyle='dashed', label='SWAPPED', markersize=10)

        if not arbitrary_offsets==None:
            for k in range(len(red)):
                if k==0:
                    label='offsets'
                else:
                    label=None
                pyplot.hlines([(-1)**(red[k].swapped.mean())*arbitrary_offsets[k]],
                              red[k].mjd.min(), red[k].mjd.max(),
                              color=(0.8, 0.3, 0), linewidth=3, label=label)
        # -- plot metrology jumps
        #pyplot.plot(x, correctForPrimetJumps(x, red[0].jump*1e6),
        #            color='m', linestyle='-.', linewidth=2, label='jumps')
        pyplot.legend(loc='upper left', ncol=4, bbox_to_anchor=(.0, 1.02, 1.2, .05))
        if not maxResiduals is None:
            pyplot.ylim(-maxResiduals, maxResiduals)
        pyplot.xlabel('MJD')
        #---------- FSM plots ------------
        if FSMplots:
            pyplot.figure(2, figsize=(10,8))
            pyplot.clf()
            pyplot.subplots_adjust(top=0.95, left=0.05,
                                  right=0.98, bottom=0.05)
            for sts in [1,2]:
                for fsm in [1,2]:
                    pyplot.subplot(220+(sts-1)*2+fsm)
                    pyplot.title('STS'+str(sts)+' FSM'+str(fsm)+':pos in $\mu$m')
                    if sts==1 and fsm==1:
                        # AT3 FSM1
                        fsm_lims = [22,25,20,28]
                        fsm_p = [10.5, -0.285, 10.5, 0.446]
                    elif sts==2 and fsm==1:
                        # AT4 FSM1
                        fsm_lims = [25,28,20,28]
                        #fsm_p = [20, -0.491, 20, 0.315]
                        fsm_p = [20, -0.025, 20, -0.210]
                    elif sts==1 and fsm==2:
                        # AT3 FSM2
                        fsm_lims = [6,9,7,15]
                        fsm_p = [6.5, -0.046, 10.5, -0.063]
                    elif sts==2 and fsm==2:
                        # AT4 FSM2
                        fsm_lims = [8.5,11.5,7,15]
                        #fsm_p = [10, -0.025, 10, -0.210]
                        fsm_p = [10, -0.491, 10, 0.315]

                    xp, yp = np.meshgrid(np.linspace(fsm_lims[0],fsm_lims[1],100),
                                         np.linspace(fsm_lims[2],fsm_lims[3],100))
                    dopd = (xp-fsm_p[0])*fsm_p[1] + (yp-fsm_p[2])*fsm_p[3]
                    print 'dopd:', dopd.min(), dopd.max()
                    pyplot.pcolormesh(xp,yp, dopd, vmin=-2, vmax=5)
                    pyplot.colorbar()
                    for x in red:
                        pyplot.plot([x.data[0].header['ESO ISS PRI STS'+str(sts)+
                                                      ' FSMPOSX'+str(fsm)+' START'],
                                    x.data[0].header['ESO ISS PRI STS'+str(sts)+
                                                     ' FSMPOSX'+str(fsm)+' END']],
                                    [x.data[0].header['ESO ISS PRI STS'+str(sts)+\
                                                      ' FSMPOSY'+str(fsm)+' START'],
                                     x.data[0].header['ESO ISS PRI STS'+str(sts)+\
                                                      ' FSMPOSY'+str(fsm)+' END']],
                                    ('sy-' if x.insmode=='NORMAL' else 'dc-'),
                                    markersize=14)

        return # skip additional plots
        pyplot.figure(2, figsize=(4,4))
        pyplot.clf()
        pyplot.subplot(111, polar=True)
        sortmjd = np.array([x.mjd_PCR_start for x in red]).argsort()
        PAmin = [x.var_start_end['PA'][0] for x in red][sortmjd[0]]
        PAmax = [x.var_start_end['PA'][1] for x in red][sortmjd[-1]]

        pyplot.bar(PAmin*np.pi/180., 1,
                   width =(PAmax-PAmin)*np.pi/180.,
                   color='red', alpha=0.5,
                   label='baseline')
        pyplot.bar(PAmin*np.pi/180.+np.pi, 1,
                   width =(PAmax-PAmin)*np.pi/180.,
                   color='red', alpha=0.5)

        pyplot.plot([best[1]*np.pi/180,best[1]*np.pi/180], [0,1], color='k',
                    alpha=0.8, linewidth=5, label='binary')
        pyplot.legend()

        #return #--------------------------------------------------
        f = pyplot.figure(1, figsize=(14,10))
        pyplot.clf()
        f.subplots_adjust(hspace=0.04, top=0.97, left=0.1,
                          right=0.98, bottom=0.05)

        #### polynomial interpolations ##################
        n = len(var_poly_c.keys())
        ### x in MJD
        x = np.linspace(min([a.mjd_PCR_start for a in red])-0.003,
                           max([a.mjd_PCR_end for a in red])+0.003,
                           300)
        for i,k in enumerate(var_poly_c.keys()):
            if i==0:
                ax0 = pyplot.subplot(n,2,1+2*i)
                pyplot.title('interpolation')
                pyplot.setp(ax0.get_xticklabels(), visible=False)
            else:
                ax1 = pyplot.subplot(n,2,1+2*i, sharex=ax0)
            if not i==(n-1):
                pyplot.setp(ax1.get_xticklabels(), visible=False)
            else:
                pyplot.xlabel('MJD')
            pyplot.ylabel(k)

            if len(var_poly_c[k])>1:
                pyplot.plot(x, np.polyval(var_poly_c[k], x-X.mean()), 'k-')
            for a in red:
                if k=='base' or k=='PA':
                    pyplot.plot(a.mjd, a.var_mjd[k], '.r')
                else:
                    pyplot.plot(a.mjd, a.var_mjd[k], '.y')
                pyplot.plot(a.mjd_PCR_start, a.var_start_end[k][0], 'oy')
                pyplot.plot(a.mjd_PCR_end, a.var_start_end[k][1], 'oy')

            # residuals
            ax2 = pyplot.subplot(n,2,2+2*i, sharex=ax0)
            if not i==(n-1):
                pyplot.setp(ax2.get_xticklabels(), visible=False)
            else:
                pyplot.xlabel('MJD')

            if i==0:
                pyplot.title('residual')

            pyplot.hlines([0],
                          np.array([a.mjd_PCR_start for a in red]).min()-0.003,
                          np.array([a.mjd_PCR_end for a in red]).max()+0.003,
                          color='k')
            if len(var_poly_c[k])>1:
                for a in red:
                    pyplot.plot([a.mjd_PCR_start,a.mjd_PCR_end],
                                [a.var_start_end[k][0]-
                                 np.polyval(var_poly_c[k],
                                               a.mjd_PCR_start-X.mean()),
                                 a.var_start_end[k][1]-
                                 np.polyval(var_poly_c[k],
                                               a.mjd_PCR_end-X.mean())],
                                'oy-')

    return

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

def bootstrapListOfFiles(files, directory, fit_param, first_guess, N=500,
                         maxResidualsMas=None, plot=False, multi=False):
    res = []
    if multi:
        import multiprocessing
        p = Pool()
        cb_boot(None, init=True)
    for k in range(N):
        if multi:
            seed = np.random.randint(1e9)
            p.apply_async(interpListOfFiles, ((files, directory,
                                               first_guess, fit_param, seed)),
                             callback=cb_boot)
        else:
            if k%100 == 0:
                print k, '/', N, '...'
            res.append(interpListOfFiles(files, directory,
                                         first_guess, fit_param, True))
    if multi:
        p.close()
        p.join()
        res = cb_boot(None, ret=True)

    X = np.array([z['param'][0]*np.cos(z['param'][1]*np.pi/180) for z in res])
    Y = np.array([z['param'][0]*np.sin(z['param'][1]*np.pi/180) for z in res])
    p = pca.pca(np.transpose(np.array([1e3*(X-X.mean()),
                                       1e3*(Y-Y.mean())])))
    print 'PCA matrix (RA, dec):', p.base
    err0 = p.coef[:,0].std()
    err1 = p.coef[:,1].std()
    print 'delta RA  =', round(1000*Y.mean(), 4), 'mas'
    print 'delta dec =', round(1000*X.mean(), 4), 'mas'
    print 'error bar large axis', round(err0*1000.,1), 'uas'
    print 'error bar small axis', round(err1*1000.,1), 'uas'

    result={'MJD':(res[0]['MJD-OBS'][0]+res[0]['MJD-OBS'][1])/2.,
            'RA_mas':1000*Y.mean(), 'DEC_mas':1000*X.mean(),
            'PCA':(list(p.base[0]),list(p.base[0])) ,
            'errs_uas':(err0*1000, err1*1000)}

    if plot:
        pyplot.figure(0, figsize=(6,5))
        pyplot.clf()
        pyplot.axes().set_aspect('equal', 'datalim')
        pyplot.plot(1000*(X-np.array(X).mean()),
                    1000*(Y-np.array(Y).mean()),'ok',
                    alpha=0.2, label='bootstrapping')
        pyplot.xlabel('$\delta$ RA (mas)')
        pyplot.ylabel('$\delta$ DEC (mas)')
        pyplot.plot([-p.base[0,0]*err0, p.base[0,0]*err0],
                    [-p.base[1,0]*err0, p.base[1,0]*err0],
                    linewidth=3, color='r',
                    label='$\pm$'+str(int(err0*1000))+'$\mu$as')
        pyplot.plot([-p.base[0,1]*err1, p.base[0,1]*err1],
                    [-p.base[1,1]*err1, p.base[1,1]*err1],
                    linewidth=3, color='b',
                    label='$\pm$'+str(int(err1*1000))+'$\mu$as')
        pyplot.legend(ncol=3)
        pyplot.title(res[0]['target']+' MJD '+
                     str(round(res[0]['MJD-OBS'][0],3))+' '+
                     str(round(res[1]['MJD-OBS'][1],3)))
        if not maxResidualsMas is None:
            pyplot.xlim(-maxResidualsMas, maxResidualsMas)
            pyplot.ylim(-maxResidualsMas, maxResidualsMas)
    return result

def doAll(files, directory='', reduceData=False, bootstrap=False,
          selection=None, maxResiduals=None, pssRec=None,
          firstGuess=[10.,90., 0.0], fittedParam=[1,1,1],
          res2fits=False, max_GD_um=5.0, sigma_clipping=3.5,
          max_err_um=2, max_length_s=1.0):
    """
    astrometric data reduction and fit of a list of files. bootstrap=True also
    provides a statistical estimate of the error bar in the separation vector.
    """
    # -- data reduction
    if reduceData:
        for f in files:
            print '#'*3, 'reducing:', f, '#'*5
            a = prima.drs(directory+f, pssRec=pssRec)
            if 'OBJ_ASTRO' in f: # reduce dual fringe tracking data
                a.astrometryFtk(max_err_um=max_err_um, max_GD_um=max_GD_um,
                                sigma_clipping=sigma_clipping,
                                max_length_s=max_length_s, writeOut=True,
                                overwrite=True)
            else: # reduce scanning data:
                a.asrtrometryScan(max_err_um=max_err_um, max_GD_um=max_GD_um,
                                sigma_clipping=sigma_clipping,
                                     writeOut=True,
                                overwrite=True)
            a.raw.close()
            a = ""

    # -- data fitting (astrometry)
    if selection is None:
        selection = range(len(files))
    if bootstrap: # boostrapped error bar
        files = [files[k] for k in selection]
        t0 = time.time()
        res = bootstrapListOfFiles(files, directory, N=500,
                                          fit_param  =fittedParam, plot=True,
                                          first_guess=firstGuess,
                                          maxResidualsMas=maxResiduals,
                                          multi=True)
        print 'bootstraping performed in', round(time.time()-t0, 1), 's'
        return res
    else: # simple leastsquare fit of thge separation vector
        interpListOfFiles(files, directory,
                                 plot=True, quiet=False,
                                 fit_param  = fittedParam,
                                 first_guess= firstGuess,
                                 fit_only_files=selection,
                                 maxResiduals=maxResiduals, res2fits=res2fits)
        res = []
    return

class atom():
    """
    atomic observation (one file). reads and load one reduced fits
    file. nothing else so far
    """
    def __init__(self, filename, debug=False):
        he = 'HIERARCH ESO '
        self.data = pyfits.open(filename)
        self.date_obs = self.data[0].header['DATE-OBS']
        self.targname = self.data[0].header['HIERARCH ESO OBS TARG NAME']
        self.PS_ID = self.data[0].header['HIERARCH ESO OCS PS ID']
        self.SS_ID = self.data[0].header['HIERARCH ESO OCS SS ID']
        self.mjd_obs = self.data[0].header['MJD-OBS']
        self.lst_obs = self.data[0].header['LST']/3600.
        self.mjd= self.data['ASTROMETRY_BINNED'].data.field('MJD')
        # simple linear interpolation
        self.lst = self.lst_obs + (self.mjd - self.mjd_obs)*(23.+56/60.+4/3600.)

        self.d_al= self.data['ASTROMETRY_BINNED'].data.field('D_AL')
        self.d_al_err= self.data['ASTROMETRY_BINNED'].data.field('D_AL_err')
        try:
            self.rot1= self.data['ASTROMETRY_BINNED'].data.field('ROT1')
            self.rot2= self.data['ASTROMETRY_BINNED'].data.field('ROT2')
        except:
            self.rot1= np.zeros(len(self.d_al))
            self.rot2= np.zeros(len(self.d_al))

        self.mjd_PCR_start  = astro.tag2mjd(self.data[0].header[he+'PCR ACQ START'])
        self.mjd_PCR_end    = astro.tag2mjd(self.data[0].header[he+'PCR ACQ END'])
        self.lst_PCR_start = astro.tag2lst(self.data[0].header['ESO PCR ACQ START'],
                                           longitude=self.data[0].header['ESO ISS GEOLON'])
        self.A1L        = self.data[0].header[he+'ISS CONF A1L']
        self.A2L        = self.data[0].header[he+'ISS CONF A2L']
        try:
            self.insmode = self.data[0].header[he+'ISS PRI STS3 GUIDE_MODE']
        except:
            self.insmode    = self.data[0].header[he+'INS MODE']

        cP  = self.data[0].header[he+'ISS PRI MET C'] # in m/s
        dnuP = self.data[0].header[he+'ISS PRI MET F_SHIFT']*1e6 # in Hz
        nuP  = self.data[0].header[he+'ISS PRI MET LASER_F']
        #self.jump =  (cP * dnuP/2/(nuP**2) ) * ( 2**24-1 ) # PRIMET jump in m, COMM14
        self.jump =  (cP * dnuP/2/(nuP**2) ) * ( 2**31-1 ) # PRIMET jump in m, COMM15

        if 'SWAP' in self.insmode: # legacy: was SWAP at some point, now is SWAPPED
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
            self.var_start_end[names[k]] =(self.data[0].header[he+keyw[k]+' START'],
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
                self.radec = [astro.ESOcoord2decimal(self.data[0].header['ESO OCS TARG1 ALPHAPMC']),
                              astro.ESOcoord2decimal(self.data[0].header['ESO OCS TARG2 DELTAPMC'])]
                self.radec = [self.data[0].header['RA']/15.,
                              self.data[0].header['DEC']]
                bb = prima.projBaseline(self.baseXYZ, self.radec, self.lst)
                if k == 'base':
                    self.var_mjd[k] = bb['B']
                    if debug:
                        print 'DEBUG: PBL12 START: computed', self.var_mjd[k][0],\
                              'in header:', self.data[0].header[he+'ISS PBL12 START'], \
                              'DELTA=', self.var_mjd[k][0]-self.data[0].header[he+'ISS PBL12 START']
                        print 'DEBUG: PBL12 END  : computed', self.var_mjd[k][-1],\
                              'in header:', self.data[0].header[he+'ISS PBL12 END'], \
                              'DELTA=', self.var_mjd[k][-1]-self.data[0].header[he+'ISS PBL12 END']
                if k == 'PA':
                    self.var_mjd[k] = bb['PA']
                    if debug:
                        print 'DEBUG: PBLA12 START: computed', self.var_mjd[k][0],\
                              'in header:', self.data[0].header[he+'ISS PBLA12 START'],\
                              'DELTA=', self.var_mjd[k][0]-self.data[0].header[he+'ISS PBLA12 START']
                        print 'DEBUG: PBLA12 END  : computed', self.var_mjd[k][-1],\
                              'in header:', self.data[0].header[he+'ISS PBLA12 END'],\
                              'DELTA=', self.var_mjd[k][-1]-self.data[0].header[he+'ISS PBLA12 END']
        self.data.close()
        return

### simulation code ###

def simu3():
    """
    what if the baseline is different for SS and PS?

    -> for small error in baselines, the residuals are still very
       small but the separation and projection angle of the binary are
       wrong...

    """
    ra = 12.0 # in h
    dec = -32.0 # in deg
    sep = 35.0 # in arcsec
    PA = 135.0 # in deg

    dra = sep*np.cos(PA*np.pi/180.)/(3600.*180.)*np.pi/15. # in h
    ddec = sep*np.sin(PA*np.pi/180.)/(3600.*180.)*np.pi # in deg

    print 'actual binary is SEP=', sep, '(\"), PA=', PA, '(deg)'
    base = np.array([-76.4,49.862,0]) # X,Y,Z in m G2-J2
    print 'actual baseline is:', base, '(m)'
    dbase = np.array([0*100e-6,0,0.0]) # error in Baseline, in m
    lst = np.linspace(12, 16, 50) # in h
    zeroM = 1.0 # in m

    # projected baseline
    projUV = prima.projBaseline(base, [ra, dec], lst)
    projPA = np.arctan(projUV[0], projUV[1])*180/np.pi
    projB = np.sqrt( projUV[0]**2+ projUV[1]**2)

    # projected baseline with offset
    print 'error in baseline is:', dbase, '(m)'
    projUVp = prima.projBaseline(base+dbase, [ra, dec], lst)
    projPAp = np.arctan(projUVp[0], projUVp[1])*180/np.pi
    projBp = np.sqrt( projUVp[0]**2+ projUVp[1]**2)

    # astrometric observables
    dopd = zeroM + \
           (prima.projBaseline(base, [ra, dec], lst)[2] -\
            prima.projBaseline(base+dbase, [ra+dra, dec+ddec], lst)[2])
    print 'dopd PTP=',dopd.ptp()*1e6, 'um'
    # swapped
    dopds = zeroM -\
            (prima.projBaseline(base, [ra, dec], lst)[2] -\
             prima.projBaseline(base+dbase, [ra+dra, dec+ddec], lst)[2])

    # observable
    B = list(projBp)
    B.extend(list(projBp))
    PA = list(projPAp)
    PA.extend(list(projPAp))
    Sw = list(np.zeros(len(projBp)))
    Sw.extend(list(np.ones(len(projBp))))
    dO = list(dopd)
    dO.extend(dopds)

    e = list(np.ones(2*len(dopd))*1e-6)
    parang =  list(np.zeros(2*len(dopd)))
    obs = [np.array(B), np.array(PA), np.array(Sw),
           np.array(parang), np.array(dO), np.array(e)]
    # fit
    param     = [35.0, 135.0, 1.0, 0,0,0]
    fit_param = [1   , 1   , 1   , 0,0,0]
    # prepare the fit
    w_fit  = np.where(np.array(fit_param))
    w_nfit = np.where(1-np.array(fit_param))
    p_fit = np.array(param)[w_fit]
    if not len(p_fit)==len(param):
        p_fixed = np.array(param)[w_nfit]
    else:
        p_fixed = []
    # -- fit
    plsq, cov, info, mesg, ier = \
          scipy.optimize.leastsq(leastsqAstromSepSimple, p_fit,
                                 args=(fit_param,p_fixed,obs,),
                                 epsfcn=1e-5, ftol=0.005, full_output=True)
    #plsq = param
    print 'fitted binary is SEP=', plsq[0], '(\"), PA=', plsq[1], '(deg)'


    # compute model
    dopd_M = np.array([astromSepSimple([projBp[k], projPAp[k], 0],
                                          [plsq[0], plsq[1], plsq[2]])
                          for k in range(len(projBp))])
    dopds_M = np.array([astromSepSimple([projBp[k], projPAp[k]+180, 1],
                                            [plsq[0], plsq[1], plsq[2]])
                            for k in range(len(projBp))])

    residuals = (dopd-dopds - (dopd_M-dopds_M))*1e6
    print 'peak to peak error in delta_dopdc is',\
          residuals.max()-residuals.min(), '(microns)'

    pyplot.figure(0)
    pyplot.clf()
    pyplot.subplot(211)
    pyplot.plot(lst, dopd-dopds, '.k', label='data')
    pyplot.plot(lst, dopd_M-dopds_M, 'k-', label='model')
    pyplot.xlabel('LST (h)')
    pyplot.ylabel('direct - swapped (m)')
    pyplot.legend()
    pyplot.subplot(212)
    pyplot.plot(lst, (dopd-dopds - (dopd_M-dopds_M))*1e6, '.k')
    pyplot.xlabel('LST (h)')
    pyplot.ylabel(r'O-C ($\mu$m)')
    return
