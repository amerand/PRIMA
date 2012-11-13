import pyfits

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
from scipy.interpolate import interp1d
from scipy.stats import poisson
#from scipy import weave
#from scipy.weave import converters
import scipy.signal
import numpy as np
import os

# --- my modules ---
import morlet #
import slidop #
import myfit  #
import dpfit
import pca    #
#import paranal#
import astro

#plt.rc('font', family='monospace', weight='book', size=10)

def history():
    return """
    - 2011/08/09: added correction from bug found by Nicolas

    - 2011/07 COMM15: some keywords changed during COMM (name of
      column in binary table). This version uses np.interp and
      np.polyfit instead of the scipy versions

    - 2011/02 lots of new functions

    - 2011/01 moved to faster CWT; some cleaning (AMe)

    - 2010/11 first implementation (AMe)
    """

def onedgauss(x,H,A,dx,w):
    """
    Returns a 1-dimensional gaussian of form
    H,A,dx,w = params
    H+A*np.exp(-(x-dx)**2/(2*w**2))
    """
    #H,A,dx,w = params
    return H+A*np.exp(-(x-dx)**2/(2*w**2))

def fringes_morlet_phase(m1,m2, quasi_pi=False):
    """
    m1 and m2 are 2D morlet transform. axis=0 is frquency,
    axis=1 is time.
    """
    ### cross spectrum
    cross_spec = np.conj(m1.cwt)*m2.cwt
    phi = np.angle(cross_spec)
    if quasi_pi:
        phi = np.mod(phi + np.pi/2, 2*np.pi)
    weight = abs(m1.cwt)*abs(m2.cwt)
    phase = np.sum(phi*weight, axis=0)/np.sum(weight, axis=0)
    if quasi_pi:
        phase -= np.pi/2
    return phase

def fringes_ABCD_phase(opd, fringesAr, fringesBr,
                       fringesCr, fringesDr,
                       wavelength=2.1e-6, plot=False):
    """
    takes 4 fringes (ABCD) and OPD vectors and computes phases using a
    Morlet CWT
    """
    # define boundaries
    n_samples_per_fringes = wavelength/np.abs(np.diff(opd).mean())

    # compute Morlet's transform
    dopd = np.abs(np.diff(opd).mean())

    wavA = morlet.CWT(fringesAr, dopd, scale_min=1/6e-6,
                      scale_max=1/2.e-6, n_per_octave=32)
    wavB = morlet.CWT(fringesBr, dopd, scale_min=1/6e-6,
                      scale_max=1/2.e-6, n_per_octave=32)
    wavC = morlet.CWT(fringesCr, dopd, scale_min=1/6e-6,
                      scale_max=1/2.e-6, n_per_octave=32)
    wavD = morlet.CWT(fringesDr, dopd, scale_min=1/6e-6,
                      scale_max=1/2.e-6, n_per_octave=32)

    data = [wavA, wavB, wavC, wavD]
    scale=[]
    for wav in data:
        weight = np.sum(np.abs(wav.cwt)**2, axis=1)
        i0 = weight.argmax()
        i0 = min(max(i0, len(weight)-5), 5)
        print i0, len(weight)
        c = np.polyfit(wav.frequencies[i0-2:i0+2],\
                       np.log(weight[i0-2:i0+2]), 2)
        scale.append(-c[1]/(2*c[0]))
    scale = np.array(scale)
    wlABCD = 1/scale*1e6
    print 'wavelength  (WT peak) [A B C D]:'
    print wlABCD
    print 100*(scale/scale.mean()-1), '(% / mean)'
    if plot:
        opd0 = (opd-opd.mean())*1e6
        Xp, Yp = np.meshgrid(opd0, 1/wavA.frequencies*1e6)

        plt.figure(4)
        plt.clf()
        ax = plt.subplot(211)
        plt.plot(opd0, fringesAr-fringesCr, 'k')
        plt.subplot(212, sharex=ax)
        plt.plot(opd0, fringesBr-fringesDr, 'k')

        plt.figure(3)
        plt.clf()
        plt.subplot(241, sharex=ax)
        plt.plot(opd0, fringesAr, 'k')
        plt.ylabel('fringes')
        plt.title('A')
        plt.subplot(245, sharex=ax)
        plt.pcolormesh(Xp,Yp, abs(wavA.cwt))
        #plt.hlines(wlABCD[0], opd0.min(), opd0.max(),\
        #              color='y')
        plt.ylabel('wavelength (um)')
        plt.subplot(242, sharex=ax)
        plt.plot(opd0, fringesBr, 'k')
        plt.title('B')
        plt.subplot(246, sharex=ax)
        plt.pcolormesh(Xp,Yp, abs(wavB.cwt))
        #plt.hlines(wlABCD[1], opd0.min(), opd0.max(),\
        #              color='y')
        plt.subplot(243, sharex=ax)
        plt.plot(opd0, fringesCr, 'k')
        plt.title('C')
        plt.subplot(247, sharex=ax)
        plt.pcolormesh(Xp,Yp, abs(wavC.cwt))
        #plt.hlines(wlABCD[2], opd0.min(), opd0.max(),\
        #              color='y')
        plt.subplot(244, sharex=ax)
        plt.plot(opd0, fringesDr, 'k')
        plt.title('D')
        plt.subplot(248, sharex=ax)
        plt.pcolormesh(Xp,Yp, abs(wavD.cwt))
        #plt.hlines(wlABCD[3], opd0.min(), opd0.max(),\
        #              color='y')

    # compute phases as function of OPD
    phiAB = fringes_morlet_phase(wavA, wavB)
    phiAC = fringes_morlet_phase(wavA, wavC, quasi_pi=True)
    phiDA = fringes_morlet_phase(wavD, wavA)
    phiBC = fringes_morlet_phase(wavB, wavC)
    phiBD = fringes_morlet_phase(wavB, wavD, quasi_pi=True)
    phiCD = fringes_morlet_phase(wavC, wavD)
    weightABCD = np.sum(abs(wavA.cwt)*abs(wavB.cwt)*\
                           abs(wavC.cwt)*abs(wavD.cwt),axis=0)
    weightABCD = weightABCD**(0.25)
    return phiAB, phiBC, phiCD, phiDA, phiAC, phiBD, weightABCD, wlABCD

    wp = np.where(weightABCD>
                  3*np.median(weightABCD))
    print 'WP=', wp
    phi_abcd = np.arctan2(fringesAr-fringesCr, \
                             fringesBr-fringesDr)
    phi_abcd = np.unwrap(phi_abcd)
    slopeAB = np.polyfit((opd[wp]-opd[wp].mean())*1e6,\
                         phiAB[wp], 1)
    slopeBC = np.polyfit((opd[wp]-opd[wp].mean())*1e6,\
                         phiBC[wp], 1)
    slopeCD = np.polyfit((opd[wp]-opd[wp].mean())*1e6,\
                         phiCD[wp], 1)
    slopeDA = np.polyfit((opd[wp]-opd[wp].mean())*1e6,\
                         phiDA[wp], 1)
    slopeAC = np.polyfit((opd[wp]-opd[wp].mean())*1e6,\
                         phiAC[wp], 1)
    slopeBD = np.polyfit((opd[wp]-opd[wp].mean())*1e6,\
                         phiBD[wp], 1)
    print 'phases: slope (rad/um of OPD), phi0/pi'
    print 'A-B %6.3f %6.3f' % (slopeAB[0], slopeAB[1]/np.pi)
    print 'B-C %6.3f %6.3f' % (slopeBC[0], slopeBC[1]/np.pi)
    print 'C-D %6.3f %6.3f' % (slopeCD[0], slopeCD[1]/np.pi)
    print 'D-A %6.3f %6.3f' % (slopeDA[0], slopeDA[1]/np.pi)
    slopes = [slopeAB[0], slopeBC[0], slopeCD[0], slopeDA[0]]
    slopes = np.array(slopes)
    wldiff = [wlABCD[0]-wlABCD[1], wlABCD[1]-wlABCD[2], \
              wlABCD[2]-wlABCD[3], wlABCD[3]-wlABCD[0]]
    wldiff = np.array(wldiff)
    #------- FIGURE ----------
    if plot:
        plt.figure(1)
        plt.clf()
        plt.subplot(211)
        plt.plot(opd, fringesBr, 'k')
        plt.subplot(212)
        plt.hlines(np.pi*np.arange(-1,1.5,0.5), \
                      opd.min(), opd.max(), color='y',\
                      linestyles='dotted')
        plt.plot(opd[wp], phiAB[wp], 'r', linewidth=2)
        plt.plot(opd[wp], phiBC[wp], 'g', linewidth=2)
        plt.plot(opd[wp], phiCD[wp], 'b', linewidth=2)
        plt.plot(opd[wp], phiDA[wp], 'm', linewidth=2)
        plt.plot(opd[wp], np.unwrap(phiAC[wp]),\
                    color=((1,0.5,0,0)), linewidth=2)
        plt.plot(opd[wp], np.unwrap(phiBD[wp]),\
                    color=((0.3,0.2,1,0)), linewidth=2)
        plt.ylabel('phase (radian)')
        plt.xlabel('OPD (m)')
        plt.legend(('A-B', 'B-C', 'C-D', 'D-A',\
                       'A-C','B-D'))
        plt.ylim(0, 4)
    return phiAB, phiBC, phiCD, phiDA, phiAC, phiBD, weightABCD, wlABCD

def n_air_P_T(wl, P=743.2, T=290, e=74.32):
    """
    wl in um
    P, in mbar (default 743.2mbar)
    T, in K    (default 290.0K)
    e partial pressure of water vapour, in mbar (default 74.32)
    """
    return 1 + 1e-6*(1+0.00752/np.array(wl)**2)*\
           (77.6*np.array(P)/np.array(T)
                             + 3.73e-5*e/np.array(T)**2)

def NoisySlope(x,y, dx, plot=None):
    """
    try to determine the 'noisy slope' of y(x), where y is noisy AND
    get jumps and heratic behaviors. dx is the (small) range for which
    """
    # compute slopes
    slopes = []
    synth = np.zeros(len(x))
    x0 = x[0]
    xmax = x.max()

    # default slope
    def_slope = (y[-1]-y[0])/(x[-1]-x[0])
    while x0+dx<xmax:
        w = np.where((x>=x0)*(x<=x0+dx))
        if len(w[0])>10:
            coef = np.polyfit(x[w]-x[w].mean(), y[w], 1)
            slopes.append(coef[0])
            synth[w] = np.polyval(coef, x[w]-x[w].mean())
        else:
            if len(w[0])>0:
                synth[w] = y[w][0]+def_slope*(x[w]-x[w][0])
        x0 += dx

    # sigma clipping
    slopes = np.array(slopes)
    t = np.linspace(slopes.min(), slopes.max(), 500)
    s = slopes.argsort()
    slopes = slopes[s]
    r = slopes[int(len(slopes)*.7)]-slopes[int(len(slopes)*.3)]
    w = np.where((slopes>=(np.median(slopes)-2*r))*\
                    (slopes<=(np.median(slopes)+2*r)))
    guess = [0.0, 0.0, np.median(slopes), r]
    best = myfit.fit(myfit.erf, slopes[w], guess, \
                     np.linspace(0,1,len(slopes))[w])
    best = best.leastsqfit()

    plt.figure(20)
    plt.clf()
    plt.plot(slopes*1e6, np.linspace(0,1,len(slopes)), 'ob')
    plt.plot(slopes[w]*1e6, np.linspace(0,1,len(slopes))[w], 'or')
    plt.plot(t*1e6, myfit.erf(t,best), 'r', linewidth=2)
    plt.xlim((best[2]-5*best[3])*1e6,\
                (best[2]+5*best[3])*1e6)
    plt.xlabel('slope (um/s)')

    return (best[2],best[3], synth)

def percolate1D(x, dx, dxmax=None):
    """
    return a table of length of x with 0,1,2,...N-1 delimiting the N
    groups x with <dx of each neighbours. If maxdx is set, will cut
    groups bigger than maxdx
    """
    g = np.zeros(len(x))
    diffx = np.diff(x)
    steps = np.int_(diffx>=dx)
    steps[0]  = 1
    steps[-1] = 1

    if not dxmax is None:
        # chop continuous sequences if their lengh is >dxmax
        w = np.where(steps==1)
        for k in range(len(w[0])-1):
            if x[w[0][k+1]]-x[w[0][k]] > dxmax:
                # number of sub-sequences
                n = int((x[w[0][k+1]]-x[w[0][k]])/dxmax)
                # chop
                for i in range(n):
                    steps[w[0][k]+
                          int(i*(w[0][k+1]-w[0][k])/float(n))]=1
    g[1:] = np.cumsum(steps)
    return g

def truncateInWindows(x,delta_x):
    """
    truncates x in packet of length delta_x
    same result as percolate
    """
    g = np.zeros(len(x))
    group = 0
    x0 = x[0]
    for k in range(len(x)):
        g[k] = group
        if x[k]-x0>delta_x:
            x0 = x[k]
            group+=1
    return g

def unwrap(x, dx):
    """
    unwrap x for steps > dx
    """
    d = np.diff(x)
    steps = np.zeros(len(x)-1)
    steps += dx*(d>dx)
    steps -= dx*(d<dx)
    corr = np.zeros(len(x))
    corr[1:] = np.cumsum(steps)
    return x-corr

def fitMax(x,y,n=2):
    """
    returns the x for which y is max. uses a polynonial fit of order 2
    around the 5 highest points. assumes x and y are sorted according
    to x.
    """
    #-- position of max
    kmax = y.argmax()

    #-- 5 points around max
    k = range(kmax-n, kmax+n)
    k = np.array(k)
    #-- slide if to close to the edges
    if k.max()>(len(y)-2):
        k -= (len(y)-k.max())+10
        #print '   ->', k
    if k.min()<0:
        #print 'CORR:', k, '(', len(y), ')'
        k -= k.min()
        #print '   ->', k
    #-- fit poly #2
    c = np.polyfit(x[k], y[k], 2)
    xmax = np.clip(-c[1]/(2*c[0]), x.min(), x.max())
    return xmax

def projBaseline(B, radec, lst, latitude=-24.62743941):
    """
    B = (Tx, Ty, Tz, A0) in m: x toward E, y toward north
    radec = (ra, dec) in (decimal hour, decimal deg)
    lst = local sidereal time in decimal hours

    see Principles of Long Baseline Interferometry, chapter 12,
    equations 12.1 and 12.2

    http://olbin.jpl.nasa.gov/iss1999/coursenotes/chapt12.pdf
    """
    # hour angle, in degrees
    ha = (np.array(lst) - radec[0])*360/24.0
    # alt-az
    dec = radec[1]
    d2r = lambda x: x*np.pi/180
    tmp1 =  np.sin(d2r(dec)) * np.sin(d2r(latitude)) +\
           np.cos(d2r(dec)) * np.cos(d2r(latitude)) *\
           np.cos(d2r(ha))
    alt = np.arcsin(tmp1)*180/np.pi

    tmp1 = np.cos(d2r(dec)) * np.sin(d2r(ha))
    tmp2 = -np.cos(d2r(latitude)) * np.sin(d2r(dec)) + \
           np.sin(d2r(latitude)) * np.cos(d2r(dec)) * \
           np.cos(d2r(ha));
    az = (np.arctan2(tmp1, tmp2)*180/np.pi)%360
    az = 360-az # same convention as Paranal

    # parallactic angle: check in Paranal TCS
    parang = -180*np.arcsin(np.sin(d2r(az))*
                           np.cos(d2r(latitude))/np.cos(d2r(dec)))/np.pi

    # trigonometric
    ch_ = np.cos(ha*np.pi/180.)
    sh_ = np.sin(ha*np.pi/180.)
    cl_ = np.cos(latitude*np.pi/180.)
    sl_ = np.sin(latitude*np.pi/180.)
    cd_ = np.cos(radec[1]*np.pi/180.)
    sd_ = np.sin(radec[1]*np.pi/180.)
    # (u,v) coordinates in m
    u = ch_ * B[0] -  sl_*sh_ * B[1] + cl_*sh_ * B[2]
    v = sd_*sh_ * B[0] + (sl_*sd_*ch_+cl_*cd_) * B[1] -\
        (cl_*sd_*ch_ - sl_*cd_) * B[2]
    # optical delay, in m
    d = -B[0]*cd_*sh_-\
        B[1]*(sl_*cd_*ch_ - cl_*sd_)+\
        B[2]*(cl_*cd_*ch_ + sl_*sd_)
    if len(B)>3: # add offset if given
        d+= B[3]

    return {'u':u,'v':v,'opd':d,'B':np.sqrt(u**2+v**2),
            'PA':np.arctan2(u,v)*180/np.pi, 'alt':alt, 'az':az,
            'parang':parang}

def testParang(dec):
    lst = np.linspace(-12,12,100)
    pb = projBaseline([10,10,0,0], [0, dec], lst)
    X = (90-pb['alt'])/90.*np.sin(pb['az']*np.pi/180)
    Y = -(90-pb['alt'])/90.*np.cos(pb['az']*np.pi/180)
    for k in range(len(lst)):
        if pb['alt'][k]>0:
            plt.plot(X[k], Y[k], 'ok')
    plt.plot(0,0,'*r', markersize=12)
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    return

def reduceOneNight(date='2012-01-18',
                   data_directory='/Users/amerand/DATA/PRIMA',
                   withPss=False, onlyFileNumber=None):
    filesToReduce = filter(lambda x: 'OBJ_ASTRO' in x and not 'RED' in x,
                    [date+'/'+o for o in os.listdir(data_directory+'/'+
                                                    date+'/')])
    for f in filesToReduce:
        print '-'*12, f, '-'*12
        if withPss:
            os.system('./reducePrimaFile.py '+
                      os.path.join(data_directory,f)+
                      ' '+'pssguiRecorderDir='+
                      os.path.join(data_directory,'pssguiRecorder'))
        else:
            os.system('./reducePrimaFile.py '+os.path.join(data_directory,f))

def write4digits(n):
    tmp = str(int(n))
    tmp = '0'*(4-len(tmp))+tmp
    return tmp

def pixModel(meas, param):
    """
    model of the sepctral channels of PRIMA
    
    meas = some array of length N (does not matter)
    param = {'opd_i':, 'flux_i':, 'vis_i', ...,# for i:0->N-1 
             'Lair': ,'TA1':, phiA1':, 'wlA1':, 'visA1' ...}
    phi's and wl's for channels A,B,C,D and wavelengths 1,2,3,4,5
    
    returns [[A1,A2,A3,A4,A5,
              B..C...D...], [], []...]

    NOT FAST!!!
    """
    res = []
    for i,m in enumerate(meas):
        tmp = []
        for p in ['A','B','C','D']:
            for c in ['1','2','3','4','5']:
                tmp.append(param['flux_'+str(i)]*param['T'+p+c]*
                           (1+param['vis_'+str(i)]*
                            np.cos(2*np.pi/param['wl'+p+c]*
                                   param['opd_'+str(i)]+
                            param['phi'+p+c])))
        res.append(tmp)
    return res
    

class drs:
    OPDSNR = 'PDSNR' # this was OPDSNR before COMM15
    """
    data reduction software for PACMAN astrometric data
    """
    __version__ = '0.1'
    def __init__(self, filename, extract_scans=True, comment='',\
                 verbose=False, pssRec=None):
        """
        load PRIMA FITS file 'filename'. Does some minimal treatment,
        in particular tries to exctract scans if extra_scans==True
        (default is True, this does not crash if there are no scans)

        pssRec (optional) is a directory containing proper files. If
        not given, Rot, Alt and Az will be interplated between start
        and en values in header.
        """
        self.verbose=verbose
        # -- open file, keep track of file address
        if not os.path.isfile(filename):
            raise NameError(filename+' does not exist')
        if self.verbose:
            print 'INFO: setting .filename and .fullpath'
        self.filename = os.path.basename(filename)
        self.dirname  = os.path.dirname(filename)
        self.fullpath = filename
        if self.verbose:
            print 'INFO: setting .raw (reading data)'
        self.raw = pyfits.open(self.fullpath)

        # -- tracking and reference DL
        dl1 = self.getKeyword('ISS CONF DL1').strip()
        dl2 = self.getKeyword('ISS CONF DL2').strip()
        dl_ref = self.getKeyword('DEL REF NAME').strip()
        if dl_ref==dl1:
            self.DLtrack = dl2
            self.DLref = dl1
        else:
            self.DLtrack = dl1
            self.DLref = dl2
        # -- tracking and reference DDL
        try:
            DDL_CONFIG = [self.getKeyword('ISS DDL'+str(k)+' CONFIG')
                          for k in [1,2,3,4]]
            self.DDLtrack = 'DDL'+\
                            str(filter(lambda x: 'Tracking' in DDL_CONFIG[x],
                                       [0,1,2,3])[0]+1)
        except:
            print 'WARNING: could not determine tracking DDL'

        # -- what FSU does what?
        self.primary_fsu = self.getKeyword('DEL FT SENSOR')
        if self.primary_fsu == 'FSUB':
            self.secondary_fsu = 'FSUA'
        else:
            self.secondary_fsu = 'FSUB'

        # -- parameters for dispersion
        opl1 = (self.getKeyword('DEL DLT1 OPL START') +
                self.getKeyword('DEL DLT1 OPL END'))/2+\
                self.getKeyword('ISS CONF A1L')
        opl2 = (self.getKeyword('DEL DLT2 OPL START')+
                self.getKeyword('DEL DLT2 OPL END'))/2+\
                self.getKeyword('ISS CONF A2L')
        self.Lair = opl2-opl1
        print 'LST=', self.getKeyword('LST')/3590.
        print 'Lair=', round(self.Lair,2), '(m)'
        # -- try to extract scans
        self.scan_nscans = 0
        if extract_scans:
            try:
                self.scanExtract()
            except:
                print 'WARNING: could not exctract SCANS'

        # -- load calibration from the header
        self.getCalibFromHeader()
        self.comment = comment

        # -- interferometric baseline, in m, according to header
        self.base = (self.getKeyword('ISS CONF T1X')-self.getKeyword('ISS CONF T2X'),
                     self.getKeyword('ISS CONF T1Y')-self.getKeyword('ISS CONF T2Y'),
                     self.getKeyword('ISS CONF T1Z')-self.getKeyword('ISS CONF T2Z'),
                     self.getKeyword('ISS CONF A1L')-self.getKeyword('ISS CONF A2L'))

        self.projB = lambda x: projBaseline(self.base,
                    [self.getKeyword('OCS TARG1 ALPHAPMC')/15.,
                     self.getKeyword('OCS TARG1 DELTAPMC')], x)
        
        # -- local sideral time in decimal H
        self.mjd_obs = self.getKeyword('MJD-OBS')
        self.mjd_start = astro.tag2mjd(self.getKeyword('PCR ACQ START'))
        self.mjd_end = astro.tag2mjd(self.getKeyword('PCR ACQ END'))
        self.lst_obs = self.getKeyword('LST')/3600.
        self.lst_start = astro.tag2lst(self.getKeyword('PCR ACQ START'),
                               longitude=self.getKeyword('ISS GEOLON'))
        self.lst_end = astro.tag2lst(self.getKeyword('PCR ACQ END'),
                               longitude=self.getKeyword('ISS GEOLON'))

        self.date_obs = self.getKeyword('DATE-OBS')
        
        # -- check for dual FTK data
        n_min = min(len(self.raw['OPDC'].data.field('STATE')),
                    len(self.raw['DOPDC'].data.field('STATE')))
        wall   = np.where((self.raw['OPDC'].data.field('STATE')[:n_min]>=5)*
                          (self.raw['OPDC'].data.field('STATE')[:n_min]<=7)*
                          (self.raw['DOPDC'].data.field('STATE')[:n_min]>=5)*
                          (self.raw['DOPDC'].data.field('STATE')[:n_min]<=7))
        
        # -- check for glitches
        self.checkGlitches()

        # -- get rotator, Altitude and Azimuth
        self.pssRec = pssRec
        self.getRotAltAz()
       
        # -- compute index of air for FSUs (K band)
        self.airRefrIndex = n_air_P_T(2.15,
                    P= self.getKeyword('ISS AMBI PRES'),
                    T= self.getKeyword('ISS TEMP TUN2' ), # center of tunnel
                    e= self.getKeyword('ISS AMBI RHUM')/100.*\
                       self.getKeyword('ISS AMBI PRES'))

        # -- ins mode, primary and secondary FSU
        self.insmode = self.getKeyword('INS MODE').strip()
        if not self.insmode in ['NORMAL', 'SWAPPED']:
            print 'WARNING: unknown INSMODE=', self.insmode
        if self.insmode == 'INCONSISTENT':
            # -- try to guess
            if self.secondary_fsu == 'FSUA':
                self.insmode = 'NORMAL'
            elif self.secondary_fsu == 'FSUB':
                self.insmode = 'SWAPPED'
            print '  -> guessing: INSMODE=', self.insmode

        # -- if lab_FSUResponse, exctract scans:
        if self.getKeyword('TPL ID')=='PACMAN_cal_Lab_FSUResponse':
            self.labResponseExtract()
            print "INFO: TPL.ID==labResponseExtract -> exctracting scans in 'self.labScans'"
     
            
        return

    def __del__(self):
        if self.verbose:
            print "INFO: closing "+self.filename
        self.raw.close()
        del self
        return

    def getKeyword(self, key):
        """ get keyword from fits header. if fails, try to add
        'HIERARCH ESO ' in front of the keyword.
        """
        try:
            return self.raw[0].header[key]
        except:
            return self.raw[0].header['HIERARCH ESO '+key]

    def timeStamp2LST(self, t):
        """
        convert a FITS time stamp (in us) in LST time in decimal
        hours. Linear interpolation based on the Header time.
        """
        if isinstance(t, list):
            t = np.array(t)
        return self.lst_start + t*1e-6/(3590.)

    def LST2timeStamp(self, lst):
        """
        convert a LST time in FITS time stamp (in us).  Linear
        interpolation based on the Header time.
        """
        if isinstance(lst, list):
            lst = np.array(lst)
        return (lst-self.lst_start)*3590.*1e6

    def getRotAltAz(self):
        """
        try to read Derotator, Altitude and Azimuth 
        """
        
        # -- try to load pssRecorder files
        if not self.pssRec is None and \
               isinstance(self.pssRec, str) and \
               os.path.isdir(self.pssRec):
            # -- check files present
            pssF = filter(lambda x: 'pssguiRecorder_lat3fsm' in x,
                          os.listdir(self.pssRec))

            # -- convert start date in MJD
            pssRmjd = [astro.tag2mjd(x.split('_')[2]+'T'+
                       x.split('_')[3].split('.')[0].replace('-', ':'))
                       for x in pssF]
            
            # -- find the appropriate file
            filtMjd0 = filter(lambda x: x<self.mjd_start,
                                    pssRmjd)
            if len(filtMjd0)==0:
                print 'WARNING: pssguiRecorder files do not cover the time range.'
                print '         Will use header instead...'
            else:
                mjd_0 = np.array(filtMjd0).max()
                pssfile = pssF[filter(lambda k: pssRmjd[k]==mjd_0,
                                      range(len(pssF)))[0]]

                print 'INFO:', self.date_obs, '-> using pssguiRecorder file:', pssfile
                # -- open files
                self.at3fsm = pssRecorder(os.path.join(self.pssRec, pssfile))
                self.at4fsm = pssRecorder(os.path.join(self.pssRec,
                                                       pssfile.replace('lat3',
                                                                       'lat4')))
                self.at3tcs = pssRecorder(os.path.join(self.pssRec,
                                                   pssfile.replace('lat3fsm',
                                                                   'wat3tcs')))
                self.at4tcs = pssRecorder(os.path.join(self.pssRec,
                                                   pssfile.replace('lat3fsm',
                                                                   'wat4tcs')))
                # -- check that the MJD is covered
                try:
                    check = (self.at3fsm.mjd.min()<=self.mjd_start and
                             self.at3fsm.mjd.max()>=self.mjd_end)
                except:
                    check = False
                    
                if not check:
                    print 'WARNING: pssguiRecorder file does not cover the time range.'
                    print '         Will use header instead...'
                else:
                    self.AT3derot_MJD=(lambda x:
                                       np.interp(x, self.at3fsm.mjd,
                                             self.at3fsm.data['Dermec.[deg]']))
                    self.AT4derot_MJD=(lambda x:
                                       np.interp(x, self.at4fsm.mjd,
                                              self.at4fsm.data['Dermec.[deg]']))
                    print 'INFO: note that Alt and Az are reversed in the "at?tcs" files!!!!!'
                    self.AT3alt_MJD=(lambda x:
                                     np.interp(x, self.at3tcs.mjd,
                                               self.at3tcs.data['Az[deg]'])) # !!!
                    self.AT3az_MJD=(lambda x:
                                    np.interp(x, self.at3tcs.mjd,
                                              self.at3tcs.data['Alt[deg]'])) # !!!
                    self.AT4alt_MJD=(lambda x:
                                     np.interp(x, self.at4tcs.mjd,
                                               self.at4tcs.data['Az[deg]'])) # !!!
                    self.AT4az_MJD=(lambda x:
                                    np.interp(x, self.at4tcs.mjd,
                                              self.at4tcs.data['Alt[deg]'])) # !!!
                    return
                
        ### extrapolating from header
        ###############################
        if self.raw[0].header.has_key('HIERARCH ESO ISS PRI STS1 DERPOS START'):
    
            if np.abs(self.getKeyword('ISS PRI STS1 DERPOS START')-
                      self.getKeyword('ISS PRI STS1 DERPOS END'))>100:
                # rotator has wrapped during the observation
                print 'WARNING: ROTATOR 3 wrapping'
                derpos_start = self.getKeyword('ISS PRI STS1 DERPOS START')
                derpos_end   = self.getKeyword('ISS PRI STS1 DERPOS END')
                print 'DEBUG:', derpos_start, derpos_end
                print 'DEBUG:', '(', self.getKeyword('ISS PRI STS2 DERPOS START'), \
                      self.getKeyword('ISS PRI STS2 DERPOS END'), ')'
                if derpos_start > derpos_end:
                    derpos_end += 180
                else:
                    derpos_start += 180.
                derpos_end = derpos_start # dirty fix
                print 'DEBUG:',derpos_start, derpos_end
                self.AT3derot_MJD = lambda x:\
                     np.interp(x, [self.mjd_start, self.mjd_end],
                               [derpos_start, derpos_end])%180
            else:
                # normal case
                self.AT3derot_MJD = lambda x:\
                     np.interp(x, [self.mjd_start, self.mjd_end],
                               [self.getKeyword('ISS PRI STS1 DERPOS START'),
                                self.getKeyword('ISS PRI STS1 DERPOS END')])
                
        if self.raw[0].header.has_key('HIERARCH ESO ISS PRI STS2 DERPOS START'):
            
            if np.abs(self.getKeyword('ISS PRI STS2 DERPOS START')-
                      self.getKeyword('ISS PRI STS2 DERPOS END'))>100:
                # rotator has wrapped during the observation
                print 'WARNING: ROTATOR 4 wrapping!'
                derpos_start = self.getKeyword('ISS PRI STS2 DERPOS START')
                derpos_end   = self.getKeyword('ISS PRI STS2 DERPOS END')
                print 'DEBUG:', derpos_start, derpos_end
                print 'DEBUG: (', self.getKeyword('ISS PRI STS1 DERPOS START'), \
                      self.getKeyword('ISS PRI STS1 DERPOS END'), ')'
                if derpos_end < derpos_start:
                    derpos_end+=180
                else:
                    derpos_start+=180
                derpos_end = derpos_start # dirty fix
                print 'DEBUG:', derpos_start, derpos_end
                self.AT4derot_MJD = lambda x:\
                     np.interp(x, [self.mjd_start, self.mjd_end],
                               [derpos_start, derpos_end])%180
            else:
                # normal case
                self.AT4derot_MJD = lambda x:\
                     np.interp(x, [self.mjd_start, self.mjd_end],
                               [self.getKeyword('ISS PRI STS2 DERPOS START'),
                                self.getKeyword('ISS PRI STS2 DERPOS END')])
        # altitude:
        self.AT3alt_MJD = lambda x:\
             np.interp(x, [self.mjd_start, self.mjd_end],
                       [self.getKeyword('ISS TEL1 ALT START'),
                        self.getKeyword('ISS TEL1 ALT END')])
        self.AT4alt_MJD = lambda x:\
             np.interp(x, [self.mjd_start, self.mjd_end],
                       [self.getKeyword('ISS TEL2 ALT START'),
                        self.getKeyword('ISS TEL2 ALT END')])
        # azimuth: specific treatment for transit 0->360 
        az_start = self.getKeyword('ISS TEL1 AZ START')
        az_end = self.getKeyword('ISS TEL1 AZ END')
        if np.abs(az_start-az_end)>180:
            az_end -= 360.
        self.AT3az_MJD = lambda x:\
             np.interp(x, [self.mjd_start, self.mjd_end],
                       [az_start, az_end])
        az_start = self.getKeyword('ISS TEL2 AZ START')
        az_end = self.getKeyword('ISS TEL2 AZ END')
        if np.abs(az_start-az_end)>180:
            az_end -= 360.
        self.AT4az_MJD = lambda x:\
             np.interp(x, [self.mjd_start, self.mjd_end],
                           [az_start, az_end])
        # done!
        return
    
    def expectedOPD(self, plot=False):
        """
        """
        # -- coordinates, corrected for proper motion
        # radec = [astro.ESOcoord2decimal(self.getKeyword('OCS TARG1 ALPHAPMC')),
        #          astro.ESOcoord2decimal(self.getKeyword('OCS TARG1 DELTAPMC'))]
        # -- coordinates of the preset (same as above)
        radec = [astro.ESOcoord2decimal(self.getKeyword('OCS PRESET COU ALPHA')),
                 astro.ESOcoord2decimal(self.getKeyword('OCS PRESET COU DELTA'))]
        # -- keep when fringe tracking and non-0 DL
        dl2 = self.raw['OPDC'].data.field(self.getKeyword('ISS CONF DL2').strip())
        dl1 = self.raw['OPDC'].data.field(self.getKeyword('ISS CONF DL1').strip())
        w = np.where((self.raw['OPDC'].data.field('STATE')>=7)*(dl2!=0)*(dl1!=0))
        lst = self.timeStamp2LST(self.raw['OPDC'].data.field('TIME')[w])
        # -- DL2-DL1 in the file:
        dl2_dl1 = (dl2-dl1)[w]
        rtoffset = self.raw['OPDC'].data.field('RTOFFSET')[w]
        # -- 10s average:
        dl2_dl1, err, tmp =slidop.sliding_avg_rms(dl2_dl1, lst, 10./(24*3600.))
        rtoffset, ree, lst =slidop.sliding_avg_rms(rtoffset, lst, 10./(24*3600.))
        # -- DL2-DL1 computed from OPD model and coordinates:
        opd = projBaseline(self.base, radec, lst,
                           latitude=self.getKeyword('ISS GEOLAT'))['opd']
        # -- compute drift speed:
        drift = np.polyfit(lst-lst.mean(), dl2_dl1-opd, 1)
        print 'offset:', round(drift[-1], 3)*1e3, 'mm'
        print 'drift :', round(drift[-2]*1e6/24., 3), 'microns/h'
        if plot:
            plt.figure(0)
            plt.clf()
            ax = plt.subplot(311)
            plt.plot(lst, opd, 'r-', label='recomputed opd', alpha=0.5, linewidth=3)
            plt.plot(lst, (dl2_dl1),'+b', label=self.getKeyword('ISS CONF DL2')+'-'+
                        self.getKeyword('ISS CONF DL1'), alpha=0.5)
            plt.legend()
            plt.xlabel('LST')
            plt.ylabel('(m)')
            plt.subplot(312, sharex = ax)
            plt.plot(lst, (dl2_dl1 - opd)*1e3,'+k',
                        label='('+self.getKeyword('ISS CONF DL2')+'-'+
                        self.getKeyword('ISS CONF DL1')+') - recomputed', alpha=0.5)
            plt.plot(lst, 1e3*np.polyval(drift, lst-lst.mean()), '-y',
                        alpha=0.8, linewidth=3, label='drift')
            plt.xlabel('LST')
            plt.ylabel('(mm)')
            plt.legend()
            plt.subplot(313, sharex=ax)
            plt.plot(lst, rtoffset*1e3, '-g', label='rtoffset')
            plt.plot(lst, (dl2_dl1 - opd - np.polyval(drift,lst-lst.mean()))*1e3,
                        '+k',
                        label='('+self.getKeyword('ISS CONF DL2')+'-'+
                        self.getKeyword('ISS CONF DL1')+') - recomputed - drift',
                        alpha=0.5)
            plt.xlabel('LST')
            plt.ylabel('(mm)')
            plt.legend()
            return 
        # -- build a dictionnary with 'reduced' data 
        if self.insmode=='NORMAL':
            DDL = np.interp(self.LST2timeStamp(lst),
                            self.raw['DOPDC'].data.field('TIME'),
                            self.raw['DOPDC'].data.field('DDL4'))-\
                  np.interp(self.LST2timeStamp(lst),
                            self.raw['DOPDC'].data.field('TIME'),
                            self.raw['DOPDC'].data.field('DDL2'))
        else:
            DDL = np.interp(self.LST2timeStamp(lst),
                            self.raw['DOPDC'].data.field('TIME'),
                            self.raw['DOPDC'].data.field('DDL3'))-\
                  np.interp(self.LST2timeStamp(lst),
                            self.raw['DOPDC'].data.field('TIME'),
                            self.raw['DOPDC'].data.field('DDL1'))
        # -- result dictionnary
        return {'TARGET':self.getKeyword('OCS PS ID'),
                'LST':lst, 'INSMODE':self.insmode,
                'RA':radec[0], 'DEC':radec[1],
                'DL2-DL1':dl2_dl1, 'ERR': err,
                'RTOFFSET': rtoffset,
                'AZ':0.25*(self.getKeyword('ISS TEL1 AZ START')+
                           self.getKeyword('ISS TEL2 AZ START')+
                           self.getKeyword('ISS TEL1 AZ END')+
                           self.getKeyword('ISS TEL2 AZ END')),
                'ALT':0.25*(self.getKeyword('ISS TEL1 ALT START')+
                            self.getKeyword('ISS TEL2 ALT START')+
                            self.getKeyword('ISS TEL1 ALT END')+
                            self.getKeyword('ISS TEL2 ALT END')),
                'XYZA':self.base, 'AIRN':self.airRefrIndex,
                'computed opd': opd, 'DDL':DDL,
                'drift':drift[-2],
                'offset':drift[-1]}
    
    def overviewCommand(self):
        """
        compare command and position of DL from OPDC
        """
        plt.figure(11)
        plt.clf()
        ax = plt.subplot(211)
        plt.plot(self.raw['OPDC'].data.field('TIME'),
                1e6*self.raw['OPDC'].data.field('FUOFFSET'),
                   color='r', label='FUOFFSET',
                    linewidth=1, alpha=1)       
        plt.plot(self.raw['OPDC'].data.field('TIME'),
                 1e6*(self.raw['OPDC'].data.field(self.DLtrack)-
                 self.raw['OPDC'].data.field('PSP')),
                 color='r', linewidth=3, alpha=0.5,
                 label=self.DLtrack+'-PSP')
        plt.legend()
        plt.subplot(212, sharex=ax)
        plt.plot(self.raw['OPDC'].data.field('TIME'),
                1e6*self.raw['OPDC'].data.field('FUOFFSET')-
                1e6*(self.raw['OPDC'].data.field(self.DLtrack)-
                 self.raw['OPDC'].data.field('PSP')),
                   color='k', label='$\Delta$',
                    linewidth=1, alpha=1)  
        
        signal = self.raw['OPDC'].data.field('FUOFFSET')
        plt.figure(12)
        plt.clf()
        ax2 = plt.subplot(111)
        Fs = 1e6/np.diff(self.raw['OPDC'].data.field('TIME')).mean()
        print Fs
        ax2.psd(signal[:50000], NFFT=5000, Fs=Fs, label='FUOFFSET',scale_by_freq=0)
        plt.legend()
    
    def overview(self, minState=5):
        """
        plotting overview of RTOFFSETs and STATEs of OPDC and DOPDC
        """
        n = 600
        
        ### first plot: the RTOFFSETs and STATES
        plt.figure(10)
        plt.clf()
        plt.subplots_adjust(hspace=0.05, top=0.95, left=0.05,
                               right=0.99, wspace=0.00, bottom=0.1)
        ax1 = plt.subplot(n+11)
        try:
            print self.insmode+' | pri:'+\
                  self.getKeyword('OCS PS ID')+' | sec:'+\
                  self.getKeyword('OCS SS ID')
      
            plt.title(self.filename+' | '+self.insmode+' | pri:'+
                         self.getKeyword('OCS PS ID')+' | sec:'+
                         self.getKeyword('OCS SS ID'))
        except:
            pass
        plt.plot(self.raw['OPDC'].data.field('TIME'),
                    self.raw['OPDC'].data.field('FUOFFSET')*1e3,
                    color=(1.0, 0.5, 0.0), label=self.DLtrack+' (FUOFFSET)',
                    linewidth=3, alpha=0.5)
        plt.legend(prop={'size':9})
        plt.ylabel('(mm)')
        plt.xlim(0)
        
        plt.subplot(n+12, sharex=ax1) # == DDL movements
        
        plt.plot(self.raw['DOPDC'].data.field('TIME'),
                    1e3*self.raw['DOPDC'].data.field(self.DDLtrack),
                    color=(0.0, 0.5, 1.0), linewidth=3, alpha=0.5,
                    label=self.DDLtrack)
        plt.plot(self.raw['DOPDC'].data.field('TIME'),
                    1e3*self.raw['DOPDC'].data.field('PSP'),
                    color=(0.0, 0.5, 1.0), linewidth=1, alpha=0.9,
                    label='PSP', linestyle='dashed')
        plt.legend(prop={'size':9})
        plt.ylabel('(mm)')
        plt.xlim(0)
        
        plt.subplot(n+13, sharex=ax1) # == states
        plt.plot(self.raw['OPDC'].data.field('TIME'),
                    self.raw['OPDC'].data.field('STATE'),
                    color=(1.0, 0.5, 0.0), label='OPDC')
        plt.plot(self.raw['DOPDC'].data.field('TIME'),
                    self.raw['DOPDC'].data.field('STATE'),
                    color=(0.0, 0.5, 1.0), label='DOPDC')
        plt.legend(prop={'size':9})
        plt.ylabel('STATES')
        yl=plt.ylim()
        plt.ylim(yl[0]-1, yl[1]+1)
        plt.xlim(0)
        ### fluxes
        plt.subplot(n+14, sharex=ax1)
        try:
            fsua_dark = self.fsu_calib[('FSUA', 'DARK')][0,0]
            fsub_dark = self.fsu_calib[('FSUB', 'DARK')][0,0]
            fsua_alldark = self.fsu_calib[('FSUA', 'DARK')].sum(axis=1)[0]
            fsub_alldark = self.fsu_calib[('FSUB', 'DARK')].sum(axis=1)[0]
        except:
            print 'WARNING: there are no FSUs calibrations in the header'
            fsua_dark = 0.0
            fsub_dark = 0.0
            fsua_alldark = 0.0
            fsub_alldark = 0.0

        M0 = 17.5
        fluxa = (self.raw['IMAGING_DATA_FSUA'].data.field('DATA1')[:,0]+
                 self.raw['IMAGING_DATA_FSUA'].data.field('DATA2')[:,0]+
                 self.raw['IMAGING_DATA_FSUA'].data.field('DATA3')[:,0]+
                 self.raw['IMAGING_DATA_FSUA'].data.field('DATA4')[:,0]-
                 fsua_alldark)/\
                 (4*self.getKeyword('ISS PRI FSU1 DIT'))
        print 'FLUX FSUA (avg, rms):', round(fluxa.mean(), 0), 'ADU/s',\
              round(100*fluxa.std()/fluxa.mean(), 0), '%'
        print '  -> pseudo mag = '+str(M0)+' - 2.5*log10(flux) =',\
              round(M0-2.5*np.log10(fluxa.mean()),2)
        fluxb = (self.raw['IMAGING_DATA_FSUB'].data.field('DATA1')[:,0]+
                 self.raw['IMAGING_DATA_FSUB'].data.field('DATA2')[:,0]+
                 self.raw['IMAGING_DATA_FSUB'].data.field('DATA3')[:,0]+
                 self.raw['IMAGING_DATA_FSUB'].data.field('DATA4')[:,0]-
                 fsub_alldark)/\
                 (4*self.getKeyword('ISS PRI FSU2 DIT'))
        print 'FLUX FSUB (avg, rms):', round(fluxb.mean(), 0), 'ADU/s',\
              round(100*fluxb.std()/fluxb.mean(), 0), '%'
        print '  -> pseudo mag = '+str(M0)+' - 2.5*log10(flux) =',\
              round(M0-2.5*np.log10(fluxb.mean()),2)
        plt.plot(self.raw['IMAGING_DATA_FSUA'].data.field('TIME'),\
                    fluxa/1000, color='b', alpha=0.5, label='FSUA')
        plt.plot(self.raw['IMAGING_DATA_FSUB'].data.field('TIME'),\
                    fluxb/1000, color='r', alpha=0.5, label='FSUB')

        plt.ylim(1)
        plt.legend(prop={'size':9})
        plt.ylabel('flux - DARK (kADU)')
        plt.xlim(0)
        plt.subplot(n+15, sharex=ax1)
        try:
            # -- old data version
            plt.plot(self.raw['IMAGING_DATA_FSUA'].data.field('TIME'),
                        self.raw['IMAGING_DATA_FSUA'].data.field('OPDSNR'),
                        color='b', alpha=0.5,  label='FSUA SNR')
            plt.plot(self.raw['IMAGING_DATA_FSUB'].data.field('TIME'),
                        self.raw['IMAGING_DATA_FSUB'].data.field('OPDSNR'),
                        color='r', alpha=0.5, label='FSUB SNR')
        except:
            plt.plot(self.raw['IMAGING_DATA_FSUA'].data.field('TIME'),
                        self.raw['IMAGING_DATA_FSUA'].data.field(self.OPDSNR),
                        color='b', alpha=0.5, label='FSUA SNR')
            plt.plot(self.raw['IMAGING_DATA_FSUB'].data.field('TIME'),
                        self.raw['IMAGING_DATA_FSUB'].data.field(self.OPDSNR),
                        color='r', alpha=0.5, label='FSUB SNR')
        plt.legend(prop={'size':9})
        
        A = (self.raw['IMAGING_DATA_FSUA'].data.field('DATA1')[:,0]-
             self.fsu_calib[('FSUA', 'DARK')][0,0])/\
             (self.fsu_calib[('FSUA', 'FLAT')][0,0]-
              2*self.fsu_calib[('FSUA', 'DARK')][0,0])
        B = (self.raw['IMAGING_DATA_FSUA'].data.field('DATA2')[:,0]-
             self.fsu_calib[('FSUA', 'DARK')][0,1])/\
             (self.fsu_calib[('FSUA', 'FLAT')][0,1]-
              2*self.fsu_calib[('FSUA', 'DARK')][0,1])
        C = (self.raw['IMAGING_DATA_FSUA'].data.field('DATA3')[:,0]-
             self.fsu_calib[('FSUA', 'DARK')][0,2])/\
             (self.fsu_calib[('FSUA', 'FLAT')][0,2]-
              2*self.fsu_calib[('FSUA', 'DARK')][0,2])
        D = (self.raw['IMAGING_DATA_FSUA'].data.field('DATA4')[:,0]-
             self.fsu_calib[('FSUA', 'DARK')][0,3])/\
             (self.fsu_calib[('FSUA', 'FLAT')][0,3]-
              2*self.fsu_calib[('FSUA', 'DARK')][0,3])
        snrABCD_a = ((A-C)**2+(B-D)**2)
        snrABCD_a /= ((A-C).std()**2+ (B-D).std()**2)
        #plt.plot(self.raw['IMAGING_DATA_FSUA'].data.field('TIME'),
        #            snrABCD_a, color='b', alpha=0.5, linestyle='dashed')
        
        A = (self.raw['IMAGING_DATA_FSUB'].data.field('DATA1')[:,0]-
             self.fsu_calib[('FSUB', 'DARK')][0,0])/\
             (self.fsu_calib[('FSUB', 'FLAT')][0,0]-
              2*self.fsu_calib[('FSUB', 'DARK')][0,0])
        B = (self.raw['IMAGING_DATA_FSUB'].data.field('DATA2')[:,0]-
             self.fsu_calib[('FSUB', 'DARK')][0,1])/\
             (self.fsu_calib[('FSUB', 'FLAT')][0,1]-
              2*self.fsu_calib[('FSUB', 'DARK')][0,1])
        C = (self.raw['IMAGING_DATA_FSUB'].data.field('DATA3')[:,0]-
             self.fsu_calib[('FSUB', 'DARK')][0,2])/\
             (self.fsu_calib[('FSUB', 'FLAT')][0,2]-
              2*self.fsu_calib[('FSUB', 'DARK')][0,2])
        D = (self.raw['IMAGING_DATA_FSUB'].data.field('DATA4')[:,0]-
             self.fsu_calib[('FSUB', 'DARK')][0,3])/\
             (self.fsu_calib[('FSUB', 'FLAT')][0,3]-
              2*self.fsu_calib[('FSUB', 'DARK')][0,3])
    
        snrABCD_b = ((A-C)**2+(B-D)**2)
        snrABCD_b /= ((A-C).std()**2+ (B-D).std()**2)
        #plt.plot(self.raw['IMAGING_DATA_FSUB'].data.field('TIME'),
        #            snrABCD_b, color='r', alpha=0.5, linestyle='dashed')    
        
        # -- SNR levels:
        #plt.hlines([self.getKeyword('INS OPDC OPEN'),
        #               self.getKeyword('INS OPDC CLOSE'),
        #               self.getKeyword('INS OPDC DETECTION')],
        #               self.raw['IMAGING_DATA_FSUB'].data.field('TIME').min(),
        #               self.raw['IMAGING_DATA_FSUB'].data.field('TIME').max(),
        #               color=(1.0, 0.5, 0.0))
        #plt.hlines([self.getKeyword('INS DOPDC OPEN'),
        #               self.getKeyword('INS DOPDC CLOSE'),
        #               self.getKeyword('INS DOPDC DETECTION')],
        #              self.raw['IMAGING_DATA_FSUB'].data.field('TIME').min(),
        #              self.raw['IMAGING_DATA_FSUB'].data.field('TIME').max(),
        #              color=(0.0, 0.5, 1.0))
        # -- plot thresholds
        plt.ylabel('SNR')
        plt.xlim(0)
        
        if self.getKeyword('OCS DET IMGNAME')=='PACMAN_OBJ_ASTRO_':
            # == dual FTK
            plt.subplot(n+16, sharex=ax1)
            plt.ylabel('PRIMET ($\mu$m)')
            #met = interp1d(np.float_(self.raw['METROLOGY_DATA'].\
            #                            data.field('TIME')),\
            #               self.raw['METROLOGY_DATA'].data.field('DELTAL'),\
            #               kind = 'linear', bounds_error=False, fill_value=0.0)
            met = lambda x: np.interp(x,
                    np.float_(self.raw['METROLOGY_DATA'].data.field('TIME')),
                    self.raw['METROLOGY_DATA'].data.field('DELTAL'))
            metro = met(self.raw['DOPDC'].data.field('TIME'))*1e6
            n_ = min(len(self.raw['DOPDC'].data.field('TIME')),
                    len(self.raw['OPDC'].data.field('TIME')))

            plt.plot(self.raw['DOPDC'].data.field('TIME'),
                        metro, color=(0.5,0.5,0.), label='A-B')

            w1 = np.where((self.raw['OPDC'].data.field('STATE')[:n_]>=minState)*\
                          (self.raw['OPDC'].data.field('STATE')[:n_]<=7))
            try:
                print 'OPDC FTK stat:', round(100*len(w1[0])/float(n_), 1), '%'
            except:
                print 'OPDC FTK stat: 0%'

            w1 = np.where((self.raw['DOPDC'].data.field('STATE')[:n_]>=minState)*\
                          (self.raw['DOPDC'].data.field('STATE')[:n_]<=7))
            try:
                print 'DOPDC FTK stat:', round(100*len(w1[0])/float(n_), 1), '%'
            except:
                print 'DOPDC FTK stat: 0%'

            w = np.where((self.raw['DOPDC'].data.field('STATE')[:n_]>=minState)*\
                         (self.raw['DOPDC'].data.field('STATE')[:n_]<=7)*\
                         (self.raw['OPDC'].data.field('STATE')[:n_]>=minState)*\
                         (self.raw['OPDC'].data.field('STATE')[:n_]<=7))
            try:
                print 'DUAL FTK stat:', round(100*len(w[0])/float(n_),1), '%'
            except:
                print 'DUAL FTK stat: 0%'

            plt.xlim(0)
            plt.plot(self.raw['DOPDC'].data.field('TIME')[w],
                        metro[w], '.g', linewidth=2,
                        alpha=0.5, label='dual FTK')
            #plt.legend()
            if len(w[0])>10 and False:
                coef = np.polyfit(self.raw['DOPDC'].data.field('TIME')[w],
                                  metro[w], 2)
                plt.plot(self.raw['DOPDC'].data.field('TIME'),
                            np.polyval(coef, self.raw['DOPDC'].
                                       data.field('TIME')),
                            color='g')
                plt.ylabel('metrology')

                print 'PRIMET drift (polyfit)   :', 1e6*coef[1], 'um/s'
                slope, rms, synth = NoisySlope(self.raw['DOPDC'].
                                               data.field('TIME')[w],
                                               metro[w], 3e6)
                plt.figure(10)
                yl = plt.ylim()
                plt.plot(self.raw['DOPDC'].data.field('TIME')[w],
                            synth, color='r')
                plt.ylim(yl)
                print 'PRIMET drift (NoisySlope):',\
                      slope*1e6,'+/-', rms*1e6, 'um/s'
        else:
            # == scanning
            plt.subplot(n+16, sharex=ax1)
            fringesOPDC = \
                 self.raw['IMAGING_DATA_'+self.primary_fsu].data.field('DATA1')[:,0]-\
                 self.raw['IMAGING_DATA_'+self.primary_fsu].data.field('DATA3')[:,0]
            
            fringesDOPDC =\
                 self.raw['IMAGING_DATA_'+self.secondary_fsu].data.field('DATA1')[:,0]-\
                 self.raw['IMAGING_DATA_'+self.secondary_fsu].data.field('DATA3')[:,0]
            
            plt.plot(self.raw['IMAGING_DATA_'+self.primary_fsu].data.field('TIME'),
                 scipy.signal.wiener(fringesOPDC/fringesOPDC.std()),
                 color=(1.0, 0.5, 0.0), alpha=0.6,
                 label=self.primary_fsu+'/OPDC')
            plt.plot(self.raw['IMAGING_DATA_'+self.secondary_fsu].data.field('TIME'),
                 scipy.signal.wiener(fringesDOPDC/fringesDOPDC.std()),
                 color=(0.0, 0.5, 1.0), alpha=0.6,
                 label=self.secondary_fsu+'/DOPDC')
            plt.legend(prop={'size':9})
            plt.ylabel('A-C')
        plt.xlabel('time stamp ($\mu$s)')
        return

    def checkGlitches(self):
        """
        use keywords to compare if PRIMET glitches happened during the
        exposure.
        """
        cP  = self.getKeyword('ISS PRI MET C') # in m/s
        dnuP = self.getKeyword('ISS PRI MET F_SHIFT')*1e6 # in Hz
        nuP  = self.getKeyword('ISS PRI MET LASER_F')
        #self.jump =  (cP*dnuP/2/(nuP**2))*(2**24-1) # PRIMET jump in m, COMM14
        self.metJumpSize =  (cP*dnuP/2/(nuP**2))*(2**31-1) # PRIMET jump in m

        relevant_keywords = filter(lambda x: 'NGLIT' in x and
                                   'START' in x,
                                   self.raw[0].header.keys())
        relevant_keywords = [k.split()[4] for k in relevant_keywords]
        glitches = {}
        glitchesStartEnd = {}
        for k in relevant_keywords:
            glitches[k] = self.getKeyword('ISS PRI MET '+k+' END')-\
                          self.getKeyword('ISS PRI MET '+k+' START')
            glitchesStartEnd[k] = (self.getKeyword('ISS PRI MET '+k+' START'),
                                self.getKeyword('ISS PRI MET '+k+' END'))
        self.glitches = glitches
        self.glitchesStartEnd = glitchesStartEnd
        if 'NGLITAB' in glitches.keys():
            if glitches['NGLITAB'] !=0:
                print '*SERIOUS WARNING*', glitches['NGLITAB'],\
                      'glitches in PRIMET A-B in this file'
        else:
            print '*WARNING*: could not assess glitches in A-B'

        if 'NGLITB' in glitches.keys():
            if glitches['NGLITB'] !=0:
                print '*SERIOUS WARNING*', glitches['NGLITB'],\
                      'glitches in PRIMET -B in this file'
        else:
            print 'WARNING: could not assess glitches in -B'

        if glitches['NGLITABFCO'] !=0:
            print 'WARNING: AB overflow!', glitches['NGLITABFCO']
        if glitches['NGLITBFCO'] !=0:
            print 'WARNING: -B overflow!', glitches['NGLITBFCO']
        self.glitches = glitches
        return

    def getCalibFromHeader(self):
        """
        retrieve DARK, FLAT, PHAS, VISI and WAVE for each channel
        (W,1,2,3,4,5) each table is 6x4 (channel, ABCD).

        updates self.fsu_calib dictionnary
        """
        calibs = ['DARK', 'FLAT', 'PHAS', 'VISI', 'WAVE']
        fsus = ['FSUA', 'FSUB']
        channels = ['W', '1', '2', '3', '4', '5']
        try:
            self.fsu_calib = {}
            for fsu in fsus:
                for calib in calibs:
                    self.fsu_calib[(fsu, calib)] = np.zeros((6,4))
                    for k, chan in enumerate(channels):
                        self.fsu_calib[(fsu, calib)][k,:] =\
                                             self.read4num('OCS '+fsu+' K'+\
                                                           chan+calib)

            return True
        except:
            if self.verbose:
                print '*WARNING* there do not seem to be calibrations'
            return False

    def read4num(self,s):
        """
        read numbers separated by ',' in a keyword 's' of the main
        header. Used to extract calibrations
        """
        s = self.getKeyword(s)
        return np.float_(s.split(','))

    def setupForFTK(self):
        """
        analyse scans to find fringes and returns the parameters to set the
        OPDC and DOPDC panels.
        """
        t1 = self.getKeyword('ISS CONF T1NAME').strip()
        t2 = self.getKeyword('ISS CONF T2NAME').strip()
        #swapped = self.getKeyword('ISS PRI STS'+t1[2]+' GUIDE_MODE').strip()

        fsub_pos_fri = self.maxSnrInScan(fsu='FSUB', opdc='OPDC', plot=1)
        fsua_pos_fri = self.maxSnrInScan(fsu='FSUA', opdc='OPDC', plot=2)
        print '---{'+self.insmode+'}---'
        if swapped == 'NORMAL':
            print ' OPDC -> [ZPD offset] x [sign',self.DLtrack,\
                '] =',-fsub_pos_fri
            print 'DOPDC -> [ZPD offset] x [sign',\
                  self.getKeyword('ISS DDL1 NAME').strip(),\
                   '] = ',(fsub_pos_fri-fsua_pos_fri)
        else:
            print ' OPDC -> [ZPD offset] x [sign',self.DLtrack,\
                '] =', fsua_pos_fri
            print 'DOPDC -> [ZPD offset] x [sign',\
                  self.getKeyword('ISS DDL2 NAME').strip(),\
                   '] = ',(fsua_pos_fri-fsub_pos_fri)
        return

    def findFringesInScan(self, fsu='FSUB', plot=False, calibrated=True):
        """
        Uses scans to find fringes position in term of FUOFFSET.
        returns fringes position in m. By default, assume you are
        fringe searching, hence it looks at the primary FSU
        """
        fringes_pos = []
        weight = []

        if plot:
            plt.figure(0)
            plt.clf()
            plt.title(self.filename+' | '+fsu+' | '+self.insmode)
            plt.xlabel('OPD '+self.DLtrack+' (m)')

        for k in range(self.scan_nscans):
            x = self.getScan(k, isolate=False, calibrated=calibrated,
                             FUOFFSET=1, fsu=fsu, resample=False)
            opd = np.linspace(x[1].min(), x[1].max(), len(x[1]))
            scan1 = np.interp(opd, x[1], x[2]-x[4])
            scan2 = np.interp(opd, x[1], x[3]-x[5])
            sigS = np.exp(-((opd-opd.mean())/10e-6)**2)*\
                   np.sin((opd-opd.mean())/2.2e-6*2*np.pi)
            sigC = np.exp(-((opd-opd.mean())/10e-6)**2)*\
                   np.cos((opd-opd.mean())/2.2e-6*2*np.pi)
            sigS = np.roll(sigS, len(sigS)/2)
            sigC = np.roll(sigC, len(sigC)/2)
            
            fft_sigS = np.fft.fft(sigS)
            fft_sigC = np.fft.fft(sigC)
            fft_scan1 = np.fft.fft(scan1)
            fft_scan2 = np.fft.fft(scan2)
            # correlate
            powerS1 = np.abs(np.fft.ifft(fft_scan1*fft_sigS))**2
            powerC1 = np.abs(np.fft.ifft(fft_scan1*fft_sigC))**2
            powerS2 = np.abs(np.fft.ifft(fft_scan2*fft_sigS))**2
            powerC2 = np.abs(np.fft.ifft(fft_scan2*fft_sigC))**2
            power1 = powerS1+powerC1
            power2 = powerS2+powerC2
            if power1.max()>(power1.mean()+8*power1.std()):
                fringes_pos.append(opd[power1.argmax()])
                weight.append(power1.max())
                if plot:
                    plt.plot(opd, (power1-power1.mean())/power1.std(),
                                linewidth=2)
                    print x[0].min(), x[0].max()
                    #plt.plot(opd, np.interp(x[0],
                    #  self.raw['IMAGING_DATA_'+fsu.upper()].data.field('TIME'),
                    #  self.raw['IMAGING_DATA_'+fsu.upper()].data.field(OPDSNR)),
                    #                           'k', alpha=0.5, linewidth=3)
            else:
                if plot:
                    plt.plot(opd, (power1-power1.mean())/power1.std(),
                                'k', alpha='0.5')

            if power2.max()>(power2.mean()+8*power2.std()):
                fringes_pos.append(opd[power2.argmax()])
                weight.append(power2.max())
                if plot:
                    plt.plot(opd, (power2-power2.mean())/power2.std(),
                                linewidth=2)
            else:
                if plot:
                    plt.plot(opd, (power2-power2.mean())/power2.std(),
                                'k', alpha='0.5')

        return (np.array(fringes_pos)*np.array(weight)).sum()/\
                np.array(weight).sum()

    def scanExtract(self, plot=False):
        """ extract scans and store the starting and stopping time
            opdc= 'opdc' (default) or 'dopdc'
            updates: self.scan_fsu, self.scan_opdc,
                     self.scan_start, self.scan_end

            returns the number of scans (or False if no scans)
        """
        # check if their are any scans:

        if (self.raw['OPDC'].data.field('STATE')==20).any() or \
               (self.raw['OPDC'].data.field('STATE')==21).any():
            self.scan_opdc='OPDC'
            self.scan_DL = self.DLtrack
        else:
            self.scan_opdc='' # nothing in OPDC

        if  (self.raw['DOPDC'].data.field('STATE')==20).any() or \
               (self.raw['DOPDC'].data.field('STATE')==21).any():
            self.scan_opdc='DOPDC'
            self.scan_DL = self.DDLtrack

        if  self.scan_opdc=='':
            if self.verbose:
                print ' info: no scans'
                self.scanning=False
            return

        self.scanning=True
        print ' ~ setting .scan_opdc to '+self.scan_opdc

        # determine which is the FSU which is scanning:
        if self.scan_opdc=='OPDC':
            self.scan_fsu = self.getKeyword('DEL FT SENSOR')
            if self.scan_fsu=='FSUA':
                self.ftk_fsu = 'FSUB'
            else:
                self.ftk_fsu = 'FSUA'
        elif self.getKeyword('DEL FT SENSOR')=='FSUA':
            self.scan_fsu = 'FSUB'
            self.ftk_fsu = 'FSUA'
        else :
            self.scan_fsu = 'FSUA'
            self.ftk_fsu = 'FSUB'

        print ' ~ setting .scan_fsu to '+self.scan_fsu

        # exctract scans
        previous_state = 0
        scan_start= []
        scan_end = []
        cond = self.raw[self.scan_opdc].data.field('STATE')
        state=[]

        if cond[0]==20 or cond[0]==21: # already scanning
            scan_start.append(0)
        for i in np.arange(1,cond.size-1):
            if cond[i]!=cond[i-1]: # change of state
                if cond[i-1]==20 or cond[i-1]==21: # end of previous scan
                    scan_end.append(i-1)
                if cond[i]==20 or cond[i]==21: # start of new scan
                    scan_start.append(i)
        if cond[-1]==20 or cond[-1]==21: # still scanning
            scan_end.append(cond.size-1)

        print ' ~ setting .scan_start (time stamps)'
        self.scan_start= self.raw[self.scan_opdc].data.field('TIME')[scan_start]
        print ' ~ setting .scan_end (time stamps)'
        self.scan_end= self.raw[self.scan_opdc].data.field('TIME')[scan_end]
        print ' ~ setting .scan_nscans to ', len(self.scan_start)
        self.scan_nscans = len(self.scan_start)
        if plot:
            if isinstance(plot, int):
                plt.figure(plot)
            plt.clf()
            plt.title(self.filename)
            plt.plot(self.raw[self.scan_opdc].data.field('TIME'), cond,\
                        color='green')
            plt.plot(self.scan_start, 0*self.scan_start+20.45,'b+')
            plt.plot(self.scan_end, 0*self.scan_start+20.55,'r+')
        scan_start = scan_end = cond = []
        return self.scan_nscans

    def scanGetStampsImaging(self, i=0):
        """ uses self.scan_start and self.scan_end to return the
            indexes in IMAGING_DATA_self.scan_fsu of scan number i.
            i in [0:self.scan_nscans-1]
        """
        wi = np.where((self.raw['IMAGING_DATA_'+self.scan_fsu].\
                          data.field('TIME')>=self.scan_start[i-1])*\
                         (self.raw['IMAGING_DATA_'+self.scan_fsu].\
                          data.field('TIME')<=self.scan_end[i-1]))
        return wi

    def scanGetStampsOpdc(self, i=0):
        """ uses self.scan_start and self.scan_end to return the
            indexes in HDU extention 'self.scan_opdc' of scan number
            i. i in [0:self.scan_nscans-1]
        """
        wo = np.where((self.raw[self.scan_opdc].\
                          data.field('TIME')>=self.scan_start[i-1])*\
                         (self.raw[self.scan_opdc].\
                          data.field('TIME')<=self.scan_end[i-1]))
        return wo

    def getScan(self, i=0, channel=0, isolate=True, scan_width=50e-6,\
                calibrated=False, resample=False, plot=False,
                fsu=None, FUOFFSET=False):
        """
        return scan i (default is first one).  ftkFSU=False assumes
        you FTK with primary FSU and scan with secondary FSU. in case
        of fringes search, on scans with both so ftkFSU=True can be
        used.

        a.getScan(3, fsu='FSUB', plot=1, isolate=True, resample=False,
        calibrated=True, scan_width=1000e-6)

        resample does a regular resampling (linear interpolation)
        """
        if fsu is None:
            if self.ftk_fsu=='FSUA':
                fsu='FSUB'
            else:
                fsu='FSUA'

        if self.scan_nscans==0:
            return None
        if i<0 or i>self.scan_nscans-1:
            print 'error: not such scans (nscans=',self.scan_nscans,')'
            return -1
        wi = self.scanGetStampsImaging(i)
        wo = self.scanGetStampsOpdc(i)

        if FUOFFSET:
            dlfdbck = lambda x: np.interp(x,
                self.raw[self.scan_opdc].data.field('TIME')[wo],
                self.raw[self.scan_opdc].data.field('FUOFFSET')[wo])
        else:
            # was DLFDBACK before
            dlfdbck = lambda x: np.interp(x,
                          self.raw[self.scan_opdc].data.field('TIME')[wo],
                          self.raw[self.scan_opdc].data.field(self.scan_DL)[wo]-
                          self.raw[self.scan_opdc].data.field(self.scan_DL)[wo].mean())
         
        ### time tags during scan:
        t = self.raw['IMAGING_DATA_'+fsu].data.field('TIME')[wi]
        t = np.clip(t,self.raw[self.scan_opdc].data.field('TIME')[wo].min(),
                    self.raw[self.scan_opdc].data.field('TIME')[wo].max())
        x = dlfdbck(t) # interpolate for imaging times
        opdc_state = np.interp(t, self.raw['OPDC'].data.field('TIME'),
                               self.raw['OPDC'].data.field('STATE'))
        print 'MEAN OPDC STATE:', opdc_state.mean()
        n_samples_per_fringes = 2.15e-6/np.median(np.abs(np.diff(x)))

        pdsnr = self.raw['IMAGING_DATA_'+fsu.upper()].data.field('PDSNR')[wi[0]]
        pd = self.raw['IMAGING_DATA_'+fsu.upper()].data.field('PD')[wi[0]]
        
        if channel>=0:
            fringesA = self.raw['IMAGING_DATA_'+fsu.upper()].\
                       data.field('DATA1')[wi[0],channel]
            fringesB = self.raw['IMAGING_DATA_'+fsu.upper()].\
                       data.field('DATA2')[wi[0],channel]
            fringesC = self.raw['IMAGING_DATA_'+fsu.upper()].\
                       data.field('DATA3')[wi[0],channel]
            fringesD = self.raw['IMAGING_DATA_'+fsu.upper()].\
                       data.field('DATA4')[wi[0],channel]
            if 'DATA5' in [c.name for c in self.raw['IMAGING_DATA_'+fsu.upper()].data.columns]:
                print 'removing dark pixel'
                No = self.raw['IMAGING_DATA_'+fsu.upper()].data.field('DATA5')[wi[0],:]
                
                No -= No.mean(axis=0)[np.newaxis,:]
                No = No.mean(axis=1)
                
                fringesA -= No*(5 if channel==0 else 1)
                fringesB -= No*(5 if channel==0 else 1)
                fringesC -= No*(5 if channel==0 else 1)
                fringesD -= No*(5 if channel==0 else 1)
            else:
                No=None
                
            if calibrated:
                ### Sahlmann et al. A&A (2008), eq. 2
                fringesA -= self.fsu_calib[(fsu.upper(), 'DARK')][channel,0]
                fringesA /= (self.fsu_calib[(fsu.upper(), 'FLAT')][channel,0]-
                             2*self.fsu_calib[(fsu.upper(), 'DARK')][channel,0])
                fringesB -= self.fsu_calib[(fsu.upper(), 'DARK')][channel,1]
                fringesB /= (self.fsu_calib[(fsu.upper(), 'FLAT')][channel,1]-
                             2*self.fsu_calib[(fsu.upper(), 'DARK')][channel,1])
                fringesC -= self.fsu_calib[(fsu.upper(), 'DARK')][channel,2]
                fringesC /= (self.fsu_calib[(fsu.upper(), 'FLAT')][channel,2]-
                             2*self.fsu_calib[(fsu.upper(), 'DARK')][channel,2])
                fringesD -= self.fsu_calib[(fsu.upper(), 'DARK')][channel,3]
                fringesD /= (self.fsu_calib[(fsu.upper(), 'FLAT')][channel,3]-
                             2*self.fsu_calib[(fsu.upper(), 'DARK')][channel,3])
        else:
            # return only the colored channels
            fringesA = self.raw['IMAGING_DATA_'+fsu].\
                       data.field('DATA1')[wi[0],1:]
            fringesB = self.raw['IMAGING_DATA_'+fsu].\
                       data.field('DATA2')[wi[0],1:]
            fringesC = self.raw['IMAGING_DATA_'+fsu].\
                       data.field('DATA3')[wi[0],1:]
            fringesD = self.raw['IMAGING_DATA_'+fsu].\
                       data.field('DATA4')[wi[0],1:]
            if calibrated:
                fringesA -= self.fsu_calib[(fsu.upper(), 'DARK')]\
                            [1:,0][np.newaxis,:]
                fringesA /= self.fsu_calib[(fsu.upper(), 'FLAT')]\
                            [1:,0][np.newaxis,:]
                fringesB -= self.fsu_calib[(fsu.upper(), 'DARK')]\
                            [1:,1][np.newaxis,:]
                fringesB /= self.fsu_calib[(fsu.upper(), 'FLAT')]\
                            [1:,1][np.newaxis,:]
                fringesC -= self.fsu_calib[(fsu.upper(), 'DARK')]\
                            [1:,2][np.newaxis,:]
                fringesC /= self.fsu_calib[(fsu.upper(), 'FLAT')]\
                            [1:,2][np.newaxis,:]
                fringesD -= self.fsu_calib[(fsu.upper(), 'DARK')]\
                            [1:,3][np.newaxis,:]
                fringesD /= self.fsu_calib[(fsu.upper(), 'FLAT')]\
                            [1:,3][np.newaxis,:]
        
        if isolate: # isolate fringe packet
            if self.verbose:
                print 'truncating scans'

            # isolate fringe packet
            width = scan_width
            fri = (fringesA-fringesB)**2 + (fringesC-fringesD)**2
            if channel<0:
                fri = np.mean(fri, axis=1)
            w = np.where(np.abs(x-x[fri.argmax()])<=2*width)
            x = x[w]
            t = t[w]
            pdsnr=pdsnr[w]
            pd = pd[w]
            if channel<0:
                fringesA=fringesA[w[0],:]
                fringesB=fringesB[w[0],:]
                fringesC=fringesC[w[0],:]
                fringesD=fringesD[w[0],:]
            else:
                fringesA=fringesA[w]
                fringesB=fringesB[w]
                fringesC=fringesC[w]
                fringesD=fringesD[w]
                if not No is None:
                    No = No[w[0],:]
                    
        if resample and channel>=0: #resample, only for single channel
            if resample==2:
                n_opt = int(2**(np.ceil(np.log2(len(fringesA)))))
                print 'resampling from ',len(fringesA), " to ", n_opt
            else:
                n_opt = len(fringesA)
            opd = np.arange(x.min(), x.max(), np.ptp(x)/float(n_opt))

            if x[0]>x[-1]:
                fringesA = np.interp(opd, x[::-1], fringesA[::-1])
                fringesB = np.interp(opd, x[::-1], fringesB[::-1])
                fringesC = np.interp(opd, x[::-1], fringesC[::-1])
                fringesD = np.interp(opd, x[::-1], fringesD[::-1])
            else:
                fringesA = np.interp(opd, x, fringesA)
                fringesB = np.interp(opd, x, fringesB)
                fringesC = np.interp(opd, x, fringesC)
                fringesD = np.interp(opd, x, fringesD)
            # --- done resampling
        else:
            opd = x
        
        if plot:
            opd *= 1e6
            
            X = (fringesA-fringesC)/np.sqrt(2)
            Y = (fringesB-fringesD)/np.sqrt(2)
            N = (fringesA + fringesB + fringesC + fringesD )/2.
            Z = (fringesA - fringesB + fringesC - fringesD )/2.

            print 'X =', [round(x,3) for x in [1/np.sqrt(2), 0, -1/np.sqrt(2), 0 ]]
            print 'Y =', [round(x,3) for x in [0,1/np.sqrt(2), 0,-1/np.sqrt(2)]]
            
            ### Sahlmann et al 2008, Eq. 4 to 6
            __alpha = self.fsu_calib[(fsu.upper(), 'PHAS')][channel,0]
            __beta =  self.fsu_calib[(fsu.upper(), 'PHAS')][channel,1]
            __gamma = self.fsu_calib[(fsu.upper(), 'PHAS')][channel,2]
            __delta = self.fsu_calib[(fsu.upper(), 'PHAS')][channel,3]    
            Csc = (__beta*__gamma - __alpha*__delta)/2
            Xp = (X*__gamma - Y*__alpha)/Csc
            Yp = (Y*__beta - X*__delta)/Csc
            
            print 'Xp=', [round(x/Csc,3) for x in [__gamma, -__alpha, -__gamma, __alpha]]
            print 'Yp=', [round(x/Csc,3) for x in [-__delta, __beta, __delta, -__beta]]
            
            
            print 'orthogonality:', 2*(__gamma*__delta+__alpha*__beta)/\
                (np.sqrt(2*__gamma**2+2*__alpha**2)*np.sqrt(2*__beta**2+2*__delta**2))
            
            dec = pca.pca(np.transpose(np.array([fringesA, fringesB,
                                                 fringesC, fringesD])))
            for k in range(4):
                print 'C'+str(k)+'=', [round(x,3) for x in list(dec.base[:,k])],
                print np.sqrt((dec.base[:,k]**2).sum())
            plt.figure(0, figsize=(6,10))
            plt.clf()
            plt.subplots_adjust(wspace=0.05, top=0.95,hspace=0,
                                left=0.1,right=0.98,bottom=0.05)
            
            ax1 = plt.subplot(611) # -- RAW DATA
            plt.plot(opd, fringesA, 'r', label='A', linewidth=2, alpha=0.5)
            plt.plot(opd, fringesB, 'g', label='B', linewidth=2, alpha=0.5)
            plt.plot(opd, fringesC, 'b', label='C', linewidth=2, alpha=0.5)
            plt.plot(opd, fringesD, 'm', label='D', linewidth=2, alpha=0.5)
        
            plt.legend(ncol=2, prop={'size':8}, loc='upper left')

            ax2 = plt.subplot(612, sharex=ax1) # == traditional ABCD
            plt.plot(opd, N/2, label='N/2=(A+B+C+D)/4',
                        linewidth=2, alpha=0.5, color='k')
            plt.plot(opd, X,  label='X=(A-C)/$\sqrt{2}$',
                        linewidth=2, alpha=0.5, color='r')
            plt.plot(opd, Y,  label='Y=(B-D)/$\sqrt{2}$',
                        linewidth=2, alpha=0.5, color='g')
            plt.plot(opd, Z,  label='Z=(A-B+C-D)/2',
                        linewidth=2, alpha=0.5, color='b')
            plt.legend(ncol=1, prop={'size':8}, loc='upper left')
            plt.ylim(-1.5,2)
            
            plt.subplot(613, sharex=ax1, sharey=ax2) # == PRIMA corrected ABCD
            cols = ['k', 'r', 'g', 'b']
            plt.plot(opd, Xp, label='$X_p$',
                     color=cols[1], linewidth=2, alpha=0.5)
            plt.plot(opd, Yp, label='$Y_p$',
                     color=cols[2], linewidth=2, alpha=0.5)
            plt.legend(ncol=1, prop={'size':8}, loc='upper left')

            plt.subplot(614, sharex=ax1, sharey=ax2) # == PCA
            for k in range(4):
                plt.plot(opd, dec.coef[:,k]/(-2 if k==0 else 1.),
                            label='PCA: $C_'+str(k)+('$/2' if k==0 else '$'),
                            linewidth=2, alpha=0.5, color=cols[k])
            plt.ylim(-1.5,2)
            plt.legend(ncol=1, prop={'size':8}, loc='upper left')
            
            plt.subplot(615, sharex=ax1) # == SNR
            SNR1 = (Xp**2+Yp**2)
            SNR1 /= np.median(SNR1) # original
            SNR2 = (Xp**2/(__gamma**2+__alpha**2)+
                    Yp**2/(__beta**2+__delta**2))
            SNR2 /= np.median(SNR2) # corrected
            SNR3 = dec.coef[:,1]**2+dec.coef[:,2]**2
            SNR3 /= np.median(SNR3)
            
            plt.plot(opd, SNR1, '-k', label='$X_p^2+Y_p^2$')
            plt.plot(opd, SNR2, '-r', label='$X_p^2+Y_p^2$ corr.')
            
            plt.plot(opd, pdsnr, '-', color='0.2',label='PDSNR',
                        linewidth=2, alpha=0.5)
            
            plt.plot(opd, SNR3, '-y', label='$C_1^2+C_2^2$',
                        alpha=0.8, linewidth=2)
            plt.legend(prop={'size':8}, loc='upper left')
            
            # ---
            plt.subplot(616, sharex=ax1) # == Phases
            phi_orig = np.arctan2(Yp, Xp) # FSU formula
            phi_orig = np.unwrap(phi_orig)
            phi_orig -= phi_orig.mean()%(2*np.pi)    
            plt.plot(opd, phi_orig, 'r', label='arctan2($Y_p$,$X_p$)')
            
            phi = np.arctan2(X, Y) # usual ABCD
            phi = np.unwrap(phi)
            if np.sign(phi[0]-phi[-1]) != np.sign(phi_orig[0]-phi_orig[-1]):
                phi *= -1
            phi -= phi.mean()%(2*np.pi)      
            plt.plot(opd, phi, 'k', label='phase from X,Y')
            
            phi = np.arctan2(dec.coef[:,1], dec.coef[:,2]) # PCA
            phi = np.unwrap(phi)
            if np.sign(phi[0]-phi[-1]) != np.sign(phi_orig[0]-phi_orig[-1]):
                phi *= -1
            phi -= phi.mean()%(2*np.pi)      
                
            plt.plot(opd, phi, 'y', label='phase from PCA')
            
            plt.xlabel('DL feedback ($\mu m$) - '+
                str(self.raw[self.scan_opdc].data.field(self.scan_DL)[wo].mean())+' (m)')
            plt.legend(prop={'size':8}, loc='upper left')
 
            plt.figure(1)
            plt.clf()
            ax1 = plt.subplot(141)
            ax1.psd(fringesA, NFFT=len(fringesA), color='r', label='A', linewidth=2, alpha=0.5)
            ax1.psd(fringesB, NFFT=len(fringesA), color='g', label='B', linewidth=2, alpha=0.5)
            ax1.psd(fringesC, NFFT=len(fringesA), color='b', label='C', linewidth=2, alpha=0.5)
            ax1.psd(fringesD, NFFT=len(fringesA), color='m', label='D', linewidth=2, alpha=0.5)
            ax1.legend(prop={'size':8})
            ax2 = plt.subplot(142, sharex=ax1, sharey=ax1)
            ax2.psd(N/2, NFFT=len(fringesA), color='k', label='N/2', linewidth=2, alpha=0.5)
            ax2.psd(X, NFFT=len(fringesA), color='r', label='X', linewidth=2, alpha=0.5)
            ax2.psd(Y, NFFT=len(fringesA), color='g', label='Y', linewidth=2, alpha=0.5)
            ax2.psd(Z, NFFT=len(fringesA), color='b', label='Z', linewidth=2, alpha=0.5)
            ax2.legend(prop={'size':8})
            ax2.set_ylabel('')
            ax3 = plt.subplot(143, sharex=ax1, sharey=ax1)
            ax3.psd(Xp, NFFT=len(fringesA), color='r', label='Xp', linewidth=2, alpha=0.5)
            ax3.psd(Yp, NFFT=len(fringesA), color='g', label='Yp', linewidth=2, alpha=0.5)
            ax3.legend(prop={'size':8})
            ax3.set_ylabel('')
            ax4 = plt.subplot(144, sharex=ax1, sharey=ax1)
            for k in range(4):
                plt.psd(dec.coef[:,k]/(-2 if k==0 else 1.),NFFT=len(fringesA),
                            label='PCA: $C_'+str(k)+('$/2' if k==0 else '$'),
                            linewidth=2, alpha=0.5, color=cols[k])
            ax4.legend(prop={'size':8})
            ax4.set_ylabel('')

            ######################################
        else:
            return t, opd, fringesA, fringesB, fringesC, fringesD

    def scanGetTimeMaxEnv(self, i=0, calibrated=False, plot=False):
        """
        returns MJD for max of AC and BD fringes
        """
        t, opd, fringesA, fringesB, fringesC, fringesD = \
           self.getScan(i=i, isolate=False, calibrated=calibrated)
        # compute Morlet's transform
        dopd = np.abs(np.diff(opd).mean())

        wavAC = morlet.CWT(fringesA-fringesC, dopd, scale_min=1/7e-6,
                          scale_max=1/2e-6, n_per_octave=32)
        wavBD = morlet.CWT(fringesB-fringesD, dopd, scale_min=1/7e-6,
                          scale_max=1/2e-6, n_per_octave=32)

        power =  np.abs(wavBD.cwt).mean(axis=0)*np.abs(wavAC.cwt).mean(axis=0)

        tmax = fitMax(t-t.mean(), np.log(power), n=5)+t.mean()
        snr = (power.max()-np.median(power))/np.median(power)
        if plot:
            plt.figure(0)
            plt.clf()
            plt.plot(t, power, '.k')
            y = plt.ylim()
            plt.vlines(tmax, y[0], y[1], 'b')
            plt.figure(1)
            plt.clf()
            plt.imshow(np.abs(wavAC.cwt))
        return [tmax, snr]

    def scanPlotCWT(self, i=0, channel=0, isolate=True, scan_width=50e-6):
        """
        Computes Morlet phases for scan i and a given channel
        (wavelength): channel 0 is white light channel, 1..5 and the
        dispersed channels. i in [0:self.scan_nscans-1]
        """
        t, x, fringesA, fringesB, fringesC, fringesD =\
           self.getScan(i, channel, isolate=isolate, scan_width=scan_width)

        fringes_ABCD_phase(x, fringesA, fringesB, fringesC, fringesD,
                           wavelength=2.1e-6, plot=True)
        plt.figure(1)
        plt.subplot(211)
        plt.title(self.filename)
        return

    def analyseSNRfluctuations(self, fsu='FSUA', snr='PD', plot=True,
                               xlims=None, title='', normalized=False):
        """
        Check statistics of closed loop and out of the loop SNR.
        """
        t = self.raw['IMAGING_DATA_'+fsu].data.field('TIME')

        if (fsu=='FSUB' and self.insmode=='NORMAL') or \
               (fsu=='FSUA' and self.insmode=='SWAPPED'):
            wno = np.where((np.interp(t, self.raw['OPDC'].data.field('TIME'),
                                    self.raw['OPDC'].data.field('STATE'))<3))
        else:
            wno = np.where(np.interp(t, self.raw['DOPDC'].data.field('TIME'),
                                       self.raw['DOPDC'].data.field('STATE'))<3)
        if (fsu=='FSUB' and self.insmode=='NORMAL') or \
               (fsu=='FSUA' and self.insmode=='SWAPPED'):
            wftk = np.where((np.interp(t, self.raw['OPDC'].data.field('TIME'),
                                      self.raw['OPDC'].data.field('STATE'))>=7))
        else:
            wftk = np.where(np.interp(t, self.raw['DOPDC'].data.field('TIME'),
                                      self.raw['DOPDC'].data.field('STATE'))>=7)

        snrNo = self.raw['IMAGING_DATA_'+fsu].data.field(snr+'SNR')[wno[0]]
        snrFtk = self.raw['IMAGING_DATA_'+fsu].data.field(snr+'SNR')[wftk[0]]

        if plot:
            fig = plt.figure(1, figsize=(8,4))
            plt.subplots_adjust(left=0.08, right=.98)
            fig.clf()
            if normalized:
                norma = np.median(snrNo)
                plt.xlabel('normalized SNR to out-of-fringes')
            else:
                norma = 1
                plt.xlabel('SNR')

            plt.hist(snrNo/norma, bins=50, normed=True, alpha=0.5,
                        color='r', label='NOT FTK')
            hno = np.histogram(snrNo, bins=50, normed=True)
            plt.hist(snrFtk/norma, bins=50, normed=True, alpha=0.5,
                        color='g', label='FTK')
            hftk = np.histogram(snrFtk, bins=50, normed=True)

            if not xlims is None:
                plt.xlim(xlims[0], xlims[1])
            plt.title(title)
            poissonDist = lambda x,p:\
                          poisson(p['m']*p['p']).pmf(np.int_(np.floor(x*p['p'])))*p['p'] +\
                          (x*p['p']-np.floor(x*p['p']))/\
                          (np.ceil(x*p['p'])-np.floor(x*p['p']))*\
                          (poisson(p['m']*p['p']).pmf(np.int_(np.ceil(x*p['p'])))*p['p'] -
                           poisson(p['m']*p['p']).pmf(np.int_(np.floor(x*p['p'])))*p['p'])

            guess =  {'m':np.median(snrNo), 'p':1}
            X = 0.5*(hno[1][:-1]+hno[1][1:])
            fit = dpfit.leastsqFit(poissonDist, X, guess, hno[0])
            guessNo = fit['best']
            uncer = fit['uncer']
            chi2 = fit['chi2']
            model = fit['model']
            print guessNo
            print 'NOFTK; POISSON: LAMBDA', guessNo['p']*guessNo['m']
            print 'NOFTK; POISSON: STD/MEAN', 1/np.sqrt(guessNo['p']*guessNo['m'])
            plt.plot(X/norma, poissonDist(X, guessNo)*norma, '-r',
                        linewidth=3, alpha=0.8, linestyle='dashed')

            guess =  {'m':np.median(snrNo), 'p':1/10.}
            X = 0.5*(hftk[1][:-1]+hftk[1][1:]) 
            fit = dpfit.leastsqFit(poissonDist, X, guess, hftk[0])
            guess = fit['best']
            uncer = fit['uncer']
            chi2 = fit['chi2']
            model = fit['model']
            print guess
            print '  FTK; POISSON: LAMBDA', guess['p']*guess['m']
            print '  FTK; POISSON: STD/MEAN', 1/np.sqrt(guess['p']*guess['m'])
            plt.plot(X/norma, poissonDist(X, guess)*norma, '-g',
                        linewidth=3, alpha=0.8, linestyle='dashed')
            plt.legend( loc='upper left')
            #plt.xscale('log')

            print  'DIFFERENCIATION',\
                  np.abs(guess['m']-guessNo['m'])/\
                  (np.sqrt(guessNo['m']/guessNo['p']) +
                   np.sqrt(guess['m']/guess['p']))
        return

    def analysePhotometry(self, fsu='FSUA', channel=0, noFTK=False,
                          plot=False, label='', normalized=True,
                          xlims=None):
        """
        analyse photometric variations outside fringes.
        """

        t = self.raw['IMAGING_DATA_'+fsu].data.field('TIME')

        if noFTK: # check fluxes outside fringes
            if (fsu=='FSUB' and self.insmode=='NORMAL') or \
               (fsu=='FSUA' and self.insmode=='SWAPPED'):
                w = np.where((np.interp(t, self.raw['OPDC'].data.field('TIME'),
                                        self.raw['OPDC'].data.field('STATE'))<5))
            else:
                w = np.where(np.interp(t, self.raw['DOPDC'].data.field('TIME'),
                                        self.raw['DOPDC'].data.field('STATE'))<5)
            print 'noFTK:', round(100*len(w[0])/float(len(t)), 3), '%'
        else:
            w = (range(len(t)),[])

        photA = self.raw['IMAGING_DATA_'+fsu].data.field('DATA1')[w[0],channel]-\
                self.fsu_calib[(fsu, 'DARK')][channel,0]
        photB = self.raw['IMAGING_DATA_'+fsu].data.field('DATA2')[w[0],channel]-\
                self.fsu_calib[(fsu, 'DARK')][channel,1]
        photC = self.raw['IMAGING_DATA_'+fsu].data.field('DATA3')[w[0],channel]-\
                self.fsu_calib[(fsu, 'DARK')][channel,2]
        photD = self.raw['IMAGING_DATA_'+fsu].data.field('DATA4')[w[0],channel]-\
                self.fsu_calib[(fsu, 'DARK')][channel,3]

        phot0 = (photA+photB+photC+photD)/4.0
        s0 = np.argsort(phot0)
        sA = np.argsort(photA)
        sB = np.argsort(photB)
        sC = np.argsort(photC)
        sD = np.argsort(photD)

        if plot:
            fig = plt.figure(0)
            fig.clf()
            if normalized:
                plt.hist(phot0/phot0.mean(), bins=50, normed=True,
                            alpha=0.8, color='y')
                h = np.histogram(phot0/phot0.mean(), bins=50, normed=True)
                plt.xlabel('flux / mean(flux)')
            else:
                plt.hist(phot0, bins=50, normed=True, alpha=0.8, color='y')
                h = np.histogram(phot0, bins=50, normed=True)
                plt.xlabel('flux (ADU)')
            if not xlims is None:
                    plt.xlim(xlims[0], xlims[1])
            plt.title(label)
            poissonDist = lambda x,p:\
                      poisson(p['m']*p['p']).pmf(np.int_(np.floor(x*p['p'])))*p['p'] +\
                      (x*p['p']-np.floor(x*p['p']))/\
                      (np.ceil(x*p['p'])-np.floor(x*p['p']))*\
                      (poisson(p['m']*p['p']).pmf(np.int_(np.ceil(x*p['p'])))*p['p'] -
                       poisson(p['m']*p['p']).pmf(np.int_(np.floor(x*p['p'])))*p['p'])
            if not normalized:
                guess =  {'m':phot0.mean(), 'p':1/10.}
                X = 0.5*(h[1][:-1]+h[1][1:])
                fit = dpfit.leastsqFit(poissonDist, X, guess, h[0])
                guess = fit['best']
                uncer = fit['uncer']
                chi2 = fit['chi2']
                model = fit['model']
                print 'POISSON: LAMBDA', guess['p']*guess['m']
                print 'POISSON: STD/MEAN', 1/np.sqrt(guess['p']*guess['m'])
                plt.plot(X, poissonDist(X, guess), '-r', linewidth=3,
                            alpha=0.8, linestyle='dashed')
            return
        res = {'MEAN':[phot0.mean(), photA.mean(), photB.mean(),
                       photC.mean(), photD.mean()],
               'STD':[phot0.std(), photA.std(), photB.std(), photC.std(), photD.std()],
               '90-10':[phot0[s0[9*len(s0)/10]]-phot0[s0[len(s0)/10]],
                        photA[sA[9*len(sA)/10]]-photA[sA[len(sA)/10]],
                        photB[sB[9*len(sB)/10]]-photA[sB[len(sB)/10]],
                        photC[sC[9*len(sC)/10]]-photA[sC[len(sC)/10]],
                        photD[sD[9*len(sD)/10]]-photA[sD[len(sD)/10]]]}

        res['STD/MEAN'] = [res['STD'][k]/res['MEAN'][k] for k in range(5)]
        res['(90-10)/MEAN'] = [res['90-10'][k]/res['MEAN'][k] for k in range(5)]
        res['(90-10)/STD'] = [res['90-10'][k]/res['STD'][k] for k in range(5)]
        res['BEAMS']=['(A+B+C+D)/4', 'A', 'B', 'C', 'D']
        return res

    def fringeTrackingPerformance(self, state_min=5, tmin=None,
                                  tmax=None, plot=0):
        """
        tmax in s
        """
        # --- OPDC ---
        time  = self.raw['OPDC'].data.field('TIME')
        dtime = self.raw['DOPDC'].data.field('TIME')

        if tmin==None:
            tmin = max(time.min(), dtime.min())*1e-6
        if tmax==None:
            tmax = min(time.max(), dtime.max())*1e-6

        wt  = np.where((time>=tmin*1e6)*(time<=tmax*1e6))
        dwt = np.where((dtime>=tmin*1e6)*(dtime<=tmax*1e6))
        time = time[wt]
        uwrap  = self.raw['OPDC'].data.field('UWPHASE')[wt]
        duwrap = self.raw['DOPDC'].data.field('UWPHASE')[dwt]
        state  = self.raw['OPDC'].data.field('STATE')[wt]
        dstate =  self.raw['DOPDC'].data.field('STATE')[dwt]

        # in micro seconds
        delta_t = np.logspace(4, 6+min(1, np.log10((tmax-tmin)/20)),20)
        phi_rms     = []
        lock_rate   = []
        dphi_rms    = []
        dlock_rate  = []
        t_lock_rate = []
        for dt in delta_t:
            xavg, xrms, tavg    = slidop.sliding_avg_rms(uwrap, time,
                                                         dt, phaser=1)
            dxavg, dxrms, dtavg = slidop.sliding_avg_rms(duwrap, dtime,
                                                         dt, phaser=1)
            smin, smax   = slidop.sliding_min_max(state,time,dt,
                                                  computeTime=False)
            dsmin, dsmax = slidop.sliding_min_max(dstate,dtime,dt,
                                                  computeTime=False)
            ws  = np.where((smin>=state_min)*(smax<=7))
            dws = np.where((dsmin>=state_min)*(dsmax<=7))
            tws = np.where((smin>=state_min)*(smax<=7)*
                              (dsmin>=state_min)*(dsmax<=7))
            phi_rms.append(np.median(xrms[ws]))
            lock_rate.append(len(ws[0])/float(len(tavg)))
            dphi_rms.append(np.median(dxrms[dws]))
            dlock_rate.append(len(dws[0])/float(len(tavg)))
            t_lock_rate.append(len(tws[0])/float(len(tavg)))

        plt.close(plot)
        plt.figure(plot, figsize=(7,10))
        plt.clf()
        ax = plt.subplot(211)
        plt.title(self.filename+' over '+str(int(tmax-tmin))+'s')
        plt.plot(delta_t/1000.0, lock_rate,   '-',color=(1,0.5,0),
                    linewidth=2)
        plt.plot(delta_t/1000.0, dlock_rate,  '-',color=(0,0.5,1),
                    linewidth=2)
        plt.plot(delta_t/1000.0, t_lock_rate, '-',color=(0.2,1,0.2),
                    linewidth=2)
        plt.legend(['OPDC', 'd-OPDC', 'both'], loc=3)
        plt.ylabel('median lock rate (state>='+str(state_min)+')')
        plt.xscale('log')
        #plt.ylim(0,1.1)

        plt.subplot(212, sharex = ax)
        plt.plot(delta_t/1000.0, phi_rms, '-',color=(1,0.5,0), linewidth=2)
        plt.plot(delta_t/1000.0, dphi_rms, '-',color=(0,0.5,1), linewidth=2)
        #plt.hlines(np.array([1,2])/4.*np.pi,
        #              delta_t.min()/1000.0, delta_t.max()/1000.0,
        #              color='y')
        #plt.ylim(0,1.6)
        plt.xlabel('time window in ms')
        plt.ylabel('median phase RMS (rad)')

    def testPCA(self, list_of_scans=None,\
                varMax=0.9, calibrated=True, plot=False):
        """
        tries to decompose ABCDx5wavelengths using PCA
        -> not usefull
        """
        if self.scan_nscans == 0:
            print 'NO SCANS!'
            return None

        if list_of_scans is None:
            list_of_scans = range(self.scan_nscans)[2:-2]

        scan_width=100e-6
        positions = [0]
        
        for i in list_of_scans:
            t, x, fringesA, fringesB, fringesC, fringesD =\
               self.getScan(i, channel=-1,fsu=self.secondary_fsu,
                            calibrated=calibrated,
                            isolate=False,
                            scan_width=scan_width, resample=0)
            x-=x.mean()
            x*=1e6
            if i==list_of_scans[0]:
                data = np.zeros((fringesA.shape[0], 20))
                positions.append(fringesA.shape[0])
            else:
                tmp = np.zeros((data.shape[0]+fringesA.shape[0], 20))
                tmp[-data.shape[0]:,:] = data
                data= tmp
                positions.append(fringesA.shape[0]+positions[-1])

            # push last scan at the beginning of the table
            for k in range(5):
                data[:fringesA.shape[0],k*4+0] = fringesA[:,k]
                data[:fringesA.shape[0],k*4+1] = fringesB[:,k]
                data[:fringesA.shape[0],k*4+2] = fringesC[:,k]
                data[:fringesA.shape[0],k*4+3] = fringesD[:,k]
            #---
        
        # now data contains all that scans
        
        # principal component analysis
       
        p = pca.pca(data)

        # cumulative variable
        var = np.cumsum(p.var)
        if not varMax is None:
            imax = (var>=varMax).argmax()-1
        else:
            imax = len(p.var)-1
        print 'IMAX=', imax 
        #filtering
        filtered = p.comp(0)
        for k in range(imax)[1:]:
            filtered += p.comp(k)
        

        self.pca_p = p
        self.pca_filtered = []
       
        for k in range(len(positions)-1):
            tmp = filtered[positions[k]:positions[k+1], :]
            self.pca_filtered.append(tmp)
        if not plot:
            return

        # first 4 vectors
        plt.figure(0)
        plt.clf()
        plt.subplot(211)
        plt.pcolormesh(p.base)
        plt.spectral()
        plt.colorbar()

        plt.subplot(212)
        plt.plot(np.cumsum(p.var), 'o', color=(1,0.5,0))
        plt.ylabel('cumulative variance')
        plt.xlabel('eigen vector order')
        plt.hlines([varMax],\
                       0, p.data.shape[1], color='y')
        #plt.xscale('log')
        plt.figure(1, figsize=(12, 8))
        plt.clf()
        plt.subplots_adjust(hspace=0, wspace=0.02, left=0.05, right=0.99,
                               bottom=0.04, top=0.96)
        offset = 0 # --- white light of first 
        offset = 8 # this will plot pixel #3, wich is the brightest dispersed pixel
        w = range(positions[0], positions[1]) # first scan
        for k in range(4):
            #--
            if k==0:
                ax0 = plt.subplot(4,3,1+3*k)
            else:
                plt.subplot(4,3,1+3*k, sharex=ax0,
                               sharey=ax0)
            plt.plot(data[w,k+offset], 'k-', label='signal',
                        alpha=0.8)
            plt.plot(p.comp(0)[w,k+offset], color='y',
                        label='C0' if k==0 else None,
                        alpha=0.8)
            plt.ylabel(['A', 'B', 'C', 'D'][k])
            if k==0:
                plt.legend(prop={'size':8})
                plt.title('Raw signal and lowest order')
            #--
            if k==0:
                ax1 = plt.subplot(4,3,1+3*k+1, sharex=ax0)
            else:
                plt.subplot(4,3,1+3*k+1, sharex=ax0, sharey=ax1)
            plt.plot(filtered[w,k+offset]-
                        p.comp(0)[w,k+offset]-
                        p.comp(1)[w,k+offset], 'g-',
                        label='C2+...+C'+str(imax),
                        linewidth=1, alpha=0.8)
            plt.plot(data[w,k+offset]-filtered[w,k+offset],
                        color='k', alpha=0.5,
                        label='C%d+...+C%d'%(imax+1, len(p.var)-1))
            if k==0:
                plt.legend(prop={'size':8})
                plt.title(str(varMax)+' of Variance')

            #--
            if k==0:
                ax2 = plt.subplot(4,3,1+3*k+2, sharex=ax0)
            else:
                plt.subplot(4,3,1+3*k+2, sharex=ax0, sharey=ax2)
            #plt.plot(p.comp(1)[w,k+offset], 'r-', label='C1',
            #            alpha=0.6)
            plt.plot(p.comp(2)[w,k+offset], 'g-', label='C2',
                        alpha=0.6)
            plt.plot(p.comp(3)[w,k+offset], 'b-', label='C3',
                        alpha=0.6)
            plt.plot(p.comp(4)[w,k+offset], 'm-', label='C4',
                        alpha=0.6)
            if k==0:
                plt.legend(prop={'size':8})
                plt.title('"fringes" components')
        ## plt.figure(2)
        ## plt.clf()
        ## for k in range(20):
        ##     plt.subplot(4,5,k+1)
        ##     res = 0*p.comp(k)[w,0+offset]
        ##     for i in range(4):
        ##         s = p.comp(k)[w,i+offset]
        ##         res+=abs(np.fft.fft(s-s.mean()))**2
        ##     plt.plot(np.sqrt(res))
        ##     plt.legend(['C'+str(k)])
        ##     plt.title(str(p.var[k]))
        ##     plt.ylim(0,2000.)
        return

    def astrometry(self,verbose=False, plot=False, max_length_s=10.0,
                   writeOut=False, overwrite=False, store_intermediate=False,
                   max_err_um = .3, max_GD_um=1.0, sigma_clipping=None ):
        if not self.scanning:
            return astrometryFtk(self, plot=plot,
                                 max_length_s = max_length_s,
                                 writeOut=writeOut, overwrite=overwrite,
                                 store_intermediate=False,
                                 max_err_um = max_err_um,
                                 max_GD_um=max_GD_um,
                                 sigma_clipping=sigma_clipping)
        else:
            pass
        return

    def astrometryScan(self, plot=False, writeOut=False, overwrite=False,
                       store_intermediate=False, max_err_um = .5,
                       max_GD_um=None, sigma_clipping=10.0, Nmax=None,
                       min_snr=4):
        """
        max_err_um is here the minimum geometric mean of SNR  A-C and B-D
        """
        et = []
        t = [] # time at maximum
        snr = [] # SNR at max
        if Nmax==None:
            Nmax = self.scan_nscans
        for k in range(np.minimum(self.scan_nscans, int(Nmax))):
            tmp = self.scanGetTimeMaxEnv(k)
            # time of maximum:
            t.append( tmp[0])
            # err bar in t:
            et.append(10e-6)
            snr.append(tmp[1])

        t  = np.array(t)
        et = np.array(et)
        snr = np.array(snr)

        # get FTK status around the time of maximum
        ftk_status = []
        for tau in t:
            w = np.where(np.abs(self.raw['OPDC'].data.field('TIME')-tau)<1e5)
            tmp = self.raw['OPDC'].data.field('STATE')[w].mean()
            ftk_status.append(tmp)
        ftk_status = np.array(ftk_status)

        w = np.where((ftk_status>=5.0))
        print '   useful data: %d/%d' % ( len(w[0]), len(t))
        t = t[w]
        et = et[w]
        snr = snr[w]
        ftk_status=ftk_status[w]

        # FSU which is tracking
        fsu_opdc = self.getKeyword('DEL FT SENSOR')

        if self.insmode=='SWAPPED' and \
               self.getKeyword('DEL FT SENSOR')=='FSUB':
            print " -> I suspect a wrong FSU config in "+\
                  "the header based on the INS.MODE"
        if fsu_opdc == 'FSUB':
            fsu_dopdc = 'FSUA'
        else:
            fsu_dopdc = 'FSUB'

        ### group delay dOPDC
        GDdopdc = lambda x: np.interp(x,
                    self.raw['IMAGING_DATA_'+fsu_dopdc].data.field('TIME'),
                    self.raw['IMAGING_DATA_'+fsu_dopdc].data.field('GD'))

        ### group delay OPDC
        GDopdc = lambda x: np.interp(x,
                    self.raw['IMAGING_DATA_'+fsu_opdc].data.field('TIME'),
                    self.raw['IMAGING_DATA_'+fsu_opdc].data.field('GD'))

        ### primet A-B
        deltal = lambda x: np.interp(x,
                    self.raw['METROLOGY_DATA'].data.field('TIME'),
                    self.raw['METROLOGY_DATA'].data.field('DELTAL'))

        ### astrometric observable
        deltal_2 = deltal(t) + GDopdc(t) - GDdopdc(t)

        ### error bar = agreement between AC and BD
        err = 0.5*np.abs(deltal(t+et) + GDopdc(t+et) -
                         deltal(t-et) - GDopdc(t-et) )

        ### select points with good agreement
        if not min_snr is None:
            w = np.where(snr >  min_snr)
            print 'error clipping:', len(err), '->', len(w[0])
        else:
            w = range(len(err))

        deltal_2 = deltal_2[w]
        t = t[w]
        err = err[w]
        snr = snr[w]
        mjd = t*1e-6/(24*3600.0)+self.mjd_start

        if not sigma_clipping is None:
            n_before = len(mjd)
            for k in range(10):
                c = np.polyfit(mjd-mjd.mean(), deltal_2,1)
                # residuals
                res = deltal_2 - np.polyval(c,mjd-mjd.mean())
                s = res.argsort()
                pseudoSTD = 0.5*(res[s[int(.84*len(s))]] -
                                 res[s[int(.16*len(s))]])
                print k, 'STD=', pseudoSTD
                werr = np.where(np.abs(res)<sigma_clipping*pseudoSTD)
                if len(werr[0]) == len(err):
                    break
                deltal_2 = deltal_2[werr]
                t   = t[werr]
                err = err[werr]
                mjd = mjd[werr]
                snr = snr[werr]
            err = np.ones(len(err))*pseudoSTD/np.sqrt(snr-min_snr+1)
            print 'sigma clipping:', n_before, '->', len(mjd)

        if writeOut:
            hdu = pyfits.PrimaryHDU(None)
            # copy original header
            for i in self.raw[0].header.items():
                if len(i[0])>8:
                    hie = 'HIERARCH '
                else:
                    hie = ''
                if len(i)==2:
                    hdu.header.update(hie+i[0], i[1])
                elif len(i)==3:
                    hdu.header.update(hie+i[0], i[1], i[2])
            # prepare data for a binary table
            cols=[]
            cols.append(pyfits.Column(name='MJD', format='F12.5',
                                      array=mjd))
            cols.append(pyfits.Column(name='D_AL', format='E', unit='m',
                                      array=deltal_2))
            cols.append(pyfits.Column(name='D_AL_ERR', format='E', unit='m',
                                      array=err))
            hducols = pyfits.ColDefs(cols)
            hdub = pyfits.new_table(hducols)
            hdub.header.update('EXTNAME', 'ASTROMETRY_BINNED', '')

            # combine all HDUs
            thdulist = pyfits.HDUList([hdu, hdub])
            # write file
            outfile = os.path.join(
                self.dirname, self.filename.split('.')[0]+'_RED.fits')
            if overwrite and os.path.exists(outfile):
                os.remove(outfile)
            print 'writting ->', outfile
            thdulist.writeto(outfile)
            thdulist=[]
            return outfile

        if plot:
            plt.plot(13)
            plt.clf()
            ax1 = plt.subplot(211)
            plt.plot(mjd, deltal_2, '.b')
            if not sigma_clipping is None:
                plt.plot(mjd, np.polyval(c,mjd-mjd.mean()), 'b-')
                plt.plot(mjd, np.polyval(c,mjd-mjd.mean())+pseudoSTD, 'y-')
                plt.plot(mjd, np.polyval(c,mjd-mjd.mean())-pseudoSTD, 'y-')

            plt.ylabel('position of max env')
            plt.subplot(212, sharex=ax1)
            plt.plot(mjd, snr, '.k')
            #plt.yscale('log')
            plt.ylabel('SNR')
        return

    def astrometryFtk(self, plot=False, max_length_s=1.0,
                      writeOut=False, overwrite=False,
                      store_intermediate=False, max_err_um=2.,
                      max_GD_um=2.5, sigma_clipping=3.5,
                      correctA_B=False, minState=5):

        """
        extract astrometric data from dual fringe tracking data.

        max_length_s is the maximum length for a continuous measurement.

        correctA_B: correct for bug discovered in Aug 2011 by Nicolas Schuller,
        only applies to data taken in July and at the beginning of August2011
        run.
        FIXME:
        """
        # check OPDC and DOPDC
        if self.verbose:
            print '--{N, Tmin(us), Tmax(us), DT(us)}-------'
            print '%%% OPDC'
            print self.raw['OPDC'].data.field('STATE').shape,\
                  self.raw['OPDC'].data.field('TIME').min(), \
                  self.raw['OPDC'].data.field('TIME').max(), \
                  np.diff(self.raw['OPDC'].data.field('TIME')).mean()
            print '%%% DOPDC'
            print self.raw['DOPDC'].data.field('STATE').shape,\
                  self.raw['DOPDC'].data.field('TIME').min(), \
                  self.raw['DOPDC'].data.field('TIME').max(), \
                  np.diff(self.raw['DOPDC'].data.field('TIME')).mean()
            print '%%% IMAGING_DATA_FSUA'
            print self.raw['IMAGING_DATA_FSUA'].data.field('TIME').shape,\
                  self.raw['IMAGING_DATA_FSUA'].data.field('TIME').min(), \
                  self.raw['IMAGING_DATA_FSUA'].data.field('TIME').max(), \
                  np.diff(self.raw['IMAGING_DATA_FSUA'].\
                             data.field('TIME')).mean()
            print '%%% IMAGING_DATA_FSUB'
            print self.raw['IMAGING_DATA_FSUB'].data.field('TIME').shape,\
                  self.raw['IMAGING_DATA_FSUB'].data.field('TIME').min(), \
                  self.raw['IMAGING_DATA_FSUB'].data.field('TIME').max(), \
                  np.diff(self.raw['IMAGING_DATA_FSUB'].\
                             data.field('TIME')).mean()
            print '%%% METROLOGY_DATA'
            print self.raw['METROLOGY_DATA'].data.field('TIME').shape,\
                  self.raw['METROLOGY_DATA'].data.field('TIME').min(), \
                  self.raw['METROLOGY_DATA'].data.field('TIME').max(), \
                  np.diff(self.raw['METROLOGY_DATA'].\
                             data.field('TIME')).mean()
            print 'debug: MJD(PCR START)',\
                  astro.tag2mjd(self.raw[0].header['HIERARCH ESO PCR ACQ START'])
            print 'debug: MJD(PCR END)',\
                  astro.tag2mjd(self.raw[0].header['HIERARCH ESO PCR ACQ END'])
            print 'debug: MJD(PCR START+maxT)-MJD(PCR END) in s',\
                  (astro.tag2mjd(self.raw[0].header['HIERARCH ESO PCR ACQ START'])+
                  self.raw['DOPDC'].data.field('TIME').max()*1e-6/(24*3600.0)-
                  astro.tag2mjd(self.raw[0].header['HIERARCH ESO PCR ACQ END']))*24*3600.
            print '----------------------------------------'

        n_min = min(len(self.raw['OPDC'].data.field('STATE')),
                    len(self.raw['DOPDC'].data.field('STATE')))
        
        fsu_opdc = self.getKeyword('DEL FT SENSOR')
        if self.insmode=='SWAPPED' and \
            self.getKeyword('DEL FT SENSOR')=='FSUB':
            print """
            -> I suspect a wrong FSU config in the header based on the
               INS.MODE"""
        if fsu_opdc == 'FSUB':
            fsu_dopdc = 'FSUA'
        else:
            fsu_dopdc = 'FSUB'
            
        # interpolate group delays in the time frame of the OPDC
        gd1 = interp1d(self.raw['IMAGING_DATA_'+fsu_opdc].data.field('TIME'),
                       self.raw['IMAGING_DATA_'+fsu_opdc].data.field('GD'),
                       kind = 'nearest', bounds_error=False,
                       fill_value=0)(self.raw['OPDC'].data.field('TIME'))
        
        gd2 = interp1d(self.raw['IMAGING_DATA_'+fsu_dopdc].data.field('TIME'),
                       self.raw['IMAGING_DATA_'+fsu_dopdc].data.field('GD'),
                       kind = 'nearest', bounds_error=False,
                       fill_value=0)(self.raw['DOPDC'].data.field('TIME'))
        
        # list of index, in OPDC time frame, when dual fringe tracking
        # happened and both group delay were less than max_GD_um
        wall = np.where((self.raw['OPDC'].data.field('STATE')[:n_min]>=minState)*\
                        (self.raw['DOPDC'].data.field('STATE')[:n_min]>=minState)*
                        (np.abs(gd1[:n_min])<max_GD_um*1e-6)*
                        (np.abs(gd2[:n_min])<max_GD_um*1e-6))
        
        if len(wall[0])<1:
            print 'no dual FTK... sorry'
            return 0
        else:
            usefulData = len(wall[0])/\
                         float(len(self.raw['OPDC'].data.field('TIME')))
            print '  - FTK and GD limit: %5.3f%% of useful data' %\
                (100*usefulData)
            
        ftk_opdc = np.where((self.raw['OPDC'].data.field('STATE')[:n_min]>=minState))[0]
        FTKlockratio_opdc = float(len(ftk_opdc))/len(self.raw['OPDC'].data.field('STATE'))
        ftk_dopdc = np.where((self.raw['DOPDC'].data.field('STATE')[:n_min]>=minState))[0]
        FTKlockratio_dopdc = float(len(ftk_dopdc))/len(self.raw['DOPDC'].data.field('STATE'))
            
        # times where both fringe track. This is the reference time
        # for the data reduction: all binary tables will be
        # interpolated in this time frame:
        t_ftk = self.raw['OPDC'].data.field('TIME')[wall][2:-2]

        # MJD is used for telescopes parameters: ALT, AZ, ROT
        mjd_ftk = self.mjd_start + t_ftk*1e-6/(24*3600.0)

        # groups of contiguous points (1D percolation)
        g_ftk = truncateInWindows(t_ftk, delta_x = max_length_s*1e6)

        # interpolate the group delay and the metrology interpolations
        # at the time when both FSUs were tracking
        deltal = interp1d(self.raw['METROLOGY_DATA'].data.field('TIME'),
                          self.raw['METROLOGY_DATA'].data.field('DELTAL'),
                          kind = 'nearest', bounds_error=False,
                          fill_value=0)(t_ftk)
        
        if correctA_B:
            # correction of a bug discovered by Nicolas Schuhler:
            # ---------------------------------------------------
            # The correction is rather easy, just apply:
            # correctedAB = AB+2*B*dnu/nuA
            # with AB and B properly unwrapped and
            # c = 299792458;
            # nu = 227257330645727.6875;
            # nuA = nu+(38.65e6+38e6)/2;
            # nuB = nu+(-40e6-39.55e6)/2;
            # dnu = nuA-nuB;
            c = 299792458
            nu = 227257330645727.6875
            nuA = nu+(38.65e6 + 38e6)/2
            nuB = nu+(-40e6 - 39.55e6)/2
            dnu = nuA-nuB
            corNSc = 2*dnu/nuA*(interp1d(self.raw['OPDC'].data.field('TIME'),
                                    self.raw['OPDC'].data.field(self.DLtrack),
                                    kind = 'nearest', bounds_error=False,
                                    fill_value=0)(t_ftk)- 28.0) # KLUDGE!
            print 'Nicolas\'s correction:', corNSc.min(), corNSc.max()
            deltal += corNSc
            # ---------------------------------------------------

        if self.glitches['NGLITABFCO'] !=0: # correct for A-B glitches
            print '  | --> correcting for A-B glitches'
            # assumes these are the good points
            wmain = np.where(np.abs(deltal-np.median(deltal))<100e-6)
            wplus = np.where(np.abs(deltal-np.median(deltal)+
                                    self.metJumpSize)<100e-6)
            wminus = np.where(np.abs(deltal-np.median(deltal)-
                                     self.metJumpSize)<100e-6)
            if len(wplus[0])>0:
                print '   correcting overflow', len(wplus[0]),\
                    'out of', len(deltal)
                deltal[wplus] += self.metJumpSize
            if len(wminus[0])>0:
                print '   correcting overflow', len(wminus[0]),\
                    'out of', len(deltal)
                deltal[wminus] -= self.metJumpSize

        # -- -FSUB metrology
        m_fsub = interp1d(self.raw['METROLOGY_DATA_FSUB'].data.field('TIME'),
                          self.raw['METROLOGY_DATA_FSUB'].data.field('DELTAL'),
                          kind = 'nearest', bounds_error=False,
                          fill_value=0)(t_ftk)
        # -- group delays:
        GDopdc = interp1d(self.raw['IMAGING_DATA_'+fsu_opdc].data.field('TIME'),
                          self.raw['IMAGING_DATA_'+fsu_opdc].data.field('GD'),
                          kind = 'nearest', bounds_error=False,
                          fill_value=0)(t_ftk)
        GDdopdc = interp1d(self.raw['IMAGING_DATA_'+fsu_dopdc].data.field('TIME'),
                           self.raw['IMAGING_DATA_'+fsu_dopdc].data.field('GD'),
                           kind = 'nearest', bounds_error=False,
                           fill_value=0)(t_ftk)
        # -- phase delays:
        PDopdc = interp1d(self.raw['IMAGING_DATA_'+fsu_opdc].data.field('TIME'),
                          self.raw['IMAGING_DATA_'+fsu_opdc].data.field('PD'),
                          kind = 'nearest', bounds_error=False,
                          fill_value=0)(t_ftk)
        PDdopdc = interp1d(self.raw['IMAGING_DATA_'+fsu_dopdc].data.field('TIME'),
                           self.raw['IMAGING_DATA_'+fsu_dopdc].data.field('PD'),
                           kind = 'nearest', bounds_error=False,
                           fill_value=0)(t_ftk)
        # -- binning
        # I checked this formula by reducing the scatter
        # it is always: deltal - FSUB + FSUA, because deltal is A-B
        deltal_2 = deltal
        if 'SWAP' in self.insmode:
            deltal_2 += -GDdopdc + GDopdc # CORRECT
            # path measured by metrology, in A and B 
            pathA = deltal_2 - m_fsub + GDopdc
            pathB = -m_fsub + GDdopdc
        else:
            deltal_2 +=  GDdopdc - GDopdc
            pathA = deltal_2 - m_fsub + GDdopdc
            pathB = -m_fsub + GDopdc

        if store_intermediate: # create variables to store intermediate results
            m_fsub = interp1d(self.raw['METROLOGY_DATA_FSUB'].data.field('TIME'),
                            self.raw['METROLOGY_DATA_FSUB'].data.field('DELTAL'),
                            kind = 'nearest', bounds_error=False,
                            fill_value=0)(t_ftk)
            m_fsub = unwrap(m_fsub, .346)
            DLfdback = interp1d(self.raw['OPDC'].data.field('TIME'),
                                # self.DLtrack was 'DLFDBACK'
                                self.raw['OPDC'].data.field(self.DLtrack),
                                kind = 'nearest', bounds_error=False,
                                fill_value=0)(t_ftk)
            DDLfdback = interp1d(self.raw['DOPDC'].data.field('TIME'),
                                 # self.DLtrack was 'DLFDBACK'
                                 self.raw['DOPDC'].data.field(self.DDLtrack),
                                 kind = 'nearest', bounds_error=False,
                                 fill_value=0)(t_ftk)
            # store intermediate data for dual FTK times
            print ' -> storing .ftk_ in the object'
            self.ftk_t = t_ftk
            self.ftk_deltal = deltal
            self.ftk_m_fsub = m_fsub
            self.ftk_DLfdback = DLfdback
            self.ftk_DDLfdback = DDLfdback
            self.ftk_GDopdc = GDopdc
            self.ftk_GDdopdc = GDdopdc
            self.ftk_PDopdc = PDopdc
            self.ftk_PDdopdc = PDdopdc
            self.ftk_dopd_reduced = deltal_2
            self.ftk_pathA = pathA
            self.ftk_pathB = pathB
        
        deltal_a   = [] # avg position in um
        deltal_s   = [] # slope
        deltal_err = [] # err on avt pos
        tmin_r     = []
        tmax_r     = []
        tavg_r     = []

        # --
        for k in range(int(g_ftk.max()+1)):
            # for each group
            w = np.where(g_ftk==k)
            if len(w[0])>5:          
                ### polynomial fit to remove slope in order to compute
                ### the scatter, to be used as error bar
                t0 = t_ftk[w].mean()
                coef = np.polyfit(t_ftk[w]-t0, deltal_2[w], 1)
                res = deltal[w] - np.polyval(coef, t_ftk[w]-t0)
                ### keep 90% best points and redo fit
                wp = np.abs(res).argsort()[:int(len(w[0])*0.9)]
                ### redo a linear fit
                coef = np.polyfit(t_ftk[w][wp]-t0, deltal_2[w][wp], 1)
                res = deltal[w] - np.polyval(coef,t_ftk[w]-t0)

                deltal_err.append(
                    (deltal[w] - np.polyval(coef,t_ftk[w]-t0)).std()*1e6)
                deltal_a.append(coef[1]*1e6) # average
                deltal_s.append(coef[0]*1e6) # local slope
                tavg_r.append(t_ftk[w].mean())
                tmin_r.append(t_ftk[w].min())
                tmax_r.append(t_ftk[w].max())
            else:
                deltal_err.append(deltal_2[w].std()*1e6)
                deltal_a.append(deltal_2[w].mean()*1e6)
                deltal_s.append(0.0)
                tavg_r.append(t_ftk[w].mean())
                tmin_r.append(t_ftk[w].min())
                tmax_r.append(t_ftk[w].max())

        deltal_err = np.array(deltal_err)
        s = deltal_err.argsort()
        print '  average local slope: %5.3f +/- %5.3f (um/s)'%(
            np.mean(np.nonzero(deltal_s)),  np.std(np.nonzero(deltal_s)))
        print '  | 1 quartile error bar=', round(deltal_err[s[len(s)//4]],3), '(um)'
        print '  | median     error bar=', round(np.median(deltal_err),3), '(um)'
        print '  | 3 quartile error bar=', round(deltal_err[s[3*len(s)//4]],3), '(um)'
        if max_err_um>0:
            werr = np.where(deltal_err<=max_err_um)
        else:
            werr = np.where(deltal_err)
            
        if len(werr[0])<len(deltal_err):
            print '  |', len(deltal_err)-len(werr[0]),\
                  'points rejected (err bar) out of',\
                  len(deltal_err)

        deltal_err = deltal_err[werr]
        deltal_a   = np.array(deltal_a)[werr]
        deltal_s   = np.array(deltal_s)[werr]
        tavg_r     = np.array(tavg_r)[werr]
        tmin_r     = np.array(tmin_r)[werr]
        tmax_r     = np.array(tmax_r)[werr]

        # -- sigma clipping
        if not sigma_clipping is None:
            n_before = len(deltal_a)
            for k in range(10):
                if len(deltal_a)<10:
                    break
                c = np.polyfit(tavg_r-tavg_r.mean(),deltal_a,1)
                # remove linear trend
                res = deltal_a - np.polyval(c,tavg_r-tavg_r.mean())
                # compute pseudo STD
                sor = res[res.argsort()]
                pseudo_std = (sor[int(0.84*len(sor))]-sor[int(0.16*len(sor))])/2.
                werr = np.where(np.abs(res)<sigma_clipping*pseudo_std)
                if len(werr[0])==len(deltal_err):
                    break
                deltal_err = deltal_err[werr]
                deltal_a   = np.array(deltal_a)[werr]
                deltal_s   = np.array(deltal_s)[werr]
                tavg_r     = np.array(tavg_r)[werr]
                tmin_r     = np.array(tmin_r)[werr]
                tmax_r     = np.array(tmax_r)[werr]

            print '  | ', n_before-len(deltal_a),\
                  'points rejected (sigma clipping) out of',\
                  n_before

        #print '  results in variables starting with .deltaL '
        self.deltaL_m     = deltal_a*1e-6
        self.deltaL_err   = deltal_err*1e-6
        self.deltaL_t     = np.array(tavg_r)
        self.deltaL_dt    = np.array(0.5*(tmax_r-tmin_r))
        self.deltaL_MJD   = self.mjd_start + np.array(tavg_r*1e-6/(24.*3600))
        self.deltaL_lst   = self.timeStamp2LST(tavg_r)

        if writeOut:
            hdu = pyfits.PrimaryHDU(None)
            # copy original header
            for i in self.raw[0].header.items():
                if len(i[0])>8:
                    hie = 'HIERARCH '
                else:
                    hie = ''
                if len(i)==2:
                    hdu.header.update(hie+i[0], i[1])
                elif len(i)==3:
                    hdu.header.update(hie+i[0], i[1], i[2])
            # add some reduction parameters:
            hie = 'HIERARCH ESO '
            hdu.header.update(hie+'RED ORIG', 'DRS amerand', 'data reduction origin')
            hdu.header.update(hie+'RED PAR TIME_WIN', max_length_s,
                              'time window average, in s')
            hdu.header.update(hie+'RED PAR MAX_GD', max_GD_um,
                              'max goup delay, in mcrons')
            hdu.header.update(hie+'RED PAR MIN_STATE', minState,
                              'min value for state machine')
            hdu.header.update(hie+'RED PAR MAX_ERR', max_err_um,
                              'max dOP RMS over time window, in microns')
            hdu.header.update(hie+'RED PAR SIG_CLIP', sigma_clipping,
                              'sigma clipping')
            hdu.header.update(hie+'RED PRO LR_OPDC', round(FTKlockratio_opdc,3),
                              'locking ratio OPDC')
            hdu.header.update(hie+'RED PRO LR_DOPDC', round(FTKlockratio_dopdc,3),
                              'locking ratio DOPDC')
            hdu.header.update(hie+'RED PRO FRAC_USEFUL', round(usefulData,3),
                              'fraction of useful data')
            # prepare binned data for a binary table
            cols=[]
            cols.append(pyfits.Column(name='MJD', format='F12.5',
                                      array=self.deltaL_MJD))
            cols.append(pyfits.Column(name='LST', format='F12.5',
                                      array=self.deltaL_lst))
            cols.append(pyfits.Column(name='D_AL', format='E', unit='m',
                                      array=self.deltaL_m))
            cols.append(pyfits.Column(name='D_AL_ERR', format='E', unit='m',
                                      array=self.deltaL_err))
            if not self.AT3derot_MJD is None:
                cols.append(pyfits.Column(name='ROT3', format='E', unit='deg',
                                          array=self.AT3derot_MJD(self.deltaL_MJD)))
            if not self.AT4derot_MJD is None:
                cols.append(pyfits.Column(name='ROT4', format='E', unit='deg',
                                          array=self.AT4derot_MJD(self.deltaL_MJD)))
            if not self.AT3az_MJD is None:
                cols.append(pyfits.Column(name='AZ3', format='E', unit='deg',
                                          array=self.AT3az_MJD(self.deltaL_MJD)))
            if not self.AT4az_MJD is None:
                cols.append(pyfits.Column(name='AZ4', format='E', unit='deg',
                                          array=self.AT4az_MJD(self.deltaL_MJD)))
            if not self.AT3alt_MJD is None:
                cols.append(pyfits.Column(name='ALT3', format='E', unit='deg',
                                          array=self.AT3alt_MJD(self.deltaL_MJD)))
            if not self.AT4alt_MJD is None:
                cols.append(pyfits.Column(name='ALT4', format='E', unit='deg',
                                          array=self.AT4alt_MJD(self.deltaL_MJD)))
            hducols = pyfits.ColDefs(cols)
            hdub = pyfits.new_table(hducols)
            hdub.header.update('EXTNAME', 'ASTROMETRY_BINNED', '')

            # prepare not binned data for a binary table
            cols=[]
            cols.append(pyfits.Column(name='MJD', format='F12.5',
                                      array=mjd_ftk))
            cols.append(pyfits.Column(name='D_AL', format='E', unit='m',
                                      array=deltal_2))
            cols.append(pyfits.Column(name='PRIMET_D_AL', format='E', unit='m',
                                       array=deltal))
            cols.append(pyfits.Column(name='GDOPDC', format='E', unit='m',
                                       array=GDopdc))
            cols.append(pyfits.Column(name='GDDOPDC', format='E', unit='m',
                                       array=GDdopdc))
            if not self.AT3derot_MJD is None:
                cols.append(pyfits.Column(name='ROT3', format='E', unit='deg',
                                          array=self.AT3derot_MJD(mjd_ftk)))
            if not self.AT4derot_MJD is None:
                cols.append(pyfits.Column(name='ROT4', format='E', unit='deg',
                                          array=self.AT4derot_MJD(mjd_ftk)))

            hducols = pyfits.ColDefs(cols)
            hduc = pyfits.new_table(hducols)
            hduc.header.update('EXTNAME', 'ASTROMETRY_RAW', '')

            # combine all HDUs
            thdulist = pyfits.HDUList([hdu, hdub, hduc])

            # write file
            outfile = os.path.join(
                self.dirname, self.filename.split('.')[0]+'_RED.fits')
            if overwrite and os.path.exists(outfile):
                os.remove(outfile)
            thdulist.writeto(outfile)
            thdulist=[]
            return outfile

        if plot: # ------------------------------------------------
            plt.figure(0, figsize=(16,6))
            plt.clf()
            ax1 = plt.subplot(311)
            plt.plot(t_ftk[::5], deltal_2[::5]*1e6, '-',
                        color=(0,0.5,1))
            plt.plot(t_ftk[::5], deltal[::5]*1e6, '-',
                        color=(1,0.5,0), linewidth=2)
            plt.errorbar(tavg_r, deltal_a, yerr=deltal_err,
                            color='g', fmt='o')
            for k in range(len(tavg_r)):
                t = np.array([tmin_r[k], tmax_r[k]])
                plt.plot(t, deltal_a[k]+(t-tavg_r[k])*deltal_s[k], 'g-')
            plt.ylabel('PRIMET (um)')
            plt.title(self.filename)
            #plt.subplot(412)
            #plt.plot(t_ftk, g_ftk, '-', color=(0,0.5,1))

            plt.subplot(312, sharex=ax1)
            plt.plot(t_ftk[::10], GDopdc[::10]*1e6,\
                        '-', color=(1,0.5,0))
            plt.ylabel('GD OPDC (um)')

            plt.subplot(313, sharex=ax1)
            plt.plot(t_ftk[::10], GDdopdc[::10]*1e6,\
                        '-', color=(1,0.5,0))
            plt.ylabel('GD DOPDC (um)')
            #-------------------------------------------------------
        return

    def correctOverflows(self):
        """
        the idea is to correct A-B when a glich occurs
        """
        plt.figure(11)
        plt.clf()
        plt.plot(self.raw['METROLOGY_DATA'].data.field('TIME'),
                    self.raw['METROLOGY_DATA'].data.field('DELTAL') )
        plt.hlines([np.median(self.raw['METROLOGY_DATA'].data.field('DELTAL')),
                       np.median(self.raw['METROLOGY_DATA'].data.field('DELTAL'))+self.metJumpSize,
                       np.median(self.raw['METROLOGY_DATA'].data.field('DELTAL'))-self.metJumpSize],
                      self.raw['METROLOGY_DATA'].data.field('TIME').min(),
                      self.raw['METROLOGY_DATA'].data.field('TIME').max(), color='r')
        return

    def countTimeJumps(self):
        toTest=['OPDC', 'DOPDC', 'IMAGING_DATA_FSUA', 'IMAGING_DATA_FSUB',
                'METROLOGY_DATA', 'METROLOGY_DATA_FSUB']
        for t in toTest:
            dt = np.diff(self.raw[t].data.field('TIME'))
            dt_min = dt.min()
            print '_'*4, t, ' DT=', round(dt_min*1e-3, 3), \
                  'ms '+'_'*(20-len(t))
            print ' DT:', len(np.where(abs(dt/dt_min-1)<0.1)[0])/float(len(dt))
            print '2DT:', len(np.where(abs(dt/dt_min-2)<0.1)[0])/float(len(dt))
            print '3DT:', len(np.where(abs(dt/dt_min-3)<0.1)[0])/float(len(dt))
            print '4DT:', len(np.where(abs(dt/dt_min-4)<0.1)[0])/float(len(dt))
        return
    
    def labResponseExtract(self):
        """
        Analyse files like 'PACMAN_LAB_FSURESPONSE_NNN_XXXX'.fits

        extract scans in self.labScans[n]['FSUA'] and
        self.labScans[n]['FSUB'], where n designates the nth scan.

        each containts fields: ['A', 'C', 'B', 'D', 'GD', 'PD',
        'TIME', 'DELTAL', 'GDSNR', 'PDSNR'] ['A', 'C', 'B', 'D']
        contains 6 fringes set: 1 white and 5 dispersed. all the
        variables are reinterpolated in the same time frame, namely
        the one of 'IMAGING_DATA_FSU'

        example, if 'a' is a is loaded with prima.drs():
        plot(a.labScans[0]['FSUA']['DELTAL'],a.labScans[0]['FSUA']['A'])
        
        """
        if not self.getKeyword('TPL ID')=='PACMAN_cal_Lab_FSUResponse':
            print 'NOT a PACMAN_cal_Lab_FSUResponse file... nothing to be done'
            return
        # determine the number of scans:
        Nscans = int(self.raw['IMAGING_DATA_FSUA'].data.field('STEPPING_PHASE').max())
        self.labScans = []
        for n in range(Nscans)[1:]:
            s = {}
            for f in ['A', 'B']:
                w = np.where(self.raw['IMAGING_DATA_FSU'+f].data.field('STEPPING_PHASE')==n)
                tmp = {}
                tmp['TIME']=self.raw['IMAGING_DATA_FSU'+f].data.field('TIME')[w]
                if f=='A':
                    tmp['DELTAL']=np.interp(self.raw['IMAGING_DATA_FSU'+f].data.field('TIME')[w],
                                            self.raw['METROLOGY_DATA'].data.field('TIME'),
                                            self.raw['METROLOGY_DATA'].data.field('DELTAL'))+\
                                  np.interp(self.raw['IMAGING_DATA_FSU'+f].data.field('TIME')[w],
                                            self.raw['METROLOGY_DATA_FSUB'].data.field('TIME'),
                                            self.raw['METROLOGY_DATA_FSUB'].data.field('DELTAL'))
                else:
                    tmp['DELTAL']=-np.interp(self.raw['IMAGING_DATA_FSU'+f].data.field('TIME')[w],
                                            self.raw['METROLOGY_DATA_FSUB'].data.field('TIME'),
                                            self.raw['METROLOGY_DATA_FSUB'].data.field('DELTAL'))
                tmp['A'] = self.raw['IMAGING_DATA_FSU'+f].data.field('DATA1')[w[0],:]
                tmp['B'] = self.raw['IMAGING_DATA_FSU'+f].data.field('DATA2')[w[0],:]
                tmp['C'] = self.raw['IMAGING_DATA_FSU'+f].data.field('DATA3')[w[0],:]
                tmp['D'] = self.raw['IMAGING_DATA_FSU'+f].data.field('DATA4')[w[0],:]
                tmp['GD']=  self.raw['IMAGING_DATA_FSU'+f].data.field('GD')[w]
                tmp['GDSNR']=  self.raw['IMAGING_DATA_FSU'+f].data.field('GDSNR')[w]
                tmp['PD']=  self.raw['IMAGING_DATA_FSU'+f].data.field('PD')[w]
                tmp['PDSNR']=  self.raw['IMAGING_DATA_FSU'+f].data.field('PDSNR')[w]
                s['FSU'+f]=tmp
            self.labScans.append(s)
        return
    
    def lab_dispersion(self, n=1, fsu='FSUA', channel=3):
        fringes = self.labScans[n][fsu]['A'][:,channel]
        opd = self.labScans[n][fsu]['DELTAL']
        fringes = fringes[opd.argsort()]
        opd = opd[opd.argsort()]
        
        opd_ = np.linspace(opd.min(), opd.max(), len(opd))
        fringes_ = np.interp(opd_, opd, fringes)
        fringes_ -= fringes_.mean()
        
        imax = int(0.5*(fringes_.argmax()+fringes_.argmin()))
        if imax < len(fringes_)/2:
            fringes_ = fringes_[:imax*2]
            opd_ = opd_[:imax*2]
        else:
            fringes_ = fringes_[-(len(fringes_)-imax)*2:]
            opd_ = opd_[-(len(opd_)-imax)*2:]
        
        plt.figure(10)
        plt.clf()
        plt.plot(opd_, fringes_)
    
        FFT = np.fft.fft(np.roll(fringes_, len(fringes_)/2))
        FFT = FFT[:len(FFT)/2]
        w2 = np.arange(-4,4)+np.abs(FFT).argmax()
        phi = np.unwrap(np.angle(FFT)[w2])
        ephi = 0.1*(np.abs(FFT)**2)[w2].max()/(np.abs(FFT)[w2]**2)
        
        a = dpfit.leastsqFit(dpfit.dpfunc.polyN, w2-w2.mean(),
                             {'a0':0.0, 'a1':0.0, 'a2':0.0},
                             phi, err=ephi)['best']
        print a
        plt.figure(11)
        plt.clf()
        ax1 = plt.subplot(211)
        plt.plot(np.abs(FFT)**2)
        plt.ylabel('FFT power')
        
        plt.subplot(212, sharex=ax1)
        plt.errorbar(w2, phi, marker='o',
                     yerr=ephi, color='r')
        plt.plot(w2, dpfit.dpfunc.polyN(w2-w2.mean(),a), linewidth=4, alpha=0.5,color='y')
        plt.xlim(np.abs(FFT).argmax()-10, np.abs(FFT).argmax()+10)
        plt.ylabel('FFT phase (rad)')
        
# ==========================================================

class env():
    def __init__(self, filename):
        self.raw = pyfits.open(filename)
        self.vars = {}
        ed = self.raw['ENVIRONMENT_DESCRIPTION'].data.field
        for k, n  in enumerate(ed('NAME')):
            self.vars[n.strip()]=(ed('ID')[k], ed('TYPE')[k].strip(),
                                  ed('LOCATION')[k].strip(),
                                  ed('UNIT')[k], ed('KEYWORD')[k])
        return
    def listVars(self):
        return self.vars.keys()

    def interpVarMJD(self,varname):
        sensor = self.raw['ENVIRONMENT'].data.field('SENSOR')
        time   = self.raw['ENVIRONMENT'].data.field('TIME')
        value  = self.raw['ENVIRONMENT'].data.field('VALUE')
        mjd0 = self.raw[0].header['MJD-OBS']
        mjd = mjd0+time/(3600*24.)
        w = np.where(sensor==self.vars[varname][0])
        if len(w[0])>1:
            return lambda x: np.inter(x, mjd[w], value[w])
        else:
            print 'error!'

    def plotData(self, varname, clf=False):
        sensor = self.raw['ENVIRONMENT'].data.field('SENSOR')
        time   = self.raw['ENVIRONMENT'].data.field('TIME')
        value  = self.raw['ENVIRONMENT'].data.field('VALUE')
        mjd0 = self.raw[0].header['MJD-OBS']
        mjd = mjd0+time/(3600*24.)
        w = np.where(sensor==self.vars[varname][0])
        if len(w[0])>0:
            if clf:
                plt.clf()
            plt.plot(mjd[w], value[w], '.', label=varname)
            plt.legend()
        else:
            print 'no data points for', varname

    def plotTemps(self, mjd_min=None, mjd_max=None):
        sensor = self.raw['ENVIRONMENT'].data.field('SENSOR')
        time   = self.raw['ENVIRONMENT'].data.field('TIME')
        value  = self.raw['ENVIRONMENT'].data.field('VALUE')
        mjd0 = self.raw[0].header['MJD-OBS']
        mjd = mjd0+time/(3600*24.)
        plt.figure(0)
        plt.clf()
        for k in self.vars.keys():
            if self.vars[k][1]=='TEMP' and k[:4]=='VLTI':
                w = np.where(sensor==self.vars[k][0])
                if len(w[0])>1:
                    plt.plot(mjd[w], value[w], label=k)
        plt.legend(ncol=3)
        return

class pssRecorder():
    """
    at3 = oneFile(os.path.join(directory,
                 'pssguiRecorder_lat3fsm_'+date+'.dat'))

    at4 = oneFile(os.path.join(directory,
                 'pssguiRecorder_lat4fsm_'+date+'.dat'))

    time = np.linspace(max(at3['time'].min(),at4['time'].min()),
                          min(at3['time'].max(),at4['time'].max()),
                          len(at3['time']))
    """
    def __init__(self, filename):
        f = open(filename)
        lines = f.read().split('\n')
        f.close()
        # read variables in file
        var_names = ['index', 'time', 'date', 'time_stamp']
        k = 4
        for l in lines:
            if '# '+str(k)+' ' in l:
                tmp = l.split()[4:]
                tmp = reduce(lambda x, y: x+y, tmp)
                var_names.append(tmp)
                k+=1
            if 'start time:' in l:
                break
        # set up
        data = filter(lambda x: x[0]!='#' if len(x)>0 else False, lines)
        del lines
        rec = [[] for k in range(len(var_names))]
        for i in range(len(data)):
            for k in range(len(var_names)):
                try:
                    rec[k].append(float(data[i].split()[k]))
                except:
                    rec[k].append(data[i].split()[k])
        res = {}
        for k in range(len(var_names)):
            res[var_names[k]] = np.array(rec[k])
        self.data = res
        del rec
        y = np.array([float(x.split('-')[0]) for x in self.data['date']])
        if any(y<50):
            y += 2000 # why use only 2 digit!!!!
        m = np.array([float(x.split('-')[1]) for x in self.data['date']])
        d = np.array([float(x.split('-')[2]) for x in self.data['date']])
        hh = np.array([float(x.split(':')[0]) for x in self.data['time_stamp']])
        mm = np.array([float(x.split(':')[1]) for x in self.data['time_stamp']])
        ss = np.array([float(x.split(':')[2]) for x in self.data['time_stamp']])
        del data
        a = (14-np.int_(m))/12
        y += 4800 - a
        m += 12*a - 3
        self.mjd = d + (153*np.int_(m)+2)/5 + 365*np.int_(y) + \
                   np.int_(y)/4 - np.int_(y)/100 + np.int_(y)/400 - 32045
        self.mjd += hh/24. + mm/(24*60.) + ss/(24*3600.0)
        self.mjd -=2400001.0
        return

def fileExplorer(directory, pssRecorderDir=None,
                 nightStartUT=None, nightEndUT=None,
                 dateAsm=None):
    """
    """
    if not os.path.isdir(directory):
        print 'ERROR:', directory, 'does not exist'
        return
    if not pssRecorderDir is None and os.path.isdir(pssRecorderDir):
        print 'ERROR:', pssRecorderDir, 'does not exist'
        return
    files = filter(lambda x: 'PACMAN' in x and '.fits' in x and
                   ('OBJ' in x or 'SKY' in x),
                   os.listdir(directory))
    mjd_start = []
    mjd_end = []
    swap = []
    nightStartMjd = None
    nightEndMjd = None
    for f in files:
        x = pyfits.open(os.path.join(directory, f))
        mjd_start.append(astro.tag2mjd(x[0].header['HIERARCH ESO PCR ACQ START']))
        mjd_end.append(astro.tag2mjd(x[0].header['HIERARCH ESO PCR ACQ END']))
        if not nightStartUT is None and  nightStartMjd is None:
            nightStartMjd=astro.tag2mjd(
                x[0].header['HIERARCH ESO PCR ACQ START'].split('T')[0]+
                'T'+nightStartUT)
            if int(x[0].header['HIERARCH ESO PCR ACQ START'].
                   split('T')[1].split(':')[0])<20:
                nightStartMjd-=1

        if not nightEndUT is None and  nightEndMjd is None:
            nightEndMjd=astro.tag2mjd(
                x[0].header['HIERARCH ESO PCR ACQ START'].split('T')[0]+
                'T'+nightEndUT)
            if int(x[0].header['HIERARCH ESO PCR ACQ START'].
                   split('T')[1].split(':')[0])>12:
                nightEndMjd+=1
        try:
            if 'OBJ' in f:
                if 'SWAP' in self.insmode:
                    swap.append(-1)
                else:
                    swap.append(1)
            else:
                swap.append(0)
        except:
            swap.append(0)
        x.close()

    # -- plot
    plt.close(0)
    plt.figure(0, figsize=(12,6))
    if not dateAsm is None:
        ax = plt.subplot(311)
    else:
        ax = plt.axes([.05,.1,.9,.9])

    if not nightEndMjd is None and not nightStartMjd is None:
        rect = Rectangle((nightStartMjd, -0.6),
                          nightEndMjd-nightStartMjd , 2.2,
                         facecolor='k', edgecolor='k',
                         alpha=0.3)
        plt.gca().add_patch(rect)

    for k in range(len(files)):
        if 'OBJ_SCAN' in files[k]:
            facecolor='orange'
            edgecolor='orange'
        elif 'OBJ_ASTRO' in files[k]:
            facecolor='green'
            edgecolor='green'
        elif 'SKY' in files[k]:
            facecolor='blue'
            edgecolor='blue'
        else:
            facecolor='0.6'
            edgecolor='0.8'

        rect = Rectangle((mjd_start[k], swap[k]*0.5),
                          mjd_end[k]-mjd_start[k] , 1,
                         facecolor=facecolor,
                         edgecolor=edgecolor)
        plt.gca().add_patch(rect)
    plt.show()
    plt.ylim(-0.6,1.6)
    plt.title(directory)

    if not dateAsm is None:
        asm = paranal.asm(dateAsm)
        plt.subplot(312, sharex=ax)

        plt.plot(asm.data['S'][0], asm.data['S'][1], 'ob',
                    label='Seeing (\")', markersize=5, alpha=0.5)
        plt.plot(asm.data['R'][0], asm.data['R'][1], '.k',
                    label='tau0 (ms)', markersize=5)
        plt.hlines(1.2, np.array(asm.data['S'][0]).min(),
                       np.array(asm.data['S'][0]).max(), color='r',
                      linewidth=4, alpha=0.5)
        plt.hlines(0.6, np.array(asm.data['S'][0]).min(),
                       np.array(asm.data['S'][0]).max(), color='g',
                      linewidth=4, alpha=0.5)

        plt.legend(loc=2)
        plt.subplot(313, sharex=ax)
        plt.ylabel('Wind')
        plt.plot(asm.data['W'][0], asm.data['W'][1],color='k',
                      linewidth=2)
        plt.hlines(12, np.array(asm.data['W'][0]).min(),
                      np.array(asm.data['W'][0]).max(), color='y',
                      linewidth=4, alpha=0.5)
        plt.hlines(18, np.array(asm.data['W'][0]).min(),
                      np.array(asm.data['W'][0]).max(), color='r',
                      linewidth=4, alpha=0.5)
        plt.ylim(0,20)

    plt.xlim(np.array(mjd_start).min()-.01,
                np.array(mjd_start).max()+.01)
    if not nightEndMjd is None and not nightStartMjd is None:
        if nightStartMjd<np.array(mjd_start).min():
            plt.xlim(xmin=nightStartMjd-0.01)
        if nightEndMjd>np.array(mjd_end).max():
            plt.xlim(xmax=nightEndMjd+0.01)

    plt.xlabel('MJD')
    return

def nonZeroMean(vector):
    vector = np.array(vector)[np.nonzero(vector)]
    return vector.mean()

def strictlyPositiveMean(vector):
    vector = np.array(vector)
    vector = vector[vector>0]
    return vector.mean()


def qc(filename, legacy=False, querySimbad=False):
    """
    this function opens a PACMAN fits file and returns a dictionnary
    containing relevant parameters for QC.

    the function skips invalid files instead of crashing.

    legacy=True reads old files
    """
    if not os.path.isfile(filename):
        print 'SKIPPING: file', filename, 'does not exist'
        return
    res = {} # the final result
    f = pyfits.open(filename)
    if f[0].header['INSTRUME'].strip() != 'PACMAN':
        print 'SKIPPING:', filename, 'is not a INSTRUME==\'PACAMA\' file!'
        return
    # -- propagate keywords:
    keywords = ['DATE-OBS', 'MJD-OBS']
    for k in keywords:
        try:
            res[k] = f[0].header[k]
        except:
            pass
    # -- propagate HIERARCH keywords:
    Hkeywords = ['TPL ID',
                 'INS MODE',
                 'ISS AMBI WINDSP']
    if legacy:
        Hkeywords.extend(['OCS PRESET SS ALPHA',
                          'OCS PRESET SS DELTA',
                          'OCS SS ID',
                          'OCS PRESET PS ALPHA',
                          'OCS PRESET PS DELTA',
                          'OCS PS ID'])
    else:
        Hkeywords.extend(['OCS TARG2 ALPHA',
                          'OCS TARG2 DELTA',
                          'OCS TARG2 NAME',
                          'OCS TARG2 KMAG',
                          'OCS TARG2 HMAG',
                          'OCS TARG1 ALPHA',
                          'OCS TARG1 DELTA',
                          'OCS TARG1 NAME',
                          'OCS TARG1 KMAG',
                          'OCS TARG1 HMAG'])
    for k in Hkeywords:
        try:
            res[k] = f[0].header['HIERARCH ESO '+k]
        except:
            pass
    # -- start-end keywords: need to do an average
    SEkeywords = ['ISS AIRM', 'ISS AMBI FWHM', 'ISS AMBI TAU0']
    for k in SEkeywords:
        try:
            res[k] = strictlyPositiveMean(
                [f[0].header['HIERARCH ESO '+k+' START'],
                 f[0].header['HIERARCH ESO '+k+' END']])
        except:
            pass
    # -- FSUs calibrations:
    calibs = ['DARK', 'FLAT', 'PHAS', 'VISI', 'WAVE']
    fsus   = ['FSUA', 'FSUB']
    channels = ['W', '1', '2', '3', '4', '5']
    try:
        for fsu in fsus:
            for calib in calibs:
                res[fsu+'_'+calib] = np.zeros((6,4))
                for k, chan in enumerate(channels):
                    s = f[0].header['HIERARCH ESO OCS '+fsu+' K'+chan+calib]
                    res[fsu+'_'+calib][k,:] = np.float_(s.split(','))
        res['FSUA FLAT-DARK W']=(res['FSUA_FLAT']-res['FSUA_DARK'])[0,:].mean()
        res['FSUB FLAT-DARK W']=(res['FSUB_FLAT']-res['FSUB_DARK'])[0,:].mean()
    except:
        pass
    return res

