from matplotlib import pyplot as plt
import numpy as np
import os
import astromNew
import scipy.signal
import pyfits
import dpfit

data_directory = '/Users/amerand/DATA/PRIMA/TT2/' # external BIG hardrive

def first156274(runOffCorrection=True):
    fi = os.listdir(data_directory+'/2012-05-17')
    fi = filter(lambda x: 'RED.fits' in x , fi)
    fi = ['2012-05-17/'+''.join(f.split('_RED')) for f in fi]
    fg = {'SEP':10.1, 'PA':260.0, 'M0':0.0, }
    doNotFit=[]
    a = astromNew.fitListOfFiles(fi, data_directory,
                             firstGuess=fg, verbose=2,
                             doNotFit=doNotFit,
                             maxResiduals=3)
    return a

def FSUB_mod(fromFiles=False):
    if fromFiles:
        dir_ = '/Users/amerand/DATA/PRIMA/TT2/RTDscope/'
        files = ['FSUB_X_0.5e-5.dat', 'FSUB_X_1e-5.dat',
                 'FSUB_X_2e-5.dat', 'FSUB_X_3e-5.dat',
                 'FSUB_X_5e-5.dat', 'FSUB_X_6e-5.dat',
                 'FSUB_X_7e-5.dat', 'FSUB_X_8e-5.dat',]
        offset = []
        flux = []
        amp = []
        
        channel = 'KWSUMA'
        freq = 50.0 # in Hz
        for f  in files:
            print f
            r = readRTDfile(os.path.join(dir_, f))
            offset.append(float(f.split('_')[2].split('d')[0][:-1])*1e6)
            flux.append( r['KWSUMA'].mean())
            amp.append(np.abs((r['KWSUMA']*
                               np.exp(2j*np.pi*r['TIME']*freq)).mean()))
        print offset
        offset = np.array(offset)
        flux = np.array(flux)
        amp = np.array(amp)
    else:
        freq = 50
        #-- IP2, visually
        offset = np.array([0,0.5,1,2,3,4,5,6,7,8])*10 # in mu rad
        # -- ADU:
        flux = 1000*np.array([40.3,40.1,38.5,32.5,24,15.5,8.5, 4.0, 1.8,1])
        amp = 1000*np.array([0, 0.4, 1, 1.5, 1.9, 1.7, 1.2, .75, .25,.15])/2.
        dark = 120
    
    print np.gradient(flux-dark)/np.gradient(offset)
    
    f0 = plt.figure(0)
    plt.clf()
    
    ax1 = f0.add_subplot(211)
    plt.title('FSUB IP2')
    ax1.plot(offset, flux-dark, label='flux', linewidth=2, color='r')
    plt.ylabel('ADU')
    ax1.legend()
    
    ax2 = f0.add_subplot(212, sharex=ax1)
    ax2.plot(offset, amp, label='observed modulation @ %4.2fHz'%(freq),
             linewidth=2, color='b')
    mod = np.median(amp/(np.gradient(flux-dark)/np.gradient(offset)))
    ax2.plot(offset, mod*np.gradient(flux-dark)/np.gradient(offset),
             linestyle='dotted',
             label=r'predicted %4.2f $\mu$rad modulation @ %4.2fHz'%(-mod, freq),
             linewidth=4, color='r')
    plt.ylabel('ADU') 
    plt.xlabel('TT offset in X ($\mu$rad)')
    ax2.legend(loc='lower center')
    return
    
def readRTDfile(filename):
    f = open(filename)
    lines = f.readlines()
    f.close()
    header = filter(lambda x: x[:2]=='##', lines)
    for h in header:
        if h.split()[1].split('=')[0][:6] == 'FIELDS':
            fields = ['' for k in range(int(h.split()[1].split('=')[1]))]
        elif h.split()[1].split('=')[0][:5] == 'FIELD':
            i = int(h.split()[1].split('=')[0][5:])
            fields[i-1]=h.split()[1].split('=')[1]
        elif h.split()[1].split('=')[0]=='PERIOD':
            period=float(h.split()[1].split('=')[1])
    #print fields
    
    lines=filter(lambda x: x[:2]!='##', lines)
    res = {}
    for k in range(len(fields)):
        try:
            res[fields[k]] = np.array([float(l.split()[k]) for l in lines])
        except:
            res[fields[k]] = [l.split()[k] for l in lines]
    res['TIME'] = np.arange(len(lines))*period
    return res

def testDDLscanning():
    global old, new # to avoid reloading all the time
    
    oldFile = '/Users/amerand/DATA/PRIMA/TT2/2012-05-17/PACMAN_OBJ_SCAN_139_0013.fits'
    newFile = '/Users/amerand/DATA/PRIMA/TT2/2012-05-18/PACMAN_OBJ_SCAN_139_0023.fits'
    
    try:
        print 'OLD:', old.filename()
        print 'NEW:', new.filename()
    except:
        old = pyfits.open(oldFile)
        new = pyfits.open(newFile)
    
    xn = new['DOPDC'].data.field('TIME')[1e4:-1e4]
    pspn = scipy.signal.detrend(new['DOPDC'].data.field('PSP')[1e4:-1e4])
    obsn = scipy.signal.detrend(new['DOPDC'].data.field('DDL1')[1e4:-1e4])
    
    
    xo = old['DOPDC'].data.field('TIME')[5e4:-1e4]
    pspo = scipy.signal.detrend(old['DOPDC'].data.field('PSP')[5.e4:-1e4])*1e-6
    obso = scipy.signal.detrend(old['DOPDC'].data.field('DDL1')[5e4:-1e4])
    
    
    plt.figure(0)
    plt.clf()
    ax1 = plt.subplot(311)
    
    dx = 0.00717e7
    
    # command
    plt.title('OLD:  '+oldFile.split('TT2/')[1]+
              '\nNEW:  '+newFile.split('TT2/')[1])
    plt.plot(xn-dx, pspn*1e3, 'r', label='PSP (new)',
             linewidth=2, linestyle='dashed')
    plt.plot(xo, pspo*1e3, 'b', label='PSP (old)',
             linewidth=2, linestyle='dashed')
    plt.hlines([-0.2, 0.2], 0, 1e8, color='y', alpha=0.6,
        linewidth=3,label='requested amplitude')
    
    
    # measured
    plt.plot(xn-dx, obsn*1e3, 'r', label='DDL1 (new)', linewidth=2, alpha=0.5)
    plt.plot(xo, obso*1e3, 'b', label='DDL1 (old)', linewidth=2, alpha=0.5)
    
    plt.legend()
    plt.ylabel('position (mm)')
    plt.grid()
    plt.legend(prop={'size':9})
    
    ax2 = plt.subplot(312, sharex=ax1)
    # measured
    plt.plot(xn-dx, (obsn-pspn)*1e6, 'r', label='DDL1 - PSP (new)',
             linewidth=2, alpha=1)
    plt.plot(xo, (obso-pspo)*1e6, 'b', label='DDL1 - PSP (old)',
             linewidth=2, alpha=1)
    plt.grid()
    plt.ylabel(r'$\delta$ ($\mu$m)')
    plt.legend(prop={'size':9})
    
    ax3 = plt.subplot(313, sharex=ax1)
    # command
    plt.plot(xn-dx, np.gradient(pspn)/np.gradient(xn), 'r', label='PSP',
             linewidth=2, linestyle='dashed')
    plt.plot(xo, np.gradient(pspo)/np.gradient(xo), 'b', label='PSP',
             linewidth=2, linestyle='dashed')
    # measured
    plt.plot(xn-dx, np.gradient(obsn)/np.gradient(xn), 'r', label='DDL1 (new)',
             linewidth=2, alpha=0.5)
    plt.plot(xo, np.gradient(obso)/np.gradient(xo), 'b', label='DDL1 (old)',
             linewidth=2, alpha=0.5)
    plt.xlabel(r'time ($\mu$s)')
    plt.ylabel('speed (m/$\mu$s)')
    # 3 samples per microns, at 1KHz
    plt.hlines([(1e-6/3.)/(1000), -(1e-6/3.)/(1000)], 0, 1e8, color='y',
        alpha=0.6, linewidth=3,label='requested speed')
    plt.grid()
    plt.xlim(3e7, 3.4e7)
    plt.legend(prop={'size':9})
    return

def EarthQuake():
    global eq
    try:
        print eq.filename()
    except:
        eq =pyfits.open('/Users/amerand/DATA/PRIMA/TT2/2012-05-19/PACMAN_OBJ_ASTRO_141_0010.fits')
    
    plt.figure(0)
    plt.clf()
    
    t = eq['METROLOGY_DATA_FSUB'].data.field('TIME')*1e-6 
    z = eq['METROLOGY_DATA_FSUB'].data.field('DELTAL')
    z = np.unwrap(z*2*np.pi/z.ptp())*0.5*z.ptp()/np.pi

    tc = eq['OPDC'].data.field('TIME')*1e-6 
    c = eq['OPDC'].data.field('FUOFFSET')

    w = np.where((t-78.7>-5)*(t-78.7<60))
    plt.plot(t[w], scipy.signal.detrend(z[w]-np.interp(t[w],tc,c))*1e6, '-k')
    plt.ylim(-30, 30)
    plt.xlim(75, 135)
    plt.xlabel('time since '+eq[0].header['HIERARCH ESO PCR ACQ START']+' (s)')
    plt.ylabel('metrology signal ($\mu$m)')

    return

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
    
def dispersion(Lair = 20., R=25):
    # simulate dispersion
    wl = np.linspace(1.8,2.4,100)*1e-6 
    x = (Lair+np.linspace(-100,100,1000)*1e-6)
    opd = x[np.newaxis,:]*n_air_P_T(wl*1e6)[:,np.newaxis]
    opd -= opd.mean()
    fringes = np.cos(np.pi*2/wl[:,np.newaxis]*opd)
    # PRIMA
    wl_p = np.linspace(1.8,2.3,6)*1e-6
    dwl_p = np.diff(wl_p).mean()
    fringes_p = np.zeros((len(wl_p), len(x)))
    print len(fringes_p[0])
    for k,l in enumerate(wl_p):
        fringes_p[k]=(fringes*(np.abs(wl-l)<dwl_p/2.)[:,np.newaxis]).mean(axis=0)

    # -- PRIMA fringes    
    fringes_p = np.array(fringes_p)
    x_, wl_ = np.meshgrid( (x-x.mean())*1e6, wl*1e6)
    x_p, wl_p = np.meshgrid( (x-x.mean())*1e6, wl_p*1e6)

    cmap = 'YlGn'
    #cmap = 'RdPu_r'

    plt.figure(0)
    plt.clf()
    ax1=plt.subplot(311)
    plt.title('$L_{air}$=%4.1fm'%(Lair))
    plt.pcolormesh(x_, wl_, fringes, cmap=cmap)
    plt.xlim(x_.min(), x_.max())
    plt.ylim(wl_.min(), wl_.max())
    plt.ylabel('wavelength ($\mu$m)')

    plt.subplot(312, sharex=ax1)
    plt.pcolormesh(x_p, wl_p, fringes_p, cmap=cmap)
    plt.xlim(x_p.min(), x_p.max())
    plt.ylim(wl_p.min(), wl_p.max())
    plt.ylabel('wavelength ($\mu$m)')
    
    plt.subplot(313, sharex=ax1)
    plt.plot((x-x.mean())*1e6, fringes.mean(axis=0), 'k', linewidth=2)
    plt.xlabel('OPD ($\mu$m)')
    plt.ylim(-1,1)
    plt.xlim(x_.min(), x_.max())
    return

def track2wl(offset_um=4.4, noise=False, Lair=0):
    wl = np.linspace(1.7,2.4,500)*1e-6 
    x = np.linspace(-100,100,2000)*1e-6
    opd = (Lair+x[:,None])*n_air_P_T(wl*1e6)[None,:]
    opd -= opd.mean()
    
    phi = np.array([0, np.pi/2., np.pi, 3*np.pi/2.])
    wl_p = np.linspace(1.9,2.3,6)*1e-6
    dwl_p = np.diff(wl_p).mean() 
    if noise:
        phi += np.random.rand(len(phi))*0.2
        wl_p += np.random.rand(len(wl_p))*0.3*dwl_p
   
    fringes = 1+np.cos(2*np.pi*opd[:,:,None]/wl[None,:,None]+
                       phi[None,None,:])
    
    fringes_p = np.zeros((len(x), len(wl_p), len(phi)))

    for k,l in enumerate(wl_p):
        fringes_p[:,k,:]=(fringes*(np.abs(wl-l)<dwl_p/2.)[None,:,None]).mean(axis=1)
    
    # fringes at 0  WL 
    c0wl = np.cos(2*np.pi*0.0e-6/wl_p[:,None]+
                       phi[None,:])
    s0wl = np.sin(2*np.pi*0.0e-6/wl_p[:,None]+
                       phi[None,:])
    
    snr0 = (fringes_p*c0wl[None,:,:]).mean(axis=1).mean(axis=1)**2+\
          (fringes_p*s0wl[None,:,:]).mean(axis=1).mean(axis=1)**2
    
    phi0 = np.arctan2( (fringes_p*c0wl[None,:,:]).mean(axis=1).mean(axis=1),
                       (fringes_p*s0wl[None,:,:]).mean(axis=1).mean(axis=1))
    
    # fringes at 2 avg WL = 4.4um 
    c2wl = np.cos(2*np.pi*offset_um*1e-6/wl_p[:,None]+
                       phi[None,:])
    s2wl = np.sin(2*np.pi*offset_um*1e-6/wl_p[:,None]+
                       phi[None,:])
      
    snr2 = (fringes_p*c2wl[None,:,:]).mean(axis=1).mean(axis=1)**2+\
          (fringes_p*s2wl[None,:,:]).mean(axis=1).mean(axis=1)**2
    phi2 = np.arctan2( (fringes_p*c2wl[None,:,:]).mean(axis=1).mean(axis=1),
                       (fringes_p*s2wl[None,:,:]).mean(axis=1).mean(axis=1))
    
    
    plt.figure(0)
    plt.clf()
    ax1=plt.subplot(311)
    plt.plot(x*1e6, fringes_p[:,:,0], alpha=0.5)
    plt.plot(x*1e6, fringes_p[:,:,0].mean(axis=1),
             linewidth=2)
    plt.subplot(312, sharex=ax1)
    plt.plot(x*1e6,snr2, label='SNR optimized at %3.1f $\mu$m'%offset_um)
    plt.plot(x*1e6,snr0, label='usual SNR')
    plt.legend(prop={'size':9})
    plt.subplot(313, sharex=ax1)
    
    wt = np.where(np.abs(x-x[snr2.argmax()])<10.0e-6)
    a = np.polyfit(x[wt]*1e6,
            np.unwrap(phi0[wt]), 1)
    
    #plt.plot(x[snr2>0.1*snr2.max()]*1e6,
    #         np.unwrap(phi0[snr2>0.1*snr2.max()]),
    #         label='phase (C0, S0)' )
    #plt.plot(x[snr2>0.1*snr2.max()]*1e6,
    #         np.polyval(a,x[snr2>0.1*snr2.max()]*1e6),
    #         label='linear fit', linewidth=2, color='y' )
    plt.plot(x[snr2>0.1*snr2.max()]*1e6,
             np.unwrap(np.unwrap(phi0[wt]-
             np.polyval(a,x[wt]*1e6))),
             label='phase residuals', linewidth=2, color='y' )

    plt.legend(prop={'size':9})

    plt.xlabel('OPD ($\mu$m)')
    
    
    Y_, X_ = np.meshgrid(phi, wl_p*1e6)
    
    plt.figure(1)
    plt.clf()
    plt.subplot(221)
    plt.title('C0')
    plt.imshow(c0wl, cmap='RdBu', vmin=-1,vmax=1,
               interpolation='nearest')

    plt.subplot(223)
    plt.title('S0')
    plt.imshow(s0wl, cmap='RdBu', vmin=-1,vmax=1,
               interpolation='nearest')

    plt.subplot(222)
    plt.title('C%3.1f$\mu$m'%offset_um)
    plt.imshow(c2wl, cmap='RdBu', vmin=-1,vmax=1,
               interpolation='nearest')

    plt.subplot(224)
    plt.title('S%3.1f$\mu$m'%offset_um)
    plt.imshow(s2wl, cmap='RdBu', vmin=-1,vmax=1,
               interpolation='nearest')
    return
    
def timeJumps():
    files = ['/Users/amerand/DATA/PRIMA/TT2/2012-05-21/PACMAN_OBJ_SCAN_143_0001.fits',
    '/Users/amerand/DATA/PRIMA/TT2/2012-05-23/PACMAN_OBJ_SCAN_145_0001.fits',
    '/Users/amerand/DATA/PRIMA/TT2/2012-05-23/PACMAN_OBS_GENERIC145_0001.fits',
    # this file has a half the RMNREC rate:
    '/Users/amerand/DATA/PRIMA/TT2/2012-05-23/PACMAN_OBJ_SCAN_145_0003.fits',
    ]
    plt.figure(0)
    plt.clf()
    
    for i,f in enumerate(files):
        a = pyfits.open(f) 
        plt.subplot(len(files)*100+11+i)
        plt.plot(a['OPDC'].data.field('TIME')[1:],
                 np.diff(a['OPDC'].data.field('TIME')))
        plt.xlim(32000000,34000000)
        plt.title(a.filename().split('TT2/')[1])
        plt.ylim(492, 508)
        plt.ylabel(r'$\Delta$TIME (OPDC)')    
        a = []
    return