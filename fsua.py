"""
single feed data analysis
"""

import pyfits
import slidop
import numpy as np
from matplotlib import pyplot as plt

class calibFSU():
    """
    sky calibration of FSU, using DL scans
    """
    # frequency at which FSUs have pickup noise, in Hz:
    _pickupNoise = [144.6, 113.8, 96.4] # UT?
    
    def __init__(self,filename, useDarkPixel=True, nScansMax=None):
        self.f = pyfits.open(filename)
        self.useDarkPixel=useDarkPixel
        self.nScansMax=nScansMax
        assert self.f[0].header['ESO TPL ID']=='PACMAN_obs_Scanning', \
           'file is not of type "PACMAN_obs_Scanning"'
        #-- standard analysis sequence
        self._loadHeader()
        self._loadScans()
        self._computeTF()
        self._computeEffWl()
        #-- close file
        self.f.close()
        self.f=[]
        
    def _loadHeader(self):
        self.header={}
        for k in ['DEL REF NAME', 'DEL FT SENSOR',
                  'ISS CONF DL1', 'ISS CONF DL2',
                  'ISS PRI FSU1 FREQ']:
            self.header[k]=self.f[0].header['ESO '+k]
        # find tracking DL
        if self.header['DEL REF NAME']==self.header['ISS CONF DL1']:
            self.header['DEL TRK NAME'] = self.header['ISS CONF DL2']
        else:
            self.header['DEL TRK NAME'] = self.header['ISS CONF DL1']
        # load calibrations:
        calibs = ['DARK', 'FLAT', 'PHAS', 'VISI', 'WAVE']
        channels = ['W', '1', '2', '3', '4', '5']
        self._fsu_calib = {}
        for calib in calibs:
            self._fsu_calib[calib] = np.zeros((6,4))
            for k, chan in enumerate(channels):
                tmp = self.f[0].header['ESO OCS '+self.header['DEL FT SENSOR']+' K'+chan+calib]
                self._fsu_calib[calib][k,:] = np.float_(tmp.split(','))
        return 
        
    def _loadScans(self):
        #-- this table will contain the scans:
        self.scans =[]
        # -- reconstruct stepping phase if needed
        stepping_phase = self.f['OPDC'].data['STEPPING_PHASE']
        if stepping_phase.max()==0: 
            tmp = np.abs(np.gradient(self.f['OPDC'].data['STATE']))
            tmp[tmp>1] = 0 # remove 0->20,21 and 20,21->0
            tmp = np.cumsum(tmp)
            tmp[tmp==tmp.max()]==0 # last state, not scanning
            stepping_phase=tmp
        #-- what HDU do we need to consider:
        IMAGING_DATA = 'IMAGING_DATA_'+self.header['DEL FT SENSOR']        
        #-- how many scan do we load?
        if self.nScansMax==None:
            self.nScansMax=int(stepping_phase.max())
        for i in range(self.nScansMax):
            #-- determine when the i-th scan happened 
            w = np.where((stepping_phase==i+1)*
                    (self.f['OPDC'].data['STATE']>0))
            tmin = self.f['OPDC'].data['TIME'][w].min()
            tmax = self.f['OPDC'].data['TIME'][w].max()
            #-- slice data
            w = np.where((self.f[IMAGING_DATA].data['TIME']>tmin)*
                        (self.f[IMAGING_DATA].data['TIME']<=tmax))
            #-- tmp will contain the scan information
            tmp={'STEPPING PHASE':i+1} 
            #-- generic data from FSU
            for k in ['TIME', 'GDSNR']:
                tmp[k]= self.f[IMAGING_DATA].data[k][w]
            #-- A,B,C,D data:
            pixels = {'A':1, 'B':2, 'C':3, 'D':4}
            for k in pixels.keys():
                tmp[k]= self.f[IMAGING_DATA].data['DATA'+str(pixels[k])][w]
                if 'DATA5' in [c.name for c in self.f[IMAGING_DATA].data.columns] and\
                        self.useDarkPixel:
                    #-- simultaneous dark
                    tmp['DARK'+k] = self.f[IMAGING_DATA].data['DATA5'][w][:,pixels[k]-1]
                    tmp[k] -= tmp['DARK'+k][:,None]*np.array([5,1,1,1,1,1])[None,:]
                else:
                    #-- averaged dark
                    tmp[k] -= self._fsu_calib['DARK'][:,pixels[k]-1][None,:]
                #-- normalize to flat-dark
                tmp[k] /= (self._fsu_calib['FLAT'][:,pixels[k]-1]-
                           self._fsu_calib['DARK'][:,pixels[k]-1])[None,:]
            #-- interpolate OPDC values
            for k in ['RTOFFSET']:
                tmp[k]=np.interp(tmp['TIME'],
                                self.f['OPDC'].data['TIME'],
                                self.f['OPDC'].data[k])
            #-- append currend scan to the table of scans
            self.scans.append(tmp)
        return
 
    def _computeTF(self, wl_min_um=1.2, wl_max_um=10.0, N=200):
        """
        compute the fourier transform of the scans (A,B,C,D) and store the
        results using 'TF_...' keywords. self.TF_wl contains the corresponding
        wavelengths (in meters).
        """
        #-- wavelength table:
        self.TF_wl = 1/np.linspace(1/wl_max_um, 1/wl_min_um, N)*1e-6
        for i in range(len(self.scans)):
            #-- find center of the scan
            try:
                snr = slidop.slidingMean(self.scans[i]['TIME'],self.scans[i]['GDSNR'],
                                         20e6/self.header['ISS PRI FSU1 FREQ'])
            except:
                snr,rms,xsnr= sliding_avg_rms(self.scans[i]['TIME'],self.scans[i]['GDSNR'],
                                         20e6/self.header['ISS PRI FSU1 FREQ'])
                snr = np.interp(self.scans[i]['TIME'], xsnr,snr)

            opd0 = self.scans[i]['RTOFFSET'][snr.argmax()]
            #-- create cos and sin waves
            _cos = np.cos(2*np.pi*(self.scans[i]['RTOFFSET']-opd0)[None,:]/self.TF_wl[:,None])
            _sin = np.sin(2*np.pi*(self.scans[i]['RTOFFSET']-opd0)[None,:]/self.TF_wl[:,None])
            #-- apodization window (width in microns):
            apod_width = np.array([50,50,50,50,50,50])*1e-6/(np.sqrt(2)/2) 
            apod = np.exp(-((self.scans[i]['RTOFFSET']-opd0)[:,None]/
                    apod_width[None,:])**2) 
            #-- computation for each channel (A,B,C,D)
            for k in ['A', 'B', 'C', 'D']:
                self.scans[i]['TF_'+k]=(apod[None,:,:]*self.scans[i][k][None,:,:]*
                              (_cos+1j*_sin)[:,:,None]).mean(axis=1)
                if self.scans[i].has_key('DARK'+k):
                    self.scans[i]['TF_DARK'+k]=(self.scans[i]['DARK'+k][None,:]*
                                      (_cos+1j*_sin)).mean(axis=1)
        return

    def _computeEffWl(self):
        """
        compute eff wavelength of each pixel using the PSD
        """
        self.effWl={}
        for i,k in enumerate(['A', 'B', 'C', 'D']):
            #-- averaged PSD over all scans
            avgPSD = np.array([np.abs(s['TF_'+k])**2
                             for s in self.scans]).mean(axis=0)
            #-- first approx
            self.effWl[k]=self.TF_wl[avgPSD.argmax(axis=0)]
            #-- integrate around highest peak
            mask = avgPSD*(np.abs(self.TF_wl[:,None]-self.effWl[k][None,:])<0.4e-6)
            self.effWl[k]=(mask*self.TF_wl[:,None]).sum(axis=0)/mask.sum(axis=0)        
        return 
    
    def waterFall(self, channel=0):
        """
        channel is 0 for white light, 1..5 for the dispersed pixels
        """
        plt.figure(0)
        plt.clf()
        ax1= plt.subplot(121)
        ax2= plt.subplot(122, sharex=ax1, sharey=ax1)
        for k in range(len(self.scans)):
            tmp= self.scans[k]['A'][:,channel]-self.scans[k]['C'][:,channel]
            x = self.scans[k]['RTOFFSET']
            x -= x.mean()
            x*=1e6
            ax1.plot(x,tmp+k,
                 color='k', alpha=0.8)
            ax2.plot(x, slidop.slidingMean(self.scans[k]['TIME'],
                                    self.scans[k]['GDSNR'],
                                    20e6/self.header['ISS PRI FSU1 FREQ'])/10.+k,
                 color='k', alpha=0.8)
        ax1.set_ylim(-1,k+1)
        return
    
    def displayPSD(self):
        plt.figure(1)
        plt.clf()
        ax1={}
        for i,k in enumerate(['A', 'B', 'C', 'D']):
            if k!='A':
                ax1[k]=plt.subplot(4,1,i+1, sharex=ax1['A'], sharey=ax1['A'])
            else:
                ax1[k]=plt.subplot(4,1,i+1)
            ax1[k].plot(1/(self.TF_wl*1e6),
                       np.array([np.abs(s['TF_'+k])**2
                                 for s in self.scans]).mean(axis=0)[:,0],
                       'k', linewidth=3, alpha=0.5)
            ax1[k].plot(1/(self.TF_wl*1e6),
                       np.array([np.abs(s['TF_'+k])**2
                                 for s in self.scans]).mean(axis=0)[:,1:])
            ax1[k].set_ylabel(k)
        ax1['D'].set_xlabel('1/$\lambda$ (1/$\mu$m)')

        plt.figure(2)
        plt.clf()
        ax2={}
        for i,k in enumerate(['A', 'B', 'C', 'D']):
            if k!='A':
                ax2[k]=plt.subplot(4,1,i+1, sharex=ax2['A'], sharey=ax2['A'])
            else:
                ax2[k]=plt.subplot(4,1,i+1)
            ax2[k].plot(1/(self.TF_wl*1e6),
                       np.array([np.angle(s['TF_'+k]-s['TF_A'])
                                 for s in self.scans]).mean(axis=0)[:,1:])
            ax2[k].set_ylabel(k)
        ax2['D'].set_xlabel('1/$\lambda$ (1/$\mu$m)')
        return
        
def findFringes(filename):
    """
    Find fringes in the scanning file 'filename'. Plots the results and returns
    the value of the offset.
    """
    f = pyfits.open(filename)
    fringes = (f['IMAGING_DATA_FSUA'].data['DATA1']-f['IMAGING_DATA_FSUA'].data['DATA3'])[:,0]
    snr = f['IMAGING_DATA_FSUA'].data['PDSNR']
    opd = np.interp(f['IMAGING_DATA_FSUA'].data['TIME'],
                    f['OPDC'].data['TIME'], f['OPDC'].data['RTOFFSET'])
    s = opd.argsort()
    fringes = fringes[s]
    snr = snr[s]
    opd = opd[s]
    
    plt.figure(0)
    plt.clf()
    ax=plt.subplot(211)
    plt.plot(opd, fringes, label='A-C')
    plt.legend()
    plt.subplot(212, sharex=ax)
    rms = slidop.slidingStd(opd,fringes, 10e-6)
    rms/= rms.mean()
    plt.plot(opd, rms, 'r', label='A-C RMS')
    plt.plot(opd, snr, 'g', label='PDSNR')
    plt.legend()
    f.close()
    return opd[rms.argmax()]

def ftkPerfo(filename, minState=5):
    """
    estimate OPDC performance as OPD residuals
    """
    f = pyfits.open(filename)
    x = f['OPDC'].data.field('TIME')*1e-6
    y = f['OPDC'].data.field('UWPHASE')
    s = f['OPDC'].data.field('STATE')
    f.close()
     
    x = x[np.where(s>=minState)]
    y = y[np.where(s>=minState)]
    
    y = y[np.where(1-np.isnan(x))]
    x = x[np.where(1-np.isnan(x))]
    
    x = x[np.where(1-np.isnan(y))]
    y = y[np.where(1-np.isnan(y))]
    
    x = x[::3]
    y = y[::3]
   
    plt.figure(0)
    plt.clf()
    #plt.plot(x,y,'+', label='raw')
    all_dx = np.logspace(-1,0.5,10)
    jitt = []
    for dx in all_dx:
        try:
            yp = slidop.slidingStd(1.0*x,1.0*y,dx)*2.2/(2*np.pi)
        except:
            print np.diff(x)
            y_,yp,x_ = slidop.sliding_avg_rms(y*2.2/(2*np.pi),x,dx)
            print yp
            yp = np.interp(x,x_,yp)
        jitt.append(yp.mean())
    plt.plot(all_dx,jitt, 'ok-', label='residual jitter')
    plt.ylabel('opd residual RMS (um)')
    plt.xlabel('time window')
    plt.xscale('log')
    plt.vlines(21*0.040, plt.ylim()[0], plt.ylim()[1], label='MIDI frame')
    plt.legend()
    #plt.yscale('log')
    return
