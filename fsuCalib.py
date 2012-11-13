import pyfits
import numpy as np
import os
from matplotlib import pyplot as plt
import dpfit

def testCalibrate(plot=False):
    #directory='/Users/amerand/DATA/PRIMA/TT2/2012-05-17/'
    #dark='PACMAN_LAB_DARK_138_0001.fits'
    #flat='PACMAN_LAB_FLAT_138_0002.fits'
    #resp='PACMAN_LAB_FSURESPONSE_138_0002.fits'
    directory='/home/amerand/DATA/2012-11-11/'
    dark='PACMAN_LAB_DARK_316_0001.fits'
    flat='PACMAN_LAB_FLAT_316_0002.fits'
    resp='PACMAN_LAB_FSURESPONSE_316_0001.fits'
    
    t = calib(os.path.join(directory,dark),
                 os.path.join(directory,flat),
                 os.path.join(directory,resp), plot=plot)
    return t

class calib():
    def __init__(self, dark=None, flat=None, response=None, plot=False):
        self.calib = {'FSUA':{}, 'FSUB':{}}
        self.__plot=plot
        #-- load files:
        self.darkFile=dark
        self._loadDark()
        self.flatFile=flat
        self._loadFlat()  
        self.responseFile=response
        self._loadResponse()
        #-- define the wavelength table for Fourier analysis:
        self.TF_wl = np.linspace(1.8, 2.6, 200)
        #-- defaut analysis sequence: 
        self._computeTF('FSUA')
        self._computeTF('FSUB')
        self.effWl={}
        self._computeFourierEffWl('FSUA')
        self._computeFourierEffWl('FSUB')
        self.phases={}
        self._computeFourierPhases('FSUA')
        self._computeFourierPhases('FSUB')
        self.phasesFit={}
        #print self.effWl
        #print self.phases
        
    def _loadDark(self,):
        raw = pyfits.open(self.darkFile)
        if not raw[0].header['HIERARCH ESO TPL ID']=='PACMAN_cal_Lab_Dark':
            print 'Not a PACMAN_cal_Lab_Dark file!!!'
            raw.close()
            return False
        for fsu in ['FSUA', 'FSUB']:
            tmp={}
            for data in ['DATA'+str(k) for k in [1,2,3,4]]:
                tmp[data] = raw['IMAGING_DATA_'+fsu].data[data].mean(axis=0)
            self.calib[fsu]['DARK']=tmp
        raw.close()
        return True
    
    def _loadFlat(self,):
        raw = pyfits.open(self.flatFile)
        if raw[0].header['HIERARCH ESO TPL ID']!='PACMAN_cal_Lab_Flat':
            print 'Not a PACMAN_cal_Lab_Flat file!!!'
            raw.close()
            return False        
        for fsu in ['FSUA', 'FSUB']:
            tmp={}
            for data in ['DATA'+str(k) for k in [1,2,3,4]]:
                tmp[data] = raw['IMAGING_DATA_'+fsu].data[data].mean(axis=0)
            self.calib[fsu]['FLAT']=tmp
        raw.close()
        return
    
    def _loadResponse(self,):
        raw = pyfits.open(self.responseFile)
        if not raw[0].header['HIERARCH ESO TPL ID']=='PACMAN_cal_Lab_FSUResponse':
            print 'NOT a PACMAN_cal_Lab_FSUResponse file... nothing to be done'
            raw.close()
            return False
        # determine the number of scans:
        Nscans = int(raw['IMAGING_DATA_FSUA'].data.field('STEPPING_PHASE').max())
        self.labScans = []
        for n in range(Nscans)[1:]:
            s = {}
            for f in ['A', 'B']:
                w = np.where(raw['IMAGING_DATA_FSU'+f].data.field('STEPPING_PHASE')==n)
                tmp = {}
                tmp['TIME']=raw['IMAGING_DATA_FSU'+f].data.field('TIME')[w]
                if f=='A':
                    tmp['DELTAL']=np.interp(raw['IMAGING_DATA_FSU'+f].data.field('TIME')[w],
                                            raw['METROLOGY_DATA'].data.field('TIME'),
                                            raw['METROLOGY_DATA'].data.field('DELTAL'))-\
                                  np.interp(raw['IMAGING_DATA_FSU'+f].data.field('TIME')[w],
                                            raw['METROLOGY_DATA_FSUB'].data.field('TIME'),
                                            raw['METROLOGY_DATA_FSUB'].data.field('DELTAL'))
                else:
                    tmp['DELTAL']=-np.interp(raw['IMAGING_DATA_FSU'+f].data.field('TIME')[w],
                                            raw['METROLOGY_DATA_FSUB'].data.field('TIME'),
                                            raw['METROLOGY_DATA_FSUB'].data.field('DELTAL'))
                tmp['A'] = raw['IMAGING_DATA_FSU'+f].data.field('DATA1')[w[0],:]
                tmp['A'] -= self.calib['FSU'+f]['DARK']['DATA1'][None,:]
                tmp['A'] /= (self.calib['FSU'+f]['FLAT']['DATA1']-
                             self.calib['FSU'+f]['DARK']['DATA1'])[None,:]

                tmp['B'] = raw['IMAGING_DATA_FSU'+f].data.field('DATA2')[w[0],:]
                tmp['B'] -= self.calib['FSU'+f]['DARK']['DATA2'][None,:]
                tmp['B'] /= (self.calib['FSU'+f]['FLAT']['DATA2']-
                             self.calib['FSU'+f]['DARK']['DATA2'])[None,:]

                tmp['C'] = raw['IMAGING_DATA_FSU'+f].data.field('DATA3')[w[0],:]
                tmp['C'] -= self.calib['FSU'+f]['DARK']['DATA3'][None,:]
                tmp['C'] /= (self.calib['FSU'+f]['FLAT']['DATA3']-
                             self.calib['FSU'+f]['DARK']['DATA3'])[None,:]

                tmp['D'] = raw['IMAGING_DATA_FSU'+f].data.field('DATA4')[w[0],:]
                tmp['D'] -= self.calib['FSU'+f]['DARK']['DATA4'][None,:]
                tmp['D'] /= (self.calib['FSU'+f]['FLAT']['DATA4']-
                             self.calib['FSU'+f]['DARK']['DATA4'])[None,:]

                tmp['GD']=  raw['IMAGING_DATA_FSU'+f].data.field('GD')[w]
                tmp['GDSNR']=  raw['IMAGING_DATA_FSU'+f].data.field('GDSNR')[w]
                tmp['PD']=  raw['IMAGING_DATA_FSU'+f].data.field('PD')[w]
                tmp['PDSNR']=  raw['IMAGING_DATA_FSU'+f].data.field('PDSNR')[w]
                s['FSU'+f]=tmp
        
            self.labScans.append(s)
        raw.close()
        return True
    
    def _computeTF(self,FSU='FSUA'):
        for s in self.labScans:
            w = np.where(np.abs(s[FSU]['DELTAL'])<70e-6) # avoid sides of the scans.
            _sin = np.sin(2e6*np.pi*(s[FSU]['DELTAL'][w][None,:])/self.TF_wl[:,None])    
            _cos = np.cos(2e6*np.pi*(s[FSU]['DELTAL'][w][None,:])/self.TF_wl[:,None])
            for k in ['A', 'B', 'C', 'D']:
                s[FSU]['TF_'+k] = (s[FSU][k][w[0],:][None,:,:]*
                                   (_cos + 1.0j*_sin)[:,:,None]).mean(axis=1)
        return
    
    def _computeFourierEffWl(self,FSU='FSUA'):
        tmp={}
        if self.__plot:
            plt.figure(0+int(FSU=='FSUB'))
            plt.clf()
            ax={}
        for i,k in enumerate(['A', 'B', 'C', 'D']):
            #-- averaged PSD over all scans
            avgPSD = np.array([np.abs(s[FSU]['TF_'+k])**2
                             for s in self.labScans]).mean(axis=0)
            #-- first approx
            tmp[k]=self.TF_wl[avgPSD.argmax(axis=0)]
            #-- integrate around highest peak
            mask = avgPSD*(np.abs(self.TF_wl[:,None]-tmp[k][None,:])<0.4)
            tmp[k]=(mask*self.TF_wl[:,None]).sum(axis=0)/mask.sum(axis=0)        
            if self.__plot:
                if k=='A':
                    ax[k]=plt.subplot(4,1,i+1)
                else:
                    ax[k]=plt.subplot(4,1,i+1,
                                      sharex=ax['A'], sharey=ax['A'])
                
                ax[k].plot(self.TF_wl, avgPSD)
                ax[k].vlines(tmp[k][1:], 1e-6, avgPSD.max(), color='k')
                ax[k].vlines(tmp[k][0], 1e-6, avgPSD.max(),
                             color='k', linestyle='dashed')
                ax[k].set_yscale('log')
                ax[k].set_ylabel(k)
                if k=='A':
                    ax[k].set_title(self.responseFile+'\n'+FSU)
                if k=='D':
                    ax[k].set_xlabel('wavelength ($\mu$m)')
        self.effWl[FSU] = tmp
        return
    
    def _computeFourierPhases(self,FSU='FSUA'):       
        tmp={}
        if self.__plot:
            plt.figure(2+int(FSU=='FSUB'))
            plt.clf()
            plt.title(self.responseFile+'\n'+FSU)
            plt.xlabel('wavelength ($\mu$m)')
            plt.ylabel('phase / A (rad)')
    
        for i,k in enumerate(['B', 'C', 'D']):
            #-- averaged PSD over all scans
            avgPSD = np.array([np.abs(s[FSU]['TF_'+k])**2
                             for s in self.labScans]).mean(axis=0)
            avgAng = np.array([(np.angle(s[FSU]['TF_'+k])-np.angle(s[FSU]['TF_A']))%(2*np.pi)
                             for s in self.labScans]).mean(axis=0)
            #-- integrate around highest peak
            mask = (np.abs(self.TF_wl[:,None]-self.effWl[FSU][k][None,:])<0.04)
            mask[:,0] =(np.abs(self.TF_wl-self.effWl[FSU][k][0])<0.12)
            tmp['PHI'+k]=(mask*avgAng).sum(axis=0)/mask.sum(axis=0)
            tmp['PHI'+k] -= (i+1)*np.pi/2.0
            if self.__plot:
                plt.plot(self.TF_wl, mask[:,1:]*avgAng[:,1:], '.k')
                plt.plot(self.TF_wl, mask[:,0]*avgAng[:,0], '.r')
       
        self.phases[FSU] = tmp
        self.phases[FSU]['alpha']=np.sin(self.phases[FSU]['PHIC'])
        self.phases[FSU]['beta']=1+np.cos(self.phases[FSU]['PHIC'])
        self.phases[FSU]['gamma']=np.cos(self.phases[FSU]['PHIB'])+\
                    np.cos(self.phases[FSU]['PHID'])
        self.phases[FSU]['delta']=-np.sin(self.phases[FSU]['PHIB'])-\
                    np.sin(self.phases[FSU]['PHID'])        
        return
    
    def _fitPhases(self, scan=1, FSU='FSUA'):
        self._fittedScan=scan
        self._fittedFSU=FSU
        
        i0 = self.labScans[scan][FSU]['GDSNR'].argmax()
        
        self._fitiMin = i0-100
        self._fitiMax = i0+100
        
        channels = [1,2,3,4,5]
        params={}
        Y, doNotFit, fitOnly =[], [], []
        for c in channels:
            tmp={#'0OPD'+str(c):self.labScans[scan][FSU]['DELTAL'][self._fitiMin],
                 'WL'+str(c):self.effWl[FSU]['A'][c]*1e-6,
                #'phiB'+str(c):self.phases[FSU]['PHIB'][c],
                #'phiC'+str(c):self.phases[FSU]['PHIC'][c],
                #'phiD'+str(c):self.phases[FSU]['PHID'][c]
                 }   
            for k in tmp.keys():
                params[k]=tmp[k]
            Y.append(self.labScans[scan][FSU]['DELTAL'][self._fitiMin:self._fitiMax])

        params['phiB_2.2']=0.0
        params['phiC_2.2']=0.0
        params['phiD_2.2']=0.0
        params['0OPD_2.2']=0.0
        params['0OPD_c']=0.0
        Y = np.array(Y)
        for k in params.keys():
            if '0OPD' in k or 'WL' in k:
                fitOnly.append(k)
                
        fit0 = dpfit.leastsqFit(self.__fitPhasesFunc, channels,
                               params,Y, doNotFit=doNotFit,
                               fitOnly=fitOnly)
        model0 = fit0['model']
        doNotFit, fitOnly = [], None
        fit = dpfit.leastsqFit(self.__fitPhasesFunc, channels,
                               fit0['best'],Y, doNotFit=doNotFit,
                               fitOnly=fitOnly, verbose=1)
        model = fit['model']
        
        plt.figure(10)
        plt.clf()
        ax1=plt.subplot(211)
        for k in range(len(Y)):
            plt.plot(Y[k]*1e6, Y[k]*1e6-model[k]*1e6, 'r', alpha=0.5)
            plt.plot(Y[k]*1e6, Y[k]*1e6-model0[k]*1e6, 'k', alpha=0.5)           

        plt.subplot(313, sharex=ax1)
        plt.plot(Y[0]*1e6,
                 self.labScans[scan][FSU]['A'][self._fitiMin:self._fitiMax,0]-
                self.labScans[scan][FSU]['C'][self._fitiMin:self._fitiMax,0])
        plt.plot(Y[0]*1e6,
                 self.labScans[scan][FSU]['B'][self._fitiMin:self._fitiMax,0]-
                self.labScans[scan][FSU]['D'][self._fitiMin:self._fitiMax,0])
        for k in fit['best'].keys():
            if 'phi' in k:
                fit['best'][k] = (fit['best'][k]+np.pi)%(2*np.pi)-np.pi
        
        #plt.figure(11)
        #plt.clf()
        #for c in channels:
        #    plt.plot(fit['best']['WL'+str(c)]*1e6, fit['best']['phiB'+str(c)], 'or')
        #    plt.plot(fit['best']['WL'+str(c)]*1e6, fit['best']['phiC'+str(c)], 'og')
        #    plt.plot(fit['best']['WL'+str(c)]*1e6, fit['best']['phiD'+str(c)], 'oy')
        return 
        
    def __fitPhasesFunc(self, channels, params):
        """
        channels should be a subset of [0,1,2,3,4,5]
        params should be {'WL0':, ...,
                          'phiB0':, ...
                          'phiC0':, ...
                          'phiD0':, ...}      
        """
        res = []
        for c in channels:
            X = self.labScans[self._fittedScan][self._fittedFSU]['A'][self._fitiMin:self._fitiMax,c] -\
                self.labScans[self._fittedScan][self._fittedFSU]['C'][self._fitiMin:self._fitiMax,c]
            Y = self.labScans[self._fittedScan][self._fittedFSU]['B'][self._fitiMin:self._fitiMax,c] -\
                self.labScans[self._fittedScan][self._fittedFSU]['D'][self._fitiMin:self._fitiMax,c]
            X -= X.mean() # CRITICAL!!!
            Y -= Y.mean() # CRITICAL!!!
            phi={'B':0.0, 'C':0.0, 'D':0.0}
            for k in phi.keys():
                if params.has_key('phi'+k+str(c)):
                    phi[k]=params['phi'+k+str(c)]
                if params.has_key('phi'+k+'_2.2'):
                    phi[k]=params['phi'+k+'_2.2']
                if params.has_key('phi'+k+'_s'):
                    phi[k]+=params['phi'+k+'_s']*(params['WL'+str(c)]*1e6-2.2)
                
            alpha = np.sin(phi['C'])
            beta = 1+np.cos(phi['C'])
            gamma = np.cos(phi['B']) +np.cos(phi['D'])
            delta = -np.sin(phi['B']) -np.sin(phi['D'])
            Xp = X*gamma-Y*alpha
            Yp = Y*beta- X*delta
            #-- wrapped phase delay:
            PD = np.arctan2(Yp,Xp)
            PD = np.unwrap(PD)
            PD *= params['WL'+str(c)]/(2*np.pi)

            if params.has_key('0OPD_2.2'):   
                PD += params['0OPD_2.2']
            if params.has_key('0OPD_s'):
                phi[k]+=params['0OPD_s']*(params['WL'+str(c)]*1e6-2.2)
            if params.has_key('0OPD'+str(c)):
                PD += params['0OPD'+str(c)]
            
            res.append(PD)
        return res

    def recomputePD(self,scan=1, FSU='FSUA', method='Fourier'):
        """
        """
        if method=='Fourier':
            phases = self.phases[FSU]
            
        X = self.labScans[scan][FSU]['A']-self.labScans[scan][FSU]['C']
        Y = self.labScans[scan][FSU]['B']-self.labScans[scan][FSU]['D']
        Xp = X*phases['gamma'][None,:]-Y*phases['alpha'][None,:]
        Yp = Y*phases['beta'][None,:]-X*phases['delta'][None,:]
        PD = np.arctan2(Y,X)
        plt.figure(4)
        plt.clf()
        for k in range(PD.shape[1]):
            plt.plot(self.labScans[scan][FSU]['DELTAL'], np.unwrap(PD[:,k]),
                     label=method+' (%s)'%('W' if k==0 else str(k)))
        plt.plot(self.labScans[scan][FSU]['DELTAL'],
                 np.unwrap(self.labScans[scan][FSU]['PD']),
                 'k', alpha=0.5, linewidth=4, label='from file')
        plt.legend(loc='upper left')
        return        
        
