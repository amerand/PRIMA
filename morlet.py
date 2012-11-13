import numpy
import time
from scipy import weave

from matplotlib import pylab


def test(plot=True):
    x = numpy.linspace(10,-10,512)
    sig = numpy.sin(2*numpy.pi*x*(1+numpy.exp(-(x-3)**2/10)))
    sighat = numpy.fft.fft(sig)

    dx = numpy.abs(numpy.diff(x).mean())
    o = numpy.fft.fftfreq(len(sig),dx)
        
    T = CWT(sig, dx, scale_min=0.1, scale_max=5,\
            sigma=3, scale='log', n_per_octave=8)    
    if plot:
        T.plot(x)
    return

def perfo():
    tau0 = time.time()
    for i in range(10):
        test(plot=False)
    print time.time()-tau0
    return
    
def Csigma(sigma):
    return (1+numpy.exp(-sigma**2)-2*numpy.exp(-0.75*sigma**2))
    
def Ksigma(sigma):
    return numpy.exp(-0.5*sigma**2)

def phi(sigma, t):
    """
    morlet wavelet, in direct space
    """
    return Csigma(sigma)*numpy.pi**(-0.25)*numpy.exp(-0.5*t**2)*\
           (numpy.exp(1j*sigma*t) - Ksigma(sigma))

def phihat(sigma, omega):
    """
    complex FFT of the wavelet
    """
    return Csigma(sigma)*numpy.pi**(-0.25)*\
           (numpy.exp(-0.5*(sigma-omega)**2)-\
            Ksigma(sigma)*numpy.exp(-0.5*omega**2))

        
def central_freq(sigma):
    x = sigma-1 # first guess
    dx = 0.001
    y = fcf(sigma,x)
    x = x+dx
    while abs(fcf(sigma,x))>1e-10:
        if y*fcf(sigma,x) < 0:
            dx = -dx/2.
        y = fcf(sigma,x)
        x = x+dx
    return x
            
def fcf(sigma,omega):
    return (omega-sigma)**2-(omega**2-1)*numpy.exp(-sigma*omega)

class CWT:
    def __init__(self, signal, dt, scale_min=None, scale_max=None, sigma=8.0,
                 n_per_octave=16, scale='linear'):
        """
        performs the direct Morlet transform
        """
        # scales
        if scale_min == None:
            scale_min = 2*dt
        if scale_max == None:
            scale_max = dt*len(signal)/2.
        n = int(n_per_octave*scale_max/scale_min/2.)
        if scale=='linear':
            scales = numpy.linspace(scale_min, scale_max, n)
        elif scale=='log':
            scales = numpy.logspace(numpy.log10(scale_min),\
                                    numpy.log10(scale_max), n)
        o = numpy.fft.fftfreq(len(signal),dt)

        # FFT of the wavelets:
        f = sigma*o[numpy.newaxis,:]/scales[:,numpy.newaxis]
        what = phihat(sigma,f)
                
        # FFT of data:
        signalhat = numpy.fft.fft(signal)  

        # transform
        cwt = numpy.fft.ifft(what*signalhat[numpy.newaxis,:], axis=1)        
        
        self.scale_min   = scale_min
        self.scale_max   = scale_max    
        self.frequencies = scales
        self.n      = n
        self.scale  = scale
        self.cwt    = cwt
        self.omega  = o
        self.signal = signal
        self.dt     = dt
        return
    
    def plot(self, T=None, figure=1):
        """
        """
        if T==None:
            T = numpy.array(range(len(self.signal)))*self.dt
        
        Xp, Yp = numpy.meshgrid(T, self.frequencies)

        pylab.figure(figure)
        pylab.clf()
        pylab.subplot(211)
        pylab.plot(T, self.signal, 'k', linewidth=2)

        pylab.ylabel('signal')

        pylab.subplot(212)
        pylab.pcolormesh(Xp, Yp, numpy.abs(self.cwt)**2)
        pylab.xlabel('time')
        pylab.ylabel('frequency')
        #pylab.yscale('log')
        
        #pylab.subplot(313)
        #pylab.pcolormesh(Xp, Yp, numpy.angle(self.cwt))
        #pylab.xlabel('time')
        #pylab.ylabel('frequency')
        #pylab.yscale('log')
        return
