"""
sliding operators
"""

import numpy as np
from multiprocessing import Pool
from scipy import weave
from scipy.weave import converters

def slidingALL_avg_rms(x,t,delta_t, progression=False):
    """
    t = [0,1,2,3,4,5,6,7,8,9,0]
         |--o--| (delta_t)
    x = [0,1,2,3,4,5,6,7,8,9,0]
         |--o--| avg, rms
           |--o--|
             |--o--|
    returns x_avg, x_rms, t_avg with len(x)-n_delta_t+1          
    """
    dt = (t[1:]-t[:-1]).mean()   # assumes regular padding in t
    di = int(delta_t/dt)         # numberof of steps
    n = len(x)-di+1
    x_rms = np.zeros(n)
    x_avg = np.zeros(n)
    t_avg = np.zeros(n)
    for i in range(di):
        if i%1000==0 and progression:
            print i, di
        x_avg += x[i:i+n]
        t_avg += t[i:i+n]
    x_avg /= float(di)
    t_avg /= float(di)
    for i in range(di):
        if i%1000==0 and progression:
            print i, di
        x_rms += (x[i:i+n]-x_avg)**2
    x_rms /= float(di)
    x_rms = np.sqrt(x_rms)
    return x_avg, x_rms, t_avg

def tmp_sA_a_r_f_m(delta):
    global tmp_x1, tmp_t1
    return slidingALL_avg_rms_fast(tmp_x1,tmp_t1,delta_t)

def slidingALL_avg_rms_fast_multi(x,t,delta_t):
    """
    version multithreaded of slidingALL_avg_rms_fast for
    delta_t being an array
    """
    global tmp_x1, tmp_t1
    tmp_x1 = x
    tmp_t1 = t
    p = Pool()
    result = pool.apply_async(tmp_sA_a_r_f_m, delta_t)
    return resut

def slidingALL_avg_rms_fast(x,t,delta_t, progression=False):
    """
    t = [0,1,2,3,4,5,6,7,8,9,0]
         |--o--| (delta_t)
           |--o--|
             |--o--|
    x = [0,1,2,3,4,5,6,7,8,9,0]
         |--o--| avg, rms
           |--o--|
             |--o--|
    returns x_avg, x_rms, t_avg with len(x)-n_delta_t+1          
    """
    dt = (t[1:]-t[:-1]).mean()   # assumes regular padding in t
    di = int(delta_t/dt)         # numberof of steps
    n = len(x)-di+1
    x_rms = np.zeros(n)
    x_avg = np.zeros(n)
    t_avg = np.zeros(n)
    tp = t - t.min()
    x_avg[0] = x[1:di].sum()
    t_avg[0] = tp[1:di].sum()
    for i in range(n)[1:]:
        x_avg[i] = x_avg[i-1] + x[i+di-1] - x[i-1]
        t_avg[i] = t_avg[i-1] + tp[i+di-1] - tp[i-1]
    x_avg /= float(di-1) # ??? why is that???
    t_avg /= float(di)
    t_avg = t_avg + t.min()
    for i in range(di):
        x_rms += (x[i:i+n]-x_avg)**2
    x_rms /= float(di)
    x_rms = np.sqrt(x_rms)
    return x_avg, x_rms, t_avg

def sliding_avg_rms(x,t,delta_t, progression=False, phaser=False):
    """
    t = [0,1,2,3,4,5,6,7,8,9,0,1]
         |--o--| (delta_t)
    x = [0,1,2,3,4,5,6,7,8,9,0,1]
         |--o--| avg, rms
                 |--o--|
                         |--o--|
    returns x_avg, x_rms, t_avg with len(x)/n_delta_t          
    """
    dt = np.diff(t).mean()  # assumes regular padding in t
    print dt, delta_t
    di = int(delta_t/dt)        # numberof of steps
    n = len(x)/di
    x_rms = np.zeros(n)
    x_avg = np.zeros(n)
    t_avg = np.zeros(n)
    for i in range(n):
        if i%1000==0 and progression:
            print i, di
        if phaser:
            tmp =np.exp(1j*x[i*di:(i+1)*di])
            tmp_mean = tmp.mean()
            x_avg[i] = np.arctan2(tmp_mean.imag, tmp_mean.real)
            tmp *= np.conjugate(tmp_mean)
            x_rms[i] = np.arctan2(tmp.imag, tmp.real).std()
        else:
            x_avg[i] = x[i*di:(i+1)*di].mean()
            x_rms[i] = x[i*di:(i+1)*di].std()
        t_avg[i] = t[i*di:(i+1)*di].mean() 
    return x_avg, x_rms, t_avg

def slidingALL_min_max(x,t,delta_t, computeTime=False, progression=False):
    """
    t = [0,1,2,3,4,5,6,7,8,9,0]
         |--o--| (delta_t)
    x = [0,1,2,3,4,5,6,7,8,9,0]
         |--o--| min, max
           |--o--|
             |--o--|
    returns x_min, x_max, t_avg with len(x)-n_delta_t+1          
    """
    dt = (t[1:]-t[:-1]).mean()    # assumes regular padding in t
    di = int(delta_t/dt)          # numberof of steps
    n = len(x)-di+1
    if computeTime:
        t_avg = np.zeros(n)
    x_max = np.zeros(n)+x.min()
    x_min = np.zeros(n)+x.max()
    for i in range(di):
        if i%1000==0 and progression:
            print i, di
        if computeTime:
            t_avg += t[i:i+n]
        x_min = np.minimum(x_min,x[i:i+n])
        x_max = np.minimum(x_max,x[i:i+n])
    if computeTime:
        t_avg /= float(di)
        return x_min, x_max, t_avg
    else:
        return x_min, x_max

def sliding_min_max(x,t,delta_t, computeTime=True):
    """
    t = [0,1,2,3,4,5,6,7,8,9,0,1]
         |--o--| (delta_t)
    x = [0,1,2,3,4,5,6,7,8,9,0,1]
         |--o--| min, max
                 |--o--|
                         |--o--|
    returns x_min, x_max, t_avg with len(x)-n_delta_t+1          
    """
    dt = (t[1:]-t[:-1]).mean()  # assumes regular padding in t
    di = int(delta_t/dt)        # numberof of steps
    n = len(x)/di
    if computeTime:
        t_avg = np.zeros(n)
    x_max = np.zeros(n)
    x_min = np.zeros(n)
    for i in range(n):
        if computeTime:
            t_avg[i] += t[i*di:(i+1)*di].mean()            
        x_min[i] = x[i*di:(i+1)*di].min()
        x_max[i] = x[i*di:(i+1)*di].max()
    if computeTime:
        return x_min, x_max, t_avg
    else:
        return x_min, x_max

# the following are highly optimized, using weave
# thay all assume x is sorted!

def slidingMin(x,y,dx):
    x = np.float_(x)
    y = np.float_(y)
    LX = len(x)
    ymin = np.ones(LX)*y.max()
    code=\
    """
    int j;
    int i;
    int j0;
    int inloop;
    j0=1;
    
    for (i=0; i<LX; i++){
        j=j0;
        inloop=0;
        while ((x(j)<=x(i)+dx/2) && (j<LX) ) {
            if ((x(j)>=x(i)-dx/2) && (x(j)<=x(i)+dx/2)) {
                if (y(j)<ymin(i)) {
                    ymin(i) = y(j);
                }
                inloop=1;
            }
            if (inloop==0) {
                j0=j; // memorize where we started before
                }
            j++;
        }
    }
    """
    err = weave.inline(code,
                       ['x', 'y', 'dx','LX','ymin'],
                       type_converters=converters.blitz,
                       compiler = 'gcc')
    return ymin

def slidingMax(x,y,dx):
    x = np.float_(x)
    y = np.float_(y)
    LX = len(x)
    ymax = np.ones(LX)*y.min()
    code=\
    """
    int j;
    int i;
    int j0;
    int inloop;
    j0=1;
    
    for (i=0; i<LX; i++){
        j=j0;
        inloop=0;
        while ((x(j)<=x(i)+dx/2) && (j<LX) ) {
            if ((x(j)>=x(i)-dx/2) && (x(j)<=x(i)+dx/2)) {
                if (y(j)>ymax(i)) {
                    ymax(i) = y(j);
                }
                inloop=1;
            }
            if (inloop==0) {
                j0=j; // memorize where we started before
                }
            j++;
        }
    }
    """
    err = weave.inline(code,
                       ['x', 'y', 'dx','LX','ymax'],
                       type_converters=converters.blitz,
                       compiler = 'gcc')
    return ymax

def slidingMean(x,y,dx):
    x = np.float_(x)
    y = np.float_(y)
    LX = len(x)
    ym = np.zeros(LX)
    code=\
    """
    int j;
    int i;
    int j0;
    int inloop;
    j0=1;
    int n;
    
    for (i=0; i<LX; i++){
        j=j0;
        inloop=0;
        n=0;
        while ((x(j)<=x(i)+dx/2) && (j<LX) ) {
            if ((x(j)>=x(i)-dx/2) && (x(j)<=x(i)+dx/2)) {
                ym(i) += y(j);
                n+=1;    
                inloop=1;
            }
            if (inloop==0) {
                j0=j; // memorize where we started before
                }
            j++;
        }
        if (n>0) {
            ym(i)/=n; }
    }
    """
    err = weave.inline(code,
                       ['x', 'y', 'dx','LX','ym'],
                       type_converters=converters.blitz,
                       compiler = 'gcc')
    return ym

def slidingStd(x,y,dx):
    """
    compute the sliding standard deviation
    """
    x = np.float_(x)
    y = np.float_(y)
    LX = len(x)
    ym = np.zeros(LX) # result
    sMean = slidingMean(x,y,dx)
    code=\
    """
    int j;
    int i;
    int j0;
    int inloop;
    j0=1;
    int n;
    
    for (i=0; i<LX; i++){
        j=j0;
        inloop=0;
        n=0;
        while ((x(j)<=x(i)+dx/2) && (j<LX) ) {
            if ((x(j)>=x(i)-dx/2) && (x(j)<=x(i)+dx/2)) {
                ym(i) += (y(j)-sMean(i))*(y(j)-sMean(i));
                n+=1;    
                inloop=1;
            }
            if (inloop==0) {
                j0=j; // memorize where we started before
                }
            j++;
        }
        if (n>0) {
            ym(i)/=n; }
    }
    """
    err = weave.inline(code,
                       ['x', 'y', 'dx','LX','ym','sMean'],
                       type_converters=converters.blitz,
                       compiler = 'gcc')
    return np.sqrt(ym)
