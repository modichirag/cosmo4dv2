import numpy
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from matplotlib.colors import SymLogNorm
from scipy.optimize import minimize
from nbodykit.lab import FFTPower

colors= ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']

def chi2(y1, y2):
    return ((y1-y2)**2).sum()

def normal(x, *p):
    A, mu, sig = p
    f = 1/sig/numpy.sqrt(2*numpy.pi)
    e = ((x) - mu)**2./2/sig**2
    if sig <=0: return np.inf
    return A *f * numpy.exp(-e)

def lognorm(x, *p):
    A, mu, sig = p
    f = 1/x/sig/numpy.sqrt(2*numpy.pi)
    e = (numpy.log(x) - mu)**2./2/sig**2
    return A *f * numpy.exp(-e)


def laplace(x, *p):
    A, mu, b = p
    f = A* 1/(2*b)*np.exp(-abs(x-mu)/b)
    return f


def ftomin(p, datay, datax, func):
    ymod = func(datax, *p)
    return chi2(datay, ymod)


def fitpdf(data,  func = lognorm, x0 = [100, 0, 0.1], bins= numpy.logspace(-2, 1, 1000), solvopt = None, normed=False):
    
    datay, datax = numpy.histogram(data, bins = bins, range = (bins[0], bins[-1]), density=normed)
    datax = (datax[1:] + datax[:-1])/2.
    if solvopt is None:
        solvopt = {'disp':False,'maxfev':10000,'xtol':1e-7}
    if func is not None:
        res = minimize(ftomin, x0 = x0, args = (datay, datax, func), \
                   method = "Nelder-Mead", options=solvopt)
    else:
        res = None
    return datax, datay, res

    
def lognormaltoratio(data, lbinfit = None):
    '''Fit lognormal distribution to ratio of columns of data
    data is (x, y) and diff fitted for is (log(x)-log(y))
    '''
    if lbinfit is None:
        lbinfit = numpy.logspace(-3, 1, 2000)
        
    fit = []
    bindata = []
    for foo in range(len(data)):
        ratio = data[foo][1]/data[foo][0]    
        datax, datay, res = fitpdf(ratio, func=lognorm, x0=[100, 0, 0.1], bins = lbinfit)
        fit.append(res)
        bindata.append([datax, datay])
    return fit, bindata
    
def normaltolog(data, binfit = None):
    '''Fit normal distribution to difference of log of columns
    '''
    if binfit is None:
        binfit = numpy.linspace(-7, 7, 1000)
    fit = []
    bindata = []
    for foo in range(len(data)):
        y = data[foo][1].copy()
        x = data[foo][0].copy()
        zeros = numpy.where(y < 1.)
        y[zeros] = 1
        diff = numpy.log(x) - numpy.log(y)
        datax, datay, res = fitpdf(diff, func=normal, x0=[100, 0, 0.1], bins = binfit)
        fit.append(res)
        bindata.append([datax, datay])
    return fit, bindata


def functolog(func, data, M0, binfit = None, x0=[100, 0, 0.5], normed=False):
    '''Fit laplace distribution to difference of log of columns
    data is (x, y) and diff fitted for is (log(x)-log(y))
    '''
    if binfit is None:
        binfit = numpy.linspace(-7, 7, 500)
    fit = []
    bindata = []
    for foo in range(len(data)):
        y = data[foo][1].copy() + M0
        x = data[foo][0].copy() + M0
        #zeros = numpy.where(y < 1.)
        #y[zeros] = 1
        try:
            diff = numpy.log(x) - numpy.log(y)
        except:
            diff = numpy.ones_like(x)
        datax, datay, res = fitpdf(diff, func=func, x0=x0, bins = binfit, normed=normed)
        fit.append(res)
        bindata.append([datax, datay])
    return fit, bindata
    



def functom(func, data, M0, binfit = None, x0=[100, 0, 0.5], normed=False):
    '''Fit laplace distribution to difference of log of columns
    data is (x, y) and diff fitted for is ((x)-(y))/M0
    '''
    if binfit is None:
        binfit = numpy.linspace(-7, 7, 500)
    fit = []
    bindata = []
    for foo in range(len(data)):
        y = data[foo][1].copy() 
        x = data[foo][0].copy() 
        #zeros = numpy.where(y < 1.)
        #y[zeros] = 1
        diff = (x-y)/M0
        try:
            if binfit.shape[1] > 1: bins = binfit[foo]
        except: bins = binfit
            
        datax, datay, res = fitpdf(diff, func=func, x0=x0, bins = bins, normed=normed)
        fit.append(res)
        bindata.append([datax, datay])
    return fit, bindata


def calc_allscatter(predict, true, scatter, mbin):
    '''calcualte all 4 combinations of scatter between 3 grids
    '''
    #sc_at_p_with_t
    scatpwt = gridscatter(predict, true, mbin)
    scatpws = gridscatter(predict, scatter, mbin)
    scattwp = gridscatter(true, predict, mbin)
    scattws = gridscatter(true, scatter, mbin)
    
    return scatpwt, scatpws, scattwp, scattws
    


def typeerror(predict, truth, prthresh, trthresh):
    mask = truth > trthresh
    predict1 = predict[mask]
    type1 = predict1[predict1 < prthresh]
    predict2 = predict[~mask]
    type2 = predict2[predict2 > prthresh]
    return type1, type2
    


def gridscatter(gridP, gridS, mbin, gridbase=None):
    '''Take in two grids, X(hgrid) and Y(sgrid), and return the value for gridS(2nd grid) in slices of gridP(first grid)
    if gridbase is None. Else use gridbase to search for slices in mbin
    '''
    if type(gridP) is list:
        print('List type detected, assuming list of meshes to be combined')
        return (gridscattercombine(gridP, gridS, mbin))
    else: pass
    scatter = []
    mhigh = 1e17
    if gridbase is None: gridbase = gridP
    pos = numpy.where(gridbase > mbin[0])
    scatter.append([gridP[pos], gridS[pos]])
    for foo in range(len(mbin) - 1):
        mhigh = mbin[foo]
        mlow = mbin[foo+1]
        pos = numpy.where((gridbase > mlow) & (gridbase < mhigh))
        scatter.append([gridP[pos], gridS[pos]])
        #dummy, dummy2 = numpy.histogram(gridP[pos] - gridS[pos], range = (0.1, 10), bins = 100)
    return scatter    


def gridscattercombine(gridPlist, gridSlist, mbin):
    '''Take in two grids, X(hgrid) and Y(sgrid), and return the value for gridS(2nd grid) in slices of gridP(first grid)
    '''
    scatters = []
    for i in range(len(gridPlist)):
        scatters.append(gridscatter(gridPlist[i], gridSlist[i], mbin))

    ns = len(scatters)
    scatter = []
    for j in range(len(scatters[0])):
        scatter.append([np.concatenate([scatters[i][j][0] for i in range(ns)]), np.concatenate([scatters[i][j][1] for i in range(ns)])]) 
    return scatter    



def plot_normalnoise( grid1, grid2, mbin = None, label1 ="", label2="", c='k', binfit = None, xtest = None, axin = None, fsize = 14, 
                      msd = 0.1, alphad = 0.3, lwf = 1):
    '''Calculate difference in 2 grids (grid1-grid2) in bins=mbin on grid1, fit it with lognormal and plot that normal
    '''
    if axin is None:
        fig, axar = plt.subplots(2, 3, figsize = (12, 8))
    else:
        axar = axin
    if mbin is None:
        mbin = numpy.logspace(5, 12, 8, endpoint=True)[::-1]
    if binfit is None:
        binfit = numpy.linspace(-7, 7, 500)
    if xtest is None:
        xtest = binfit

    scatter = gridscatter(grid1, grid2, mbin)
    fit, data = normaltolog(scatter, binfit = binfit)
    for foo in range(axar.flatten().size):
        try:
            ax = axar.flatten()[foo]
        except:
            continue
        ax.set_title("1 dex above %0.2e"%mbin[foo], fontsize = fsize)
    
        ax.plot(*data[foo], ms = msd, color = c, marker = ".", alpha = alphad)
        l = ax.plot(xtest, normal(xtest, *fit[foo].x), "-", color = c, lw = lwf, 
                     label = "$\mu$ = %0.2f, \n $\sigma$ = %0.2f"%(fit[foo].x[1], fit[foo].x[2]))

        ax.legend(loc = 0, fontsize = fsize-2, frameon = False)
        if foo is 0:
            l[0].set_label("$ln(M_{%s}) - ln(M_{%s})$"%(label1, label2))
            ax.legend( fontsize = fsize-1)
    
        if foo > 2:
            ax.set_xlabel(r"$ln(M_1) - ln(M_2)$", fontsize = fsize)

    plt.tight_layout()
    if axin is None:
        return fig, axar


def plot_noise( grid1, grid2, M0, func = None, mbin = None, label1 ="", label2="", label="", binfit = None, xtest = None, axin = None, fsize = 14, 
                      msd = 4, c='k', alphad = 0.3, lw = 0.5, lwf = 1, lsf = "--", alphaf = 1, cf=None, retfit = False, pltdata=True, x0=[1000, 0, 0.5], 
                normed=False, gridbase=None):
    '''Calculate difference (log(grid1+M0)-log(grid2+M0)) in 2 grids in bins=mbin on grid1, fit the difference with pdf='func'
    '''
    if axin is None:
        fig, axar = plt.subplots(2, 3, figsize = (15, 10))
    else:
        axar = axin
    if mbin is None:
        mbin = numpy.logspace(5, 12, 8, endpoint=True)[::-1]
    if cf is None: cf = c
    if binfit is None:
        binfit = numpy.linspace(-7, 7, 500)
    if xtest is None:
        xtest = binfit

    scatter = gridscatter(grid1, grid2, mbin, gridbase=gridbase)
    fit, data = functolog(func, scatter, M0 = M0, binfit = binfit, x0=x0, normed=normed)

    for foo in range(axar.flatten().size):
        try:
            ax = axar.flatten()[foo]
        except:
            continue
        ax.set_title("1 dex above %0.2e"%mbin[foo], fontsize = fsize)
    
        if pltdata:
            p = ax.plot(*data[foo], ms = msd, color = c, marker = "o", alpha = alphad, lw=lw)
        if func is normal:
            sigma = '\sigma'
        elif func is laplace:
            sigma = 'b'
        if func is not None:
            l = ax.plot(xtest, func(xtest, *fit[foo].x), ls = lsf, color = cf, lw = lwf, alpha=alphaf, 
                     label = "$\mu$ = %0.2f, \n $%s$ = %0.2f"%(fit[foo].x[1], sigma, fit[foo].x[2]))

        ax.legend(loc = 0, fontsize = fsize-2, frameon = False)
        ax.set_xlim(xtest[0], xtest[-1])
        if foo == axar.size -1 and pltdata:
            if len(label1) > 1:
                p[0].set_label("$ln(M_{%s}) - ln(M_{%s})$"%(label1, label2))
                ax.legend( fontsize = fsize-1)
            elif len(label) > 1:
                p[0].set_label(label)
                ax.legend( fontsize = fsize-1)

        if foo > 2:
            ax.set_xlabel(r"$ln(M_1+M_0) - ln(M_2+M_0)$", fontsize = fsize)

    plt.tight_layout()
    toret = []
    if retfit:
        toret.append(fit)
    if axin is None:
        toret.append(fig)
        toret.append(axar)
    if len(toret):
        return toret




def plot_mnoise( grid1, grid2, M0, func = None, mbin = None, label1 ="", label2="", label="", binfit = None, xtest = None, axin = None, fsize = 14, 
                      msd = 4, c='k', alphad = 0.3, lw = 0.5, lwf = 1, lsf = "--", alphaf = 1, cf=None, retfit = False, pltdata=True, x0=[1000, 0, 0.5], 
                normed=False):
    '''Calculate difference (log(grid1+M0)-log(grid2+M0)) in 2 grids in bins=mbin on grid1, fit the difference with pdf='func'
    '''
    if axin is None:
        fig, axar = plt.subplots(2, 3, figsize = (15, 10))
    else:
        axar = axin
    if mbin is None:
        mbin = numpy.logspace(5, 12, 8, endpoint=True)[::-1]
    if cf is None: cf = c
    if binfit is None:
        binfit = numpy.linspace(-100, 100, 500)
    if xtest is None: 
        makex = True
    else: makex = False
    scatter = gridscatter(grid1, grid2, mbin)
    fit, data = functom(func, scatter, M0 = M0, binfit = binfit, x0=x0, normed=normed)

    for foo in range(axar.flatten().size):
        try:
            ax = axar.flatten()[foo]
        except:
            continue
        ax.set_title("1 dex above %0.2e"%mbin[foo], fontsize = fsize)
    
        if makex:
            try:
                if binfit.shape[1]>1: 
                    xtest = binfit[foo]
            except: xtest = binfit

        if pltdata:
            p = ax.plot(*data[foo], ms = msd, color = c, marker = "o", alpha = alphad, lw=lw)
        if func is normal:
            sigma = '\sigma'
        elif func is laplace:
            sigma = 'b'
        if func is not None:
            l = ax.plot(xtest, func(xtest, *fit[foo].x), ls = lsf, color = cf, lw = lwf, alpha=alphaf, 
                     label = "$\mu$ = %0.2f, \n $%s$ = %0.2f"%(fit[foo].x[1], sigma, fit[foo].x[2]))

        ax.legend(loc = 0, fontsize = fsize-2, frameon = False)
        ax.set_xlim(xtest[0], xtest[-1])
        if foo == axar.size -1 and pltdata:
            if len(label1) > 1:
                p[0].set_label("$(M_{%s} - M_{%s})/M0$"%(label1, label2))
                ax.legend( fontsize = fsize-1)
            elif len(label) > 1:
                p[0].set_label(label)
                ax.legend( fontsize = fsize-1)

        if foo > 2:
            ax.set_xlabel(r"$(M_{1} - M_{2})/M_0$", fontsize = fsize)

    plt.tight_layout()
    toret = []
    if retfit:
        toret.append(fit)
    if axin is None:
        toret.append(fig)
        toret.append(axar)
    if len(toret):
        return toret



def plot_2pt( hfield, pfield , color = 'k', label = "", ls = "-", lw = 1, alpha = 1, axin = None, returnp = False, tfdensity=False, pm=None, c=None, meannorm=True):
    '''Plot stochasticity, cross correlation and transfer function, given 2 fields 'h'(primary) and 'f'
    '''
    if axin is None:
        fig, axar = plt.subplots(1, 3, figsize = (15, 4))
    else:
        axar = axin
    
    if c is not None:
        color=c
    if type(hfield) == numpy.ndarray:
        if pm is None:
            print('Need real fields or pm object to convert given arrays to real fields')
            return None
        else:
            h1, p1 = hfield.copy(), pfield.copy()
            hfield, pfield = pm.create(mode='real', value=0), pm.create(mode='real', value=0)
            hfield[...], pfield[...] = h1, p1


    if meannorm:
        hmean, pmean = hfield.cmean(), pfield.cmean()
    else:
        hmean, pmean = 1, 1
    halop = FFTPower(hfield/hmean, mode = '1d').power['power']
    fieldp = FFTPower(pfield/pmean, mode = '1d').power['power']
    cross = FFTPower(hfield/hmean, second = pfield/pmean, mode = '1d').power
    k, crossp = cross['k'], cross['power']

    cmap = plt.cm.jet
    cstyle = numpy.empty([cmap.N, 4])
    for i in range(cmap.N):
        cstyle[i,:] = cmap(i)
    cstyle = cstyle[::50, :]

    if not type(color) == str:
        color = cstyle[color]
    axar[0].plot(k, crossp/(halop*fieldp)**0.5, label = label, ls = ls, lw = lw, alpha = alpha, color = color)  
    axar[1].plot(k, halop - crossp**2./fieldp, label = label, ls = ls, lw = lw, alpha = alpha, color = color)
    if tfdensity:
        axar[2].plot(k, (fieldp/halop)**0.5, label = label, ls = ls, lw = lw, alpha = alpha, color =  color)  
        pass
    else:
        axar[2].plot(k, fieldp/halop, label = label, ls = ls, lw = lw, alpha = alpha, color =  color)  

    axar[0].set_ylabel("Cross correlation", fontsize = 14)
    axar[0].set_ylim(0.8,1.1)
    axar[0].axhline(1, lw = 0.5)
    axar[0].legend(loc = 0, fontsize = 13, ncol = 2)
    
    axar[1].set_ylim(10**-1, 10**5.)
    axar[1].set_yscale("log")
    axar[1].set_ylabel("Stochasticity", fontsize = 14)
    
    if tfdensity:
        print('Transfer function is for density')
        axar[2].set_ylabel("Transfer Function", fontsize = 14)
    else:
        print('Transfer function is for power')
        axar[2].set_ylabel("Transfer Function", fontsize = 14)
    axar[2].set_ylim(0.5,1.5)
    axar[2].axhline(1, lw = 0.5)
    
    for ax in axar:
        ax.set_xscale("log")
        ax.set_xlabel("k (h/Mpc)", fontsize = 14)

    plt.tight_layout()

    if returnp:        
        if axin is None:
            return fig, axar, np.array([k, halop, fieldp, crossp])
        else:
            return  np.array([k, halop, fieldp, crossp])
    elif axin is None:
        return fig, axar





def plot_2pt2d( hfield, pfield , color = 'k', label = "", ls = "-", lw = 1, alpha = 1, axin = None, returnp = False, tfdensity=False, pm=None, c=None, meannorm=True, Nmu=5, los=[0, 0, 1], muplot=[0], lsmu=['-', '--', ':', '-.'], lsmustyle=True, mulab=True):
    '''Plot stochasticity, cross correlation and transfer function, given 2 fields 'h'(primary) and 'f'
    '''
    if axin is None:
        fig, axar = plt.subplots(1, 3, figsize = (15, 4))
    else:
        axar = axin
    
    if c is not None:
        color=c
    if type(hfield) == numpy.ndarray:
        if pm is None:
            print('Need real fields or pm object to convert given arrays to real fields')
            return None
        else:
            h1, p1 = hfield.copy(), pfield.copy()
            hfield, pfield = pm.create(mode='real', value=0), pm.create(mode='real', value=0)
            hfield[...], pfield[...] = h1, p1


    if meannorm:
        hmean, pmean = hfield.cmean(), pfield.cmean()
    else:
        hmean, pmean = 1, 1
    halop = FFTPower(hfield/hmean, mode = '2d', Nmu=Nmu, los=los).power['power']
    fieldp = FFTPower(pfield/pmean, mode = '2d', Nmu=Nmu, los=los).power['power']
    cross = FFTPower(hfield/hmean, second = pfield/pmean, mode = '2d', Nmu=Nmu, los=los).power
    k, mus, crossp = cross.coords['k'], cross.coords['mu'], cross['power']

    cmap = plt.cm.jet
    cstyle = numpy.empty([cmap.N, 4])
    for i in range(cmap.N):
        cstyle[i,:] = cmap(i)
    cstyle = cstyle[::50, :]

    if not type(color) == str:
        color = cstyle[color]

    addmulab = True
    for nmu, imu in enumerate(muplot):
        if lsmustyle: ls = lsmu[nmu%len(lsmu)]
        else: ls = ls 
        if mulab:
            if addmulab and nmu == 0: 
                labelp = label + '\n$\mu$ = %0.2f'%mus[imu]
            else: labelp = '$\mu$ = %0.2f'%mus[imu]
        else: 
            if nmu == 0: labelp = label
            else: labelp = None
            
        axar[0].plot(k, (crossp/(halop*fieldp)**0.5)[:, imu], label = labelp, ls = ls, lw = lw, alpha = alpha, color = color)  
        axar[1].plot(k, (halop - crossp**2./fieldp)[:, imu], label = labelp, ls = ls, lw = lw, alpha = alpha, color = color)
        if tfdensity:
            axar[2].plot(k, (fieldp/halop)[:, imu]**0.5, label = labelp, ls = ls, lw = lw, alpha = alpha, color =  color)  
        else:
            axar[2].plot(k, (fieldp/halop)[:, imu], label = labelp, ls = ls, lw = lw, alpha = alpha, color =  color)  

    axar[0].set_ylabel("Cross correlation", fontsize = 14)
    axar[0].set_ylim(0.8,1.1)
    axar[0].axhline(1, lw = 0.5)
    axar[0].legend(loc = 0, fontsize = 13, ncol = 2)
    
    axar[1].set_ylim(10**-1, 10**5.)
    axar[1].set_yscale("log")
    axar[1].set_ylabel("Stochasticity", fontsize = 14)
    
    if tfdensity:
        print('Transfer function is for density')
        axar[2].set_ylabel("Transfer Function", fontsize = 14)
    else:
        print('Transfer function is for power')
        axar[2].set_ylabel("Transfer Function", fontsize = 14)
    axar[2].set_ylim(0.5,1.5)
    axar[2].axhline(1, lw = 0.5)
    
    for ax in axar:
        ax.set_xscale("log")
        ax.set_xlabel("k (h/Mpc)", fontsize = 14)

    plt.tight_layout()

    if returnp:        
        if axin is None:
            return fig, axar, np.array([k, halop, fieldp, crossp])
        else:
            return  np.array([k, halop, fieldp, crossp])
    elif axin is None:
        return fig, axar



def visual(f00, f01, f10=None, f11=None, full=False, x1=None, x2=None, y1=None, y2=None, z1=None, z2=None, v1=None, v2=None, proj=-1,
           l00='Nets', l01='FOF', l10='Nets', l11='FOF', axin=None, supt=None):
    nc = f00.shape[0]
    if full:
        x1, y1, z1 = 0, 0, 0
        z2, y2, z2 = nc, nc, nc
    else:
        x1, x2 = nc//3, 2*nc//3
        y1, y2 = nc//3, 2*nc//3
        z1, z2 = nc//2, nc//2 + 5
        if v1 is None:
            v1 = -1e12
        if v2 is None:
            v2 = 1e12
    # v1, v2 = -2, 2

    symnorm = SymLogNorm(linthresh= 1e5, vmin=v1, vmax=v2)

    if axin is None:
        if f10 is None:
            fig, axar = plt.subplots(1, 2, figsize = (8, 4))
            axar = axar.reshape(1, 2)
        else:
            fig, axar = plt.subplots(2, 2, figsize = (8, 7))
    else:
        axar = axin

    ax = axar[0]
    ax[0].imshow(f00[x1:x2, y1:y2, z1:z2].sum(axis = proj), cmap = "RdBu_r", vmin = v1, vmax = v2)
    ax[0].set_title(l00)
    ax[1].imshow(f01[x1:x2, y1:y2, z1:z2].sum(axis = proj), cmap = "RdBu_r", vmin = v1, vmax = v2)
    ax[1].set_title(l01)
    if f10 is not None:
        ax = axar[1]
        ax[0].imshow(f10[x1:x2, y1:y2, z1:z2].sum(axis = proj), cmap = "RdBu_r", vmin = v1, vmax = v2)
        ax[0].set_title(l10)
        ax[1].imshow(f11[x1:x2, y1:y2, z1:z2].sum(axis = proj), cmap = "RdBu_r", vmin = v1, vmax = v2)
        ax[1].set_title(l11)
    # ax[2].imshow(pmhall[x1:x2, y1:y2, z1:z2].sum(axis = -1), cmap = "RdBu_r", vmin = v1, vmax = v2)
    # ax[2].set_title('Truth')
    if supt is not None:
        plt.suptitle(supt)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if axin is None:
        return fig, axar


def imresidual(f00, f01, x1=None, x2=None, y1=None, y2=None, z1=None, z2=None, v1=None, v2=None, vr1 = None, vr2 = None, proj=-1,
           l00='Nets', l01='FOF', axin=None):

    if axin is None:
        fig, axar = plt.subplots(1, 3, figsize = (12, 4))
    else:
        axar=axin

    ax=axar
    ax[0].imshow(f00[x1:x2, y1:y2, z1:z2].sum(axis = proj), cmap = "RdBu_r", vmin = v1, vmax = v2)
    ax[0].set_title(l00)
    ax[1].imshow(f01[x1:x2, y1:y2, z1:z2].sum(axis = proj), cmap = "RdBu_r", vmin = v1, vmax = v2)
    ax[1].set_title(l01)
    ax[2].imshow((f00 - f01)[x1:x2, y1:y2, z1:z2].sum(axis = proj), cmap = "RdBu_r", vmin = vr1, vmax = vr2)
    ax[2].set_title('Residual')

    if axin is None:
        return fig, axar




def visnetwork(kobj):
    layers = []
    for l in kobj.layers:
        l0 = l.get_weights()
        if len(l0):
            layers.append(l0)
    nl = len(layers)
    
    fig, axar = plt.subplots(2, nl, figsize = (4*nl, 7))
    ax = axar[0]
    for i in range(nl):
        ax[i].plot(np.log10(abs(layers[i][0].flatten())), ".")
    ax = axar[1]
    for i in range(nl):
        ax[i].plot(np.log10(abs(layers[i][1].flatten())), ".")
    
    return fig, ax



def scatter_halos(hmass, sigma, seed=100):
    '''Take in halo mass in solar mass and return the log-normally scattered mass                                                                              
    '''
    logl = numpy.log10(hmass)
    rng = numpy.random.RandomState(seed)
    t = rng.normal(scale=sigma, size=len(logl))
    logl = logl + t
    return 10**logl


def scatter_catalog(hmass, hpos, sigma, seed=123, hvel=None, truemass=False):
    halomass2 = scatter_halos(hmass, sigma, seed=seed)
    sort2 = numpy.argsort(halomass2)[::-1]
    halomass2 = halomass2[sort2]
    halopos2 = hpos[sort2]
    halomass0 = hmass[sort2]
    if hvel is None :
        if truemass is False:
            return halomass2, halopos2
        else: return halomass2, halopos2, halomass0
    else:
        if truemass is False:
            return halomass2, halopos2, hvel[sort2]
        else: return halomass2, halopos2, hvel[sort2], halomass0


def neuralnetpredict(pm, bs, nc, numd, seed, zz=0):
    '''For the given box, mesh, numd, seed and redshift, do the neural network prediction
    from the network saved in cfg file and return predict, nnpred, nnmass
    '''
    dpath = '/global/project/projectdirs/astro250/chmodi/cosmo4d/'

    import yaml
    import nettools as ntools

    with open(dpath + 'train/models.yaml', 'r') as ymlfile:
        cfg = yaml.load(ymlfile)


    ptpath = dpath + 'train/' + cfg['%s-%s'%(int(bs),int(nc))][0][numd]['ppath']
    mtpath = ptpath + cfg['%s-%s'%(int(bs),int(nc))][0][numd]['mpath']
    # mtpath = ptpath + cfg['%s-%s'%(bs,nc)][0][numd]['mpathnf']
    ptup, pftname, plocal, pdict = ntools.setuppos2(ptpath)
    mtup, mftname, mlocal, mdict = ntools.setupmass(mtpath )
    R1, R2 = pdict['R1'], pdict['R2']

    num = int(numd*bs**3)
    meshdict, hdict = ntools.readfiles(pm, dpath + '/data/z%02d/L%04d_N%04d_S%04d_05step/'%(zz, bs, nc, seed), R1=R1, R2=R2, mexp=None)

    ftt = ntools.createdata(pm, meshdict, pdict['pftname'], plocal)
    mftt = ntools.createdata(pm, meshdict, mftname, mlocal)

    nnpred = ntools.applynet(ftt, ptup).reshape(nc, nc, nc)
    nnmass = ntools.applynet(mftt, mtup).reshape(nc, nc, nc)
    predict = nnpred*nnmass
    ##
    return pm.create(mode='real', value=predict), nnpred, nnmass



##
##def visual2(f00, f01, f10, f11, full=False, x1=None, x2=None, y1=None, y2=None, z1=None, z2=None, v1=None, v2=None, proj=-1, axin=None, 
##           l00='Nets', l01='Truth', l10='Nets', l11='Truth'):
##
##    nc = f00.shape[0]
##    if full:
##        x1, y1, z1 = 0, 0, 0
##        z2, y2, z2 = nc, nc, nc
##    else:
##        x1, x2 = nc//3, 2*nc//3
##        y1, y2 = nc//3, 2*nc//3
##        z1, z2 = nc//2, nc//2 + 5
##        if v1 is None:
##            v1 = -1e12
##        if v2 is None:
##            v2 = 1e12
##    # v1, v2 = -2, 2
##
##    symnorm = SymLogNorm(linthresh= 1e5, vmin=v1, vmax=v2)
##
##    if axin is None:
##        fig, axar = plt.subplots(2, 2, figsize = (8, 7))
##    else:
##        axar = axin
##
##    ax = axar[0]
##    ax[0].imshow(f00[x1:x2, y1:y2, z1:z2].sum(axis = proj), cmap = "RdBu_r", vmin = v1, vmax = v2)
##    ax[0].set_title('Nets')
##    ax[1].imshow(f01[x1:x2, y1:y2, z1:z2].sum(axis = proj), cmap = "RdBu_r", vmin = v1, vmax = v2)
##    ax[1].set_title('Truth')
##    ax = axar[1]
##    ax[0].imshow(f10[x1:x2, y1:y2, z1:z2].sum(axis = proj), cmap = "RdBu_r", vmin = v1, vmax = v2)
##    ax[0].set_title('Nets')
##    ax[1].imshow(f11[x1:x2, y1:y2, z1:z2].sum(axis = proj), cmap = "RdBu_r", vmin = v1, vmax = v2)
##    ax[1].set_title('Truth')
##    # ax[2].imshow(pmhall[x1:x2, y1:y2, z1:z2].sum(axis = -1), cmap = "RdBu_r", vmin = v1, vmax = v2)
##    # ax[2].set_title('Truth')
##    plt.suptitle('Visual Comparison of Fields, smoothed and unsmoothed')
##    plt.tight_layout(rect=[0, 0, 1, 0.95])
##    
##    if axin is None:
##        return fig, axar
##
