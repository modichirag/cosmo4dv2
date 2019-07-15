import numpy, math
from scipy.optimize import curve_fit as cf
from scipy.optimize import minimize
#from sklearn import svm

##slope = 0
#########################################################################
def sm_scale(mass, ovd = 178., rhoc = 27.756, M = 0.3175):
    '''Return Eulerian scale of the halo for solar mass'''    
    rhobar = ovd * rhoc*M*10**10.
    return (3 * mass / (4*numpy.pi*rhobar))**(1/3.)    


def sigmoidf(delta, hold, alpha = 0.8, b=1):
    '''Pass delta through a sigmoid of threshold = hold, multiplicative coeff = alpha, power = b'''
    z = alpha * (delta**b - hold**b)    
    return 1./(1 + numpy.exp(-z))

def normalize(ar):
    '''Normalize the array by subtracting the mean and dividing by std
    '''
    mean = ar.mean()
    std = ar.std()
    return (ar - mean)/std

def svm_thresh(ft, cls, cls_wt = None):
    '''Do the svm on the ft(feature) and cl(ass) 
    '''
    if cls_wt is None:
        clf = svm.LinearSVC(max_iter=10000, dual = False)
    else:
        clf = svm.LinearSVC(max_iter=10000, dual = False, class_weight = cls_wt)
    clf.fit(ft, cls)
    return clf.coef_[0], clf.intercept_[0], clf.score(ft, cls)


def find_mask(ftlist, cls,  mesh, catpos,  width = 20., b = 1, file = None, label = "None", cls_wt = None, NN = False):
    '''Find mask passed through sigmoid'''
    nft = len(ftlist)
    ft = numpy.empty([cls.size, nft])
    
    if NN:
        nc = mesh.shape[0]
        bs = mesh.BoxSize[0]
        side = bs/nc
        pos = numpy.round(catpos/side).astype(int)
        pos[pos == nc] = 0
        for foo in range(nft):
            ft[:, foo] = ftlist[foo][tuple(pos.T)]
        
    else:
        for foo in range(nft):
            try:
                mesh.real[:] = ftlist[foo]
            except:
                mesh[...] = ftlist[foo]
            ft[:, foo] = mesh.readout(catpos)
    
    coeff, intercept, score = svm_thresh(ft, cls, cls_wt)
    if file is not None:
        file.write("label = %s , nft = %d\n"%(label, nft))
        file.write("coeff = %s, intercept = %s \n"%(str(coeff), str(intercept)))
        file.write("threshold = %s, score = %0.3f \n"%(str(intercept/coeff[0]), score))

    try:
        pmgrid = numpy.zeros_like(mesh.real)
    except:
        pmgrid = numpy.zeros_like(mesh[...])
    for foo in range(nft):
        pmgrid += coeff[foo]*ftlist[foo]
        
    mask = sigmoidf(pmgrid, -intercept, width, b = b )

    return mask, coeff, intercept
    



#########################################################################
###Functions for fitting mass
def fit_log(x, *p):
   ''' y = b*numpy.log10(1 + a*x) + c'''
   a, b, c = p
   x2 = numpy.log10(1 + a*x)
   return b*x2 + c

def quad_exp(x, *p):
    '''y = 10**z *(ax**2 + b*x + c)'''
    a, b, c, y  = p
    return (10**y) * (a*x**2 + b*x+ c )

def chi_sqf(p, xdata, ydata, sigma):
    model =  quad_exp(xdata, *p)
    return (((model - ydata)/sigma)**2).sum()

def quad_exp_der(p, xdata, ydata, sigma):
    model = quad_exp(xdata, *p)
    fac = (2*(model - ydata)/sigma)
    a, b, c, y  = p
    #a, b, c, y  = p; model = (10**y) * (a*x**2 + b*x+ c )
    dmdy = numpy.log(10)*model*fac
    dmda = (10**y)*(xdata**2)*fac
    dmdb = (10**y)*xdata*fac
    dmdc = (10**y)*fac
    
    return numpy.array([dmda.sum(), dmdb.sum(), dmdc.sum(), dmdy.sum()]).T




def fitovd(Y, X, func = chi_sqf, p0 = [1, 1, 1, 1], lim = 0, sigma = True, absig = False, \
    retdata = False, tol = 0):
    
    xdata = X.astype("float64").flatten()
    ydata = Y.astype("float64").flatten()

    pos = numpy.where(ydata > (lim))[0]
    ydata = ydata[pos]
    xdata = xdata[pos]

        
    if sigma:
        sigmaval = ydata.copy()
        sigmaval[sigmaval == 0] = 1
    else:
        sigmaval = numpy.ones_like(ydata)

#    res = minimize(func, x0 = p0, args =(xdata, ydata, sigmaval), jac = quad_exp_der,\
#                   method='L-BFGS-B', options = {'ftol' : 10**-15, 'gtol':10**-10})
    res = minimize(func, x0 = p0, args =(xdata, ydata, sigmaval), jac = quad_exp_der,\
                   method='BFGS', options = {'gtol':10**-10})

#     if tol:
#         res = minimize(func, x0 = p0, args =(xdata, ydata, sigmaval), method = 'Nelder-Mead', tol=10**-10)
#     else:res = minimize(func, x0 = p0, args =(xdata, ydata, sigmaval), method = 'Nelder-Mead', options ={'fatol':0.0001})
    print(res.message, res.nit)
    sigmass = quad_exp(X, *res.x)
    return sigmass, res.x




##########################################################################################################


def fitovdmask(Y, X, func = chi_sqf, p0 = [1, 1, 1, 1], lim = 0, sigma = True, absig = False, \
    retdata = False, tol = 0):
    
    xdata = X.astype("float64").flatten()
    ydata = Y.astype("float64").flatten()

    pos = numpy.where(ydata > (lim))[0]
    ydata = ydata[pos]
    xdata = xdata[pos]

        
    if sigma:
        sigmaval = ydata.copy()
        sigmaval[sigmaval == 0] = 1
    else:
        sigmaval = numpy.ones_like(ydata)

#    res = minimize(func, x0 = p0, args =(xdata, ydata, sigmaval), jac = quad_exp_der,\
#                   method='L-BFGS-B', options = {'ftol' : 10**-15, 'gtol':10**-10})
    res = minimize(func, x0 = p0, args =(xdata, ydata, sigmaval), jac = quad_exp_der,\
                   method='BFGS', options = {'gtol':10**-10})

    print(res.message, res.nit)
    sigmass = quad_exp(X, *res.x)
    return sigmass, res.x



def domass_10min(Y, X, func = chi_sqf, p0 = [1, 1, 1, 1], lim = False, sigma = True, absig = False, \
    abund = False, retdata = False, nonzeroy = True, ranzero = 0, tol = 0):
    
    xdata = X.astype("float64").flatten()
    ydata = Y.astype("float64").flatten()
    if nonzeroy:
        pos = numpy.where(ydata > 0)[0]
        ydata = ydata[pos]
        xdata = xdata[pos]
    if abund:
        xdata = numpy.sort(xdata)[::-1]
        ydata = numpy.sort(ydata)[::-1]

    if lim:
        pos = numpy.where(ydata > (lim))[0]
        ydata = ydata[pos]
        xdata = xdata[pos]

    if ranzero:
        posz = numpy.where(ydata == 0)[0]
        posz = numpy.random.permutation(posz)
        pos = numpy.concatenate((pos, posz[:int(ranzero*Y.size/100)]))
        
    if sigma:
        sigmaval = ydata.copy()
        sigmaval[sigmaval == 0] = 1
    else:
        sigmaval = numpy.ones_like(ydata)

#    res = minimize(func, x0 = p0, args =(xdata, ydata, sigmaval), jac = quad_exp_der,\
#                   method='L-BFGS-B', options = {'ftol' : 10**-15, 'gtol':10**-10})
    res = minimize(func, x0 = p0, args =(xdata, ydata, sigmaval), jac = quad_exp_der,\
                   method='BFGS', options = {'gtol':10**-10})

#     if tol:
#         res = minimize(func, x0 = p0, args =(xdata, ydata, sigmaval), method = 'Nelder-Mead', tol=10**-10)
#     else:res = minimize(func, x0 = p0, args =(xdata, ydata, sigmaval), method = 'Nelder-Mead', options ={'fatol':0.0001})
    print(res.message, res.nit)
    sigmass = quad_exp(X, *res.x)
    if retdata:
        return sigmass, res.x, xdata, ydata
    else:
        return sigmass, res.x



##########################################################################################################

def predictpos(meshdict, pdict):

    ftname = pdict['ftname']
    width = pdict['width']
    coeff = pdict['pcoeff']
    intercept = pdict['intercept']

    ftlist = [meshdict[i] for i in ftname]    
    pmgrid = numpy.zeros_like(meshdict['final'])
    for i in range(len(coeff)):
        pmgrid += ftlist[i] * coeff[i]
    
    mask = sigmoidf(pmgrid, -intercept, width)
    return mask


def predictmass(meshdict, pdict):

    ftname = pdict['ftname']
    poptm = pdict['poptm']
    mbase = pdict['mbase']
    fac = pdict['mfactor']

    mbase = meshdict[mbase]
    mpred = quad_exp(mbase, *poptm)*fac
    if pdict['fitovd']:
        mpred += fac
    return mpred


##
##def parseinfo(ppath):
##    mmin, mmean = 0, 0
##    for line in open(ppath):
##        if 'Features' in line:
##            ftname = (line.split("'")[1::2])
##        if 'sigmoid parameters' in line:
##            sigp = [float(i) for i in line.split('=')[1].split()]
##        if 'coeff' in line:
##            coeff = [float(i) for i in line.split('=')[1].split()]
##        if 'intercept' in line:
##            intercept = [float(i) for i in line.split('=')[1].split()][0]
##        if 'mass parameters' in line:
##            poptm = [float(i) for i in line.split('=')[1].split()]
##        if 'Min mass' in line:
##            mmin = float(line.split('=')[1])
##        if 'Mean mass' in line:
##            mmean = float(line.split('=')[1])
##        if 'Mbase' in line:
##            base = (line.split("=")[1].split()[0])    
##    
##    fac = max(mmin, mmean)
##    if mmean:
##        add = True
##    else:
##        add = False
##    mprop = [fac, add, base]
##    tup = (ftname, sigp, coeff, intercept, poptm, mprop)
##    tdict = {'ftname':ftname, 'sigp':sigp, 'coeff':coeff, 'intercept':intercept, 'poptm':poptm, 'mprop':mprop}
##    return tup, tdict
##
##
##
