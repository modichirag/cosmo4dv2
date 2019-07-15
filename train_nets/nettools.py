import numpy as np
import features as ft
import datalib
import diagnostic as dg
import re, os, h5py, json
from functools import partial
from nbodykit.source.catalog import BigFileCatalog
from nbodykit.lab import cosmology, HaloCatalog, Zheng07Model
from pmesh.pm import ParticleMesh
import sys
sys.path.append('/global/homes/c/chmodi/Programs/cosmo4d/cosmo4d/')
import mymass_function as MF


pfile = "/global/homes/c/chmodi/Data/ics_matterpow_0.dat"
mf = MF.Mass_Func(pfile, 0.3175)
#proj = '/project/projectdirs/cosmosim/lbl/chmodi/cosmo4d/'
#proj = '/global/cscratch1/sd/chmodi/cosmo4d/'
proj = '/project/projectdirs/astro250/chmodi/cosmo4d/'

def relu(x):
    xx = x.copy()
    mask=x>0
    xx[~mask] = 0
    return xx


def elu(x, alpha=1):
    xx = x.copy()
    mask=x>0
    xx[~mask] = alpha*(np.exp(x[~mask]) - 1)
    return xx

def sigmoid(x, t=0, w=1):
    return 1/(1+np.exp(-w*(x-t)))

def linear(x):
    return x.copy()


def readfiles(pm, dpath, R1, R2, abund = True, quad=False, z=0, shear=False, doexp=False, mexp=None, cc=None, stellar=False, Rmore=None, verbose=True):
    '''Read halo and matter file to create features for net.
    The path should have folder dynamic/1 in it. 
    '''
    #Matter
    bs, nc = pm.BoxSize[0], pm.Nmesh[0]
    from nbodykit.lab import CurrentMPIComm
    CurrentMPIComm.set(pm.comm) #WORKAROUND because BigFileCatalog ignores comm o/w. Should not need after nbodykit v0.3.4
    dyn = BigFileCatalog(dpath + 'dynamic/1/', header = 'Header', comm=pm.comm)
    ppos = dyn['Position'].compute()
    final = pm.paint(ppos)
    dmdict = smoothft(final, R1, R2, quad=quad, shear=shear, pm=pm, Rmore=Rmore)

    #Halos
    if verbose: print('Read Halo files')
    hdict, halos = gridhalos(pm, dpath=dpath, R1=R1, R2=R2, abund=abund, z=z, doexp=doexp, mexp=mexp, cc=cc, stellar=stellar, verbose=verbose)

    meshdict ={}
    meshdict.update(dmdict)
    meshdict.update(hdict)    
    return meshdict, halos



def smoothft(mesh, R1=None, R2=None, quad=False, shear=False, pm=None, Rmore=None):
    '''Currently does the following to the mesh-
    Deconvolve -> Subtract mean -> Smooth at given scales & GD
    '''
    pmdc = ft.decic(mesh)
    meshdict = {'final':mesh.value, 'decic':pmdc.value}
    mean = pmdc[...].mean()
    pmdc[...] -= mean
    pmdc[...] /= mean

    dk = pmdc.r2c()

    #Create features and parameters
    if R1 is not None:
        R1mesh = ft.smooth(dk, R1, 'fingauss')
        meshdict['R1'] = R1mesh.value
    if R2 is not None:
        R2mesh = ft.smooth(dk, R2, 'fingauss')
        meshdict['R2'] = R2mesh.value
    if R2 is not None and R2 is not None:
        R12mesh = R1mesh - R2mesh
        meshdict['R12'] = R12mesh.value

    if shear: 
        s2 = ft.shear(pm, mesh)
        meshdict['shear'] = s2.value
    if Rmore is not None:
        for i, R in enumerate(Rmore):
            Rmesh = ft.smooth(dk, R, 'fingauss')
            meshdict['Rmore%d'%i] = Rmesh.value
            
    if quad:
        meshdict['fR1'] = (R1mesh*mesh).value
        meshdict['fR2'] = (R2mesh*mesh).value
        meshdict['R1R2'] = (R1mesh*R2mesh).value

    return meshdict
    

def gridhalos(pm, dpath=None, pos=None, mass=None, R1=None, R2=None, rank=None, abund = True, sigma = None, seed = 123, pmesh = True, z = 0,  
              doexp=False, mexp=None, cc=None, stellar=False, verbose=True):
    '''Read halo file or pos/mas arrays and grid and smooth them after scattering or matching abundance or both
    The path should have folder FOF in it. 
    '''
    #Matter
    bs, nc = pm.BoxSize[0], pm.Nmesh[0]

    #Halos
    if dpath is not None:
        from nbodykit.lab import CurrentMPIComm
        CurrentMPIComm.set(pm.comm) #WORKAROUND because BigFileCatalog ignores comm o/w. Should not need after nbodykit v0.3.4
        fofcat = BigFileCatalog(dpath + 'FOF', header='Header', comm=pm.comm)

        halopos = fofcat['PeakPosition'].compute()[1:]
        halomass = (fofcat['Mass'].compute()*10**10)[1:]
        try:
            hvel = fofcat['CMVelocity'].compute()[1:]
        except Exception as e: 
            print(e)
            print('Cannot read velocity')
            hvel = None

        if verbose: print('BigFileCatalog read')
    elif pos is not None:
        halopos = pos
        if mass is not None:
            halomass = mass
        else:
            print('No halo masses given, so mass=1. Scatter and abundance not valid')
            halomass = np.ones(halopos.shape[0])
            sigma = None
            abund = False
    else:
        print('Need either path of Bigfile, or catalogs')
        return None

    if abund:
        if verbose: print('Match abundance')
        halomass = mf.icdf_sampling(hmass = halomass, bs = bs, z = z)

    if stellar:
        halomass = mf.stellar_mass(halomass)

    if sigma is not None:
        print('Scatter catalog with sigma = %0.2f'%sigma)
        halomass, halopos = dg.scatter_catalog(halomass, halopos, sigma, seed)


    if doexp:
        if verbose: print('Doing doexp with mexp = %0.2f, cc=%0.2f'%(mexp, cc))
        halomass = mf.fmexp(halomass, mexp, cc)
    #print('In gridhalos')
    #print(rank, halomass.size)
    #print(dpath)
    if rank is not None:
        if rank > halomass.size:
            print('\nCatalog not big enough to be able to match abundance')
            print('\nDesired rank = %d, catalog size = %d'%(rank, halomass.size))
            return None
        else:
            halopos = halopos[:rank+1]
            halomass = halomass[:rank+1]
            hvel = hvel[:rank+1]
        
    halos = {'position':halopos, 'mass':halomass, 'velocity':hvel}
    if pmesh:
        halomesh = pm.paint(halopos, mass = halomass)
        hposmesh = pm.paint(halopos)
        meshdict = {'hposmesh':hposmesh.value, 'halomesh':halomesh.value}

        dkh = halomesh.r2c()        

        if R1 is not None:
            hmR1mesh = ft.smooth(dkh, R1, 'fingauss')
            meshdict['hmR1mesh'] = hmR1mesh.value
        if R2 is not None:
            hmR2mesh = ft.smooth(dkh, R2, 'fingauss')
            meshdict['hmR2mesh'] = hmR2mesh.value
    else:
        meshdict  = None
    return meshdict, halos




def populategalaxies(halos, zz=0, hodparams=None, cosmo=None, seedhod=43):
    '''Read halo file or pos/mas arrays and grid and smooth them after scattering or matching abundance or both
    The path should have folder FOF in it. 
    '''

    if hodparams is None: hodparams = {'alpha':0.775668, 'logMmin':13.053998, 'logM1':14.3, 'logM0':13.6176, 'sigma_logM':0.397894}
    if cosmo is None: cosmo = cosmology.Cosmology(Omega_b = 0.049, Omega_cdm = 0.2685, h=0.6711)

    #Matter
    halos['Position'] = halos['CMPosition']

    halos['Velocity'] = halos['CMVelocity']
    if halos['Velocity'].compute().max() < 1e2:
        halos['Velocity'] = halos['Velocity']*100

    if halos['Mass'].compute().max() < 1e8:
        halos['Mass'] = halos['Mass'] * 1e10

    # create the halo catalog
    halocat = HaloCatalog(halos, cosmo=cosmo, redshift=zz, mdef='vir') #mdef can be m200,c200,vir
    
    # populate halos with galaxies
    galcat = halocat.populate(Zheng07Model, seed=seedhod, **hodparams, BoxSize=halos.attrs['BoxSize']) 

    print('Satellite fraction  = ', galcat.attrs['fsat'])
    return galcat


def pdfcenzheng(m, params): 
    return 0.5*(1 + erf((np.log10(m) - params['logMmin'])/params['sigma_logM']))

def pdfsatzheng(m, params, cenocc=False):
    try:
        if (m > 10**params['logM0']): return 0 
        else: 
            toret =  ((m - 10**params['logM0'])/10**params['logM1'])**params['alpha']
    except ValueError:
        mask = (m > 10**params['logM0'])
        toret = np.zeros_like(m)
        toret[mask] = ((m[mask] - 10**params['logM0'])/10**params['logM1'])**params['alpha']

    if cenocc : return pdfcen(m)*toret
    else: return toret



#### Position Specific

def readwts(ppath):
    with h5py.File(ppath, 'r') as f:
        pwts, pbias = [], []
        for i in [0, 1, 2, 3, 4]: #pytorch has 0, 1, 2. Keras has 0, 2, 4
            try: 
                pwts.append(np.array(f['w%d'%i]))
                pbias.append(np.array(f['b%d'%i]))
            except:
                continue
        #pwts = [np.array(f['w0']), np.array(f['w2']), np.array(f['w4'])]
        #pbias = [np.array(f['b0']), np.array(f['b2']), np.array(f['b4'])]
        pmx, psx = np.array(f['mx']), np.array(f['sx'])
        try:
            pmy, psy = np.array(f['my']), np.array(f['sy'])
            return pwts, pbias, pmx, psx, pmy, psy
        except:
            return pwts, pbias, pmx, psx

def setuppos2(ifolder):
    '''returns (pwts, pbias, pacts, pmx, psx), ftname, local, pdict
    '''
    pwts, pbias, pmx, psx = readwts(ifolder + '/pos.hd5')
    with open(ifolder + '/pinfo.json') as fp: pdict = json.load(fp)
    R1 = pdict['R1']
    try: R2 = pdict['R2']
    except:
        sfac = pdict['sfac']
        R2 = R1 * sfac
    
    ftname = pdict['pftname']
    try: 
        acts = pdict['activations']
        acts = ['sigmoid' if 'sigmoid' in s else s for s in acts]
    except: acts = ['relu', 'relu', 'sigmoid'] #Pytorch did not write activations
    width = pdict['width']
    local = pdict['plocal']
    #neurons = pdict['neurons']

    actdict = {'relu':relu, 'elu':elu, 'linear':linear, 'sigmoid':partial(sigmoid, t=0,  w=width)}
    pacts = [actdict[i] for i in acts]
    arch = tuple(zip([i.shape[1] for i in pwts], acts))
    info = 'Network architecture for position: '
    for i in arch:
        info += '%d, %s; '%i
    print(info)
    print('Width of sigmoid is = ', width)
    print('Position features are ', ftname)
    return (pwts, pbias, pacts, pmx, psx), ftname, local, pdict


#def setuppostorch(ifolder):
#    '''returns (pwts, pbias, pacts, pmx, psx), ftname, local, pdict
#    '''
#    pwts, pbias, pmx, psx = readwts(ifolder + '/pos.hd5')
#    with open(ifolder + '/pinfo.json') as fp: pdict = json.load(fp)
#    R1 = pdict['R1']
#    try: R2 = pdict['R2']
#    except:
#        sfac = pdict['sfac']
#        R2 = R1 * sfac
#    
#    ftname = pdict['pftname']
#    acts = ['relu', 'relu', 'sigmoid']
#    width = pdict['width']
#    local = pdict['plocal']
#
#    actdict = {'relu':relu, 'elu':elu, 'linear':linear, 'sigmoid':partial(sigmoid, t=0,  w=width)}
#    pacts = [actdict[i] for i in acts]
#    arch = tuple(zip([i.shape[1] for i in pwts], acts))
#    info = 'Network architecture for position: '
#    for i in arch:
#        info += '%d, %s; '%i
#    print(info)
#    print('Width of sigmoid is = ', width)
#    print('Position features are ', ftname)
#    return (pwts, pbias, pacts, pmx, psx), ftname, local, pdict
#

def setupmass(ifolder):
    '''returns (pwts, pbias, pacts, pmx, psx), ftname, local, pdict
    '''
    pwts, pbias, pmx, psx, pmy, psy = readwts(ifolder + '/mass.hd5')
    with open(ifolder + '/minfo.json') as fp: mdict = json.load(fp)
    ftname = mdict['mftname']
    try: acts = mdict['activations']
    except : 
        print('WARNING: ASSUMING ELU')
        acts = ['elu', 'elu', 'linear']
    actdict = {'relu':relu, 'elu':elu, 'linear':linear, 'sigmoid':partial(sigmoid, t=0,  w=0)}
    macts = [actdict[i] for i in acts]
    arch = tuple(zip([i.shape[1] for i in pwts], acts))
    info = 'Network architecture for position: '
    for i in arch:
        info += '%d, %s; '%i
    print(info)
    local = True
    return (pwts, pbias, macts, pmx, psx, pmy, psy), ftname, local, mdict



def applynet(tt, tup):
    try:
        wts, bias, act, mx, sx = tup
        my, sy = None, None
    except:
        wts, bias, act, mx, sx, my, sy = tup
    ttn = tt - mx
    ttn /= sx
    ly = ttn.copy()
    if len(wts) != len(bias) or len(wts) != len(act):
        print('Inconsistent network in nettools.applynet')
        print('Length weigts = %d'%len(wts))
        print('Length bias = %d'%len(bias))
        print('Length activations = %d'%len(act))
        import sys
        sys.exit()

    for i in range(len(wts)):
        try: ly = np.dot(ly, wts[i]) + bias[i]
        except: ly = np.dot(ly, wts[i].T) + bias[i].T #Pytroch has trasnposed weights of Keras
        #print(i, ly.max(), ly.min())
        ly = act[i](ly)

    if my is not None:
        ly *= sy
        ly += my
    return ly



def setupmassreg(ifolder):
    '''returns (pwts, pbias, pacts, pmx, psx), ftname, local, pdict
    '''
    with open(ifolder + '/minfo.json') as fp: mdict = json.load(fp)
    
    mftname = mdict['mftname']
    mcoef = mdict['coeff'][0]
    massmin = mdict['massmin']
    return (massmin, mcoef), mftname, True, mdict



def applymassreg(tt, tup):
    '''returns (pwts, pbias, pacts, pmx, psx), ftname, local, pdict
    '''
    
    massmin, mcoef = tup
    ones = np.ones_like(tt[:, 0]).reshape(-1, 1)
    tt = np.concatenate((ones, tt), axis=-1)
    Nf = tt.shape[1]
    counter = 0
    mpredict = np.zeros_like(tt[:, 0])
    for i in range(Nf):
        l1 = tt[:, i]
        for j in range(i, Nf):
            l2 = tt[:, j]
            tmp = tt[:, i]*tt[:, j]*mcoef[counter]
            mpredict += tmp
            counter += 1 

    mpredict *= massmin
    mpredict = relu(mpredict)
    return mpredict

def testdata(pm, bs, nc, nsteps, seeds, R1, R2, ftname, local=False, rethalos = False, abund=True, quad=False, z=0, shear=False, 
             doexp=False, mexp=None, cc=None, stellar=False, Rmore=None):
    '''Call create data for different seeds and concatenate them
    '''
    ndim = len(ftname)
    if local:
        fac = 1
    else:
        fac = 27
    tt = np.zeros([1, fac*ndim])
    for seed in seeds:
        print('Read testdata for seed  = ', seed)
        dpath = proj + 'data/z%02d/L%04d_N%04d_S%04d_%02dstep/'%(z*10, bs, nc, seed, nsteps)

        meshdict, halos = readfiles(pm, dpath, R1, R2, abund=abund, quad=quad, z=z, shear=shear, doexp=doexp, mexp=mexp, cc=cc, stellar=stellar, Rmore=Rmore)
        print('Features are - ', ftname)
        ftt = createdata(pm, meshdict, ftname, local)
        tt = np.concatenate([tt, ftt])
    tt = tt[1:]

    if rethalos:
        return meshdict, halos, tt
    else:
        return meshdict, tt


def createdata(pm, meshdict, ftname, local=False):
    '''Create testing data, basically the last part of balanced27gridpt
    '''
    ndim = len(ftname)
    if local:
        fac = 1
    else:
        fac = 27
    ftlist = [meshdict[key] for key in ftname]

    ftt = np.zeros([pm.Nmesh[0]**3, fac*ndim])
    for boo in range(ndim):
        if fac == 27:
            ftt[:, 27*boo: 27*boo + 27] = datalib.read27(ftlist[boo], pm=pm)
        else:
            ftt[:, boo] = ftlist[boo].flatten()
    return ftt


##

##def setuppos(ifolder):
##    pwts, pbias, pmx, psx = readwts(ifolder + 'pos.hd5')
##    ifile = open(ifolder + 'info.log')
##    for line in ifile:
##        if 'Features' in line:
##            lf = line
##        if 'architecture' in line:
##            ln = line
##    ifile.close()
##    #
##    ftname = lf.split("'")[1::2]
##    acts = []
##    for s in ['relu', 'sigmoid', 'linear']:
##        acts +=[(s, k.span()[0]) for k in re.finditer(s, ln)]    
##    acts.sort(key = lambda tup: tup[1])
##    acts = [i[0] for i in acts]
##    try:
##        width = [int(ln[k.span()[0]-1]) for k in re.finditer('x', ln)][0]
##    except:
##        width = 1
##    thresh = 0
##    actdict = {'relu':relu, 'linear':linear, 'sigmoid':partial(sigmoid, t=thresh, w=width)}
##    pacts = [actdict[i] for i in acts]
##    #
##    if pwts[0].shape[0] % 27:
##        plocal = True
##    else:
##        plocal = False
##    #
##    arch = tuple(zip([i.shape[1] for i in pwts], acts))
##    info = 'Network architecture for position: '
##    for i in arch:
##        info += '%d, %s; '%i
##    print(info)
##    print('Sigmoid parameters are t, w = ', thresh, width)
##    print('Position features are ', ftname)
##    return (pwts, pbias, pacts, pmx, psx), ftname, plocal
##
