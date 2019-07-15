import numpy as np
import matplotlib.pyplot as plt
from pmesh.particlemesh import ParticleMesh
from pmesh.pm import ParticleMesh as PMnew
from nbodykit.lab import BigFileCatalog, BigFileMesh, FFTPower
import yaml, json
from scipy.optimize import curve_fit as cf

import sys, os
#sys.path.append('/global/homes/c/chmodi/Programs/Py_codes/modules/')
sys.path.append('/global/homes/c/chmodi/Programs/cosmo4d/cosmo4d/')
#import mytools as tools
import mymass_function as mass_function
#import mycosmology
#import mysigmoid as sigmoid

sys.path.append("/global/homes/c/chmodi/Programs/cosmo4d/")
sys.path.append('/global/homes/c/chmodi/Programs/cosmo4d/cosmo4d/')
sys.path.append('/global/homes/c/chmodi/Programs/cosmo4d/train_nets')
import features as ft
import datalib as lib
import datatools as dtools
import nettools as ntools
import sigtools as stools
import diagnostic as dg
import icdfhod_sampling as icdf
#scratch = '/global/cscratch1/sd/chmodi/cosmo4d/'
proj = '/project/projectdirs/astro250/chmodi/cosmo4d/'
train = proj + 'train/'
data = proj + 'data/'

pfile = "/global/homes/c/chmodi/Data/ics_matterpow_0.dat"
klin, plin = np.loadtxt(pfile, unpack=True)
mf = mass_function.Mass_Func(pfile, 0.3175)

from cosmo4d import report as dgrep
import yaml

########


with open(proj + 'train/models.yaml', 'r') as ymlfile:
    cfg = yaml.load(ymlfile)


bs, nc, numd, seed = 400, 128, 5e-4, 100
zz = 0.0

abund = False
doexp = False
mexp, cc = None, None
stellar = False
regression = False
Rsm = 3
truemass = False  ####truemass for the sactter i.e. use correct mass but only scatter the positions via abundance matching 

num = int(numd*bs**3)
sfine = 4
pm = PMnew(BoxSize=bs, Nmesh=[nc]*3)


ptpath = train + cfg['%s-%s'%(bs,nc)][zz][numd]['ppath']
mtpath  = ptpath + cfg['%s-%s'%(bs,nc)][zz][numd]['gpath']
print('Loaded and saved in \n%s'%mtpath)

ptup, pftname, plocal, pdict = ntools.setuppos2(ptpath)
if regression:
    mtup, mftname, mlocal, mdict = ntools.setupmassreg(mtpath )
else:
    mtup, mftname, mlocal, mdict = ntools.setupmass(mtpath )
R1 = pdict['R1']
try : R2 = pdict['R2']
except :
    sfac = pdict['sfac']
    R2 = R1*sfac


#model
print('Reading Files')
meshdict, dummy = ntools.readfiles(pm, proj + '/data/z%02d/L%04d_N%04d_S%04d_05step/'%(zz*10, bs, nc, seed), R1=R1, R2=R2, abund=abund,
                                   doexp=doexp, mexp=mexp, cc=cc, stellar=stellar)
ftt = ntools.createdata(pm, meshdict, pdict['pftname'], plocal)
mftt = ntools.createdata(pm, meshdict, mftname, mlocal)
nnpred = ntools.applynet(ftt, ptup).reshape(nc, nc, nc)
nnmass = ntools.applynet(mftt, mtup).reshape(nc, nc, nc)
predict = pm.create(mode ='real')
predict[...] = nnpred*nnmass
predictr = ntools.relu(predict[...])
predictR = ft.smooth(predict, Rsm, 'fingauss')


#data

print('Generating data')
dpath = proj +'/data/z%02d/L%04d_N%04d_S%04d_40step/'%(zz*10, bs, sfine*nc, seed)
galcat = BigFileCatalog(dpath + 'galaxies_n%02d/galcat'%(numd*1e4), header='Header', comm=pm.comm)
datap = pm.paint(galcat['Position'], mass = galcat['Mass'])
datapR = ft.smooth(datap, Rsm, 'fingauss')

with open(dpath + 'galaxies_n%02d/hodparams.json'%(numd*1e4)) as fp: hodparams = json.load(fp)
#hodparams = {'alpha':0.775668, 'logMmin':13.053998, 'logM1':14.3, 'logM0':13.6176, 'sigma_logM':0.397894}
with open(mtpath + 'hodparams.json', 'w') as fp:
        json.dump(hodparams, fp, sort_keys=True, indent=4)

print('Data generated')

#########
#Get the prediction for galaxy mass
#
#
#def mhtomstar2(mh, satfunc=False):
#    tmpmc = icdf.mstarcen(mh)
#    if satfunc: tmpms = icdf.mstarsat(mh)
#    else: tmpms = tmpmc/10**0.25
##     else: tmpms = tmpmc
#    return tmpmc + tmpms*ntools.pdfsatzheng(mh, hodparams)    
#
#def fitaxb(x, a, b, c, d):
#    xx = np.log(x)
#    return (a*xx**3 + b*xx**2 + c*xx + d)
#
#mm = np.logspace(12, 15.5)
#fitparams = cf(fitaxb, mm, np.log(mhtomstar2(mm)))[0] #fit to return natural log of mass
#np.savetxt(mtpath + 'galmassparams.txt', fitparams, header='galmass=exp(a*xx**3 + b*xx**2 + c*xx + d), xx=log(M)')
#
#print('Mass parameters are - ', fitparams)    
#
#nncenmask = nnpred*ntools.sigmoid(np.log10(ntools.relu(nnmass) + 1), t=hodparams['logMmin'])
#predict = pm.create(mode='real', value=nncenmask*np.exp(fitaxb(1+predictr, *fitparams)))
#predictR = ft.smooth(predict, Rsm, 'fingauss')
#
#########


bins = np.linspace(-3, 3, 300)
func = dg.normal
colors = ['r', 'b', 'g', 'y', 'm', 'orange', 'brown', 'k']

	
mms = [1e7, 1e8, 1e9, 1e10]
       
for M0 in mms:
    print('For M0 = %0.2e'%M0)

    mbinsm = np.linspace(max(7, np.log10(M0)-2), min(10, np.log10(M0) + 3), 16)[::-1]
    
    mbinsm = 10**mbinsm
    msave = [mbinsm[0]*100] + list(mbinsm)

#    bins = np.zeros((mbinsm.size, 1000 ))
#    for i in range(0, bins.shape[0]):
#        bins[i] = np.linspace(-3*mbinsm[i]/M0, 2*mbinsm[i]/M0, bins.shape[1])
#
    print('mbinsm -- ', mbinsm)

    
    fig, ax = plt.subplots(4, 4, figsize = (15, 15))
    fit0 = dg.plot_noise(datapR.value, predictR.value, M0=M0, binfit=bins, c='k', axin=ax, func=func, mbin=mbinsm, retfit=True, lsf='--')[0]
    
    fits = []

    ff = mtpath + 'noisehist_M%02d.png'%(10*np.log10(M0))
    if truemass: ff = ff[:-4] + 'trueM.png'
    if Rsm !=3 : ff = ff[:-4] + '_R%d.png'%Rsm
    fig.savefig(ff)

    tosave = []
    for i, res in enumerate(fit0):
        #print(res)
        tosave.append([msave[i], msave[i+1], res.x[1], res.x[2]])

    fpath = mtpath + 'hist_M%02d.txt'%(10*np.log10(M0))
    if truemass: fpath = fpath[:-4] + 'trueM.txt'
    if Rsm !=3 : fpath = fpath[:-4] + 'R%d.txt'%Rsm
    header = 'hodparams=%s\nMmax, Mmin, mean(offset: data-model), b, Mass function not matched for data'%(str(hodparams))
    np.savetxt(fpath, np.array(tosave), header = header)




