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
if regression:
    mtpath  = ptpath + cfg['%s-%s'%(bs,nc)][zz][numd]['rpath']
else:
    mtpath  = ptpath + cfg['%s-%s'%(bs,nc)][zz][numd]['mpath']
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
#predictR = ft.smooth(predict, Rsm, 'fingauss')


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


def mhtomstar2(mh, satfunc=False):
    tmpmc = icdf.mstarcen(mh)
    if satfunc: tmpms = icdf.mstarsat(mh)
    else: tmpms = tmpmc/10**0.25
#     else: tmpms = tmpmc
    return tmpmc + tmpms*ntools.pdfsatzheng(mh, hodparams)    

def fitaxb(x, a, b, c, d):
    xx = np.log(x)
    return (a*xx**3 + b*xx**2 + c*xx + d)

mm = np.logspace(12, 15.5)
fitparams = cf(fitaxb, mm, np.log(mhtomstar2(mm)))[0] #fit to return natural log of mass
np.savetxt(mtpath + 'galmassparams.txt', fitparams, header='galmass=exp(a*xx**3 + b*xx**2 + c*xx + d), xx=log(M)')

print('Mass parameters are - ', fitparams)    

nncenmask = nnpred*ntools.sigmoid(np.log10(ntools.relu(nnmass) + 1), t=hodparams['logMmin'])
predict = pm.create(mode='real', value=nncenmask*np.exp(fitaxb(1+predictr, *fitparams)))
predictR = ft.smooth(predict, Rsm, 'fingauss')

#########

func = dg.normal
colors = ['r', 'b', 'g', 'y', 'm', 'orange', 'brown', 'k']
normed = True
	
mms = [1e6, 1e7, 1e8, 1e9, 1e10]
       
for M0 in mms:
    print('For M0 = %0.2e'%M0)

    #mbins = np.linspace(np.log10(M0)-1, min(10, np.log10(M0) + 4), 10)[::-1]
    #mbins = 10**mbins
    
    mbins = np.logspace(7, 10, 16)[::-1]    
    msave = [mbins[0]*100] + list(mbins)


    scatter = dg.gridscatter(datapR[...]/(datapR[...].mean()+M0), predictR[...]/(predictR[...].mean()+M0), mbins, datapR[...])
    tosave = []
    #bins = np.linspace(-0.5, 0.7)
    fig, ax = plt.subplots(4, 4, figsize = (15, 15))

    for i in range(16):
        #Asymmettric
        mean, std = (scatter[i][1] - scatter[i][0]).mean(), (scatter[i][1] - scatter[i][0]).std()
        bins = np.linspace(mean - 3*std, mean + 3*std)
        #bins = np.linspace(-(4*ii+5)/(i+1), (3*ii+5)/(i+1))
        axis = ax.flatten()[i]
        axis.hist(scatter[i][1] - scatter[i][0], histtype='step', bins=bins, normed=normed, color='C%d'%(i%9), label='%0.2e'%mbins[i])
        if normed: x0 = [1/(bins[-1]-bins[0]), mean, std]
            #if normed: x0 = [scatter[i][0].size, mean, std]
        else: x0 = [scatter[i][0].size, mean, std]
        #if normed:  x0 = [10, 0, 1]
        #else: x0 = [scatter[i][1].size/10., 0, 1]
        xx, yy, res = dg.fitpdf(scatter[i][1] - scatter[i][0], func, bins=bins, normed=normed, x0=x0)
        axis.plot(xx, func(xx, *res.x), 'k--', label='%0.2f, %0.2f'%(res.x[1], res.x[2]))
        tosave.append([msave[i], msave[i+1], res.x[1], res.x[2]])
        axis.legend()

    ff = mtpath + 'noisehist_ovd_M%02d_gal.png'%(10*np.log10(M0))
    if Rsm !=3 : ff = ff[:-4] + '_R%d.png'%Rsm
    fig.savefig(ff)

    fpath = mtpath + 'hist_ovd_M%02d_gal.txt'%(10*np.log10(M0))
    if Rsm !=3 : fpath = fpath[:-4] + 'R%d.txt'%Rsm
    header = 'hodparams=%s\nMmax, Mmin, mean(offset: data-model), b, Mass function not matched for data'%(str(hodparams))
    np.savetxt(fpath, np.array(tosave), header = header)





##fofcat = BigFileCatalog(dpath + 'FOF', header='Header', comm=pm.comm)[1:int(numd*bs**3)]
##hpos = fofcat['PeakPosition'].compute()
##hmass = (fofcat['Mass'].compute()*10**10)
##hodparams = {'alpha':0.775668, 'logMmin':13.053998, 'logM1':14.3, 'logM0':13.6176, 'sigma_logM':0.397894}
##with open(mtpath + 'hodparams.json', 'w') as fp:
##        json.dump(hodparams, fp, sort_keys=True, indent=4)
##
##galcat = ntools.populategalaxies(halos = fofcat, zz=zz, hodparams=hodparams)
##
##print('Galaxies populated')
###
##Mthresh = icdf.mstarsat(hmass[-1])
##print('Mthresh = %0.2e'%Mthresh)
##nmbins = 50
##galmass = icdf.assigngalaxymass(galcat, hmass, nmbins=nmbins, Mthresh=Mthresh, sortcen=False, sortsat=False)
##print('Galaxy masses assigned')
##
