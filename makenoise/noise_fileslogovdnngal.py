#Make noise histograms for difference in overdensity

import numpy as np
import matplotlib.pyplot as plt
from pmesh.particlemesh import ParticleMesh
from pmesh.pm import ParticleMesh as PMnew
from nbodykit.lab import BigFileCatalog, BigFileMesh, FFTPower
import yaml

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

#scratch = '/global/cscratch1/sd/chmodi/cosmo4d/'
scratch = '/project/projectdirs/astro250/chmodi/cosmo4d/'
train = scratch + 'train/'
data = scratch + 'data/'

pfile = "/global/homes/c/chmodi/Data/ics_matterpow_0.dat"
klin, plin = np.loadtxt(pfile, unpack=True)
mf = mass_function.Mass_Func(pfile, 0.3175)

from cosmo4d import report as dgrep
import yaml

########


with open(scratch + 'train/models.yaml', 'r') as ymlfile:
    cfg = yaml.load(ymlfile)


bs, nc, numd, seed = 400, 128, 5e-4, 800
zz = 0.0

abund = False
Rsm = 3


num = int(numd*bs**3)
sfine = 4
pm = PMnew(BoxSize=bs, Nmesh=[nc]*3)


ptpath = train + cfg['%s-%s'%(bs,nc)][zz][numd]['ppath']
mtpath  = ptpath + cfg['%s-%s'%(bs,nc)][zz][numd]['gpath']
print('Loaded and saved in \n%s'%mtpath)


ptup, pftname, plocal, pdict = ntools.setuppos2(ptpath)
mtup, mftname, mlocal, mdict = ntools.setupmass(mtpath )
R1 = pdict['R1']
try : R2 = pdict['R2']
except :
    sfac = pdict['sfac']
    R2 = R1*sfac


#model
print('Reading Files')
meshdict, dummy = ntools.readfiles(pm, scratch + '/data/z%02d/L%04d_N%04d_S%04d_05step/'%(zz*10, bs, nc, seed), R1=R1, R2=R2, abund=abund)

ftt = ntools.createdata(pm, meshdict, pdict['pftname'], plocal)
mftt = ntools.createdata(pm, meshdict, mftname, mlocal)
nnpred = ntools.applynet(ftt, ptup).reshape(nc, nc, nc)
nnmass = ntools.applynet(mftt, mtup).reshape(nc, nc, nc)
predict = pm.create(mode ='real', value=nnpred*nnmass)
#predictR = ft.smooth(predict, Rsm, 'fingauss')
predictr = ntools.relu(predict[...])
predictR = ft.smooth(predict, Rsm, 'fingauss')

#data

print('Generating data')

#hdictf = ntools.gridhalos(pm, dpath=scratch +'/data/z%02d/L%04d_N%04d_S%04d_40step/'%(zz*10, bs, sfine*nc, seed), R1=R1, R2=R2, pmesh=False, abund=abund)[1]
#datap = pm.paint(hdictf['position'][:num], mass = hdictf['mass'][:num])
#datapR = ft.smooth(datap, Rsm, 'fingauss')
dpath = scratch +'/data/z%02d/L%04d_N%04d_S%04d_40step/'%(zz*10, bs, sfine*nc, seed)
galcat = BigFileCatalog(dpath + 'galaxies_n%02d/galcat'%(numd*1e4), header='Header', comm=pm.comm)
datap = pm.paint(galcat['Position'], mass = galcat['Mass'])
datapR = ft.smooth(datap, Rsm, 'fingauss')

print('Data generated')

func = dg.normal
#func = dg.laplace
colors = ['r', 'b', 'g', 'y', 'm', 'orange', 'brown', 'k']


###########################################

#mbins = np.logspace(10, 13, 16)[::-1]
#msave = [mbins[0]*100] + list(mbins)
#print('mbins -- ', mbins)

####

mms = [1e6, 1e7, 1e8, 1e9, 1e10]
for ii, M0 in enumerate(mms[::-1]):
    print('For M0 = %0.2e'%M0)

    #mbins = np.linspace(np.log10(M0)-1, min(10, np.log10(M0) + 2), 16)[::-1]    
    mbins = np.logspace(7, 10, 16)[::-1]    
    #mbins = 10**mbins
    msave = [mbins[0]*100] + list(mbins)
    print(mbins)
    normed=False

    normp, normd = predictR.cmean(), datapR.cmean()
    scatter = dg.gridscatter(np.log((datapR[...]+M0)/(normd)), np.log((predictR[...]+M0)/(normp)), mbins, datapR[...])
    #scatter = dg.gridscatter(datapR[...]/(datapR[...].mean()+M0), predictR[...]/(predictR[...].mean()+M0), mbins, datapR[...])
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

    ff = mtpath + 'noisehist_logovd_M%02d.png'%(10*np.log10(M0))
    if Rsm !=3 : ff = ff[:-4] + '_R%d.png'%Rsm
    fig.savefig(ff)

    fpath = mtpath + 'hist_logovd_M%02d.txt'%(10*np.log10(M0))
    if Rsm !=3 : fpath = fpath[:-4] + 'R%d.txt'%Rsm
    np.savetxt(fpath, np.array(tosave), header ='Mmax, Mmin, mean(offset: data-model), \sigma, Mass function not matched for data')


    

