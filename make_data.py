import numpy
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as interpolate
#from cosmo4d.lab import (UseComplexSpaceOptimizer,
#                        NBodyModel,
#                        LBFGS, ParticleMesh)
from cosmo4d.lab import NBodyModel#, LPTModel
from pmesh.pm import ParticleMesh

#from cosmo4d.lab import maphd as map
from cosmo4d.iotools import save_map, load_map

from nbodykit.cosmology import Planck15, EHPower
from nbodykit.algorithms.fof import FOF
from nbodykit.lab import ArrayCatalog
from nbodykit.lab import KDDensity, BigFileCatalog
import os, sys, json
sys.path.append('/global/homes/c/chmodi/Programs/cosmo4d/cosmo4d/')
sys.path.append('/global/homes/c/chmodi/Programs/cosmo4d/train_nets')
import nettools as ntools
import icdfhod_sampling as icdf
from time import time

def make_data(bs, nc, seed, nsteps, path='', z=0):
    #initiate
    pm = ParticleMesh(BoxSize=bs, Nmesh=(nc, nc, nc), dtype='f8')

    ofolder = path + 'z%02d/L%04d_N%04d_S%04d_%02dstep/'%(z*10, bs, nc, seed, nsteps)
#        
    
    if pm.comm.rank == 0:
        print('Make data for seed = %d'%seed)
        print('Data to be saved at path - %s'%ofolder)

    klin, plin = numpy.loadtxt('ics_matterpow_0.dat', unpack = True)
    pk = interpolate(klin, plin)
    cosmo = Planck15.clone(Omega_cdm = 0.2685, h = 0.6711, Omega_b = 0.049)
    #pk = EHPower(Planck15, redshift=0)
    #cosmo = Planck15

    s_truth = pm.generate_whitenoise(seed, type='complex')\
            .apply(lambda k, v: v * (pk(sum(ki **2 for ki in k) **0.5) / v.BoxSize.prod()) ** 0.5)\
            .c2r()

    #dynamics
    aa = 1.0/(1+z)
    if pm.comm.rank == 0:
        print('Evolve to redshift = %0.1f, scale factor = %0.2f'%(z, aa))
    stages = numpy.linspace(0.1, aa, nsteps, endpoint=True)

    start = time()
    dynamic_model = NBodyModel(cosmo, pm, B=2, steps=stages)
    #dynamic_model = LPTModel(cosmo, pm, B=2, steps=stages)

    #Save data

    X, V, final = dynamic_model.get_code().compute(['X', 'V', 'final'], init={'parameters':s_truth})
    end = time()
    print('Time taken = ', end-start)

    save_map(s_truth, ofolder + 'mesh', 's')
    save_map(final, ofolder + 'mesh', 'd')

    if pm.comm.rank == 0:
        print('X, V computed')
    cat = ArrayCatalog({'Position': X, 'Velocity' : V}, BoxSize=pm.BoxSize, Nmesh=pm.Nmesh)
    kdd = KDDensity(cat).density
    cat['KDDensity'] = kdd
    cat['InitPosition'] = dynamic_model.get_code().engine.q
    cat.save(ofolder + 'dynamic/1', ('InitPosition', 'Position', 'Velocity', 'KDDensity'))
    if pm.comm.rank == 0:
        print('dynamic model created')

    #FOF
    fof = FOF(cat, linking_length=0.2, nmin=12)
    fofcat = fof.find_features(peakcolumn='KDDensity')
    fofcat['Mass'] = fofcat['Length'] * cosmo.Om0 * 27.7455 * pm.BoxSize.prod() / pm.Nmesh.prod()
    fofcat.save(ofolder+'FOF', ('CMPosition', 'CMVelocity', 'PeakPosition', 'PeakVelocity', 'Length', 'Mass'))




def make_galcat(bs, nc, seed, nsteps, numd, path='', z=0, sortcen=False, sortsat=False, sigc=0.212, mstarc=1e12):
    #initiate
    pm = ParticleMesh(BoxSize=bs, Nmesh=(nc, nc, nc), dtype='f8')

    dpath = path + 'z%02d/L%04d_N%04d_S%04d_%02dstep/'%(z*10, bs, nc, seed, nsteps)
    ofolder = dpath + 'galaxies_n%02d/'%(numd*1e4)
    try: os.makedirs(ofolder)
    except: pass
    
    fofcat = BigFileCatalog(dpath + 'FOF', header='Header', comm=pm.comm)[1:int(numd*bs**3)]
    
    fofcat['Velocity'] = fofcat['CMVelocity']*100
    # halos['Position'] = halos['CMPosition']
    fofcat['Position'] = fofcat['PeakPosition']
    fofcat['Mass'] = fofcat['Mass'] * 1e10

    hodparams = {'alpha':0.775668, 'logMmin':13.053998, 'logM1':14.3, 'logM0':13.6176, 'sigma_logM':0.397894}

    with open(ofolder + 'hodparams.json', 'w') as fp:
            json.dump(hodparams, fp, sort_keys=True, indent=4)

    galcat = ntools.populategalaxies(halos = fofcat, zz=z, hodparams=hodparams)

    print('Galaxies populated')
    #
    Mthresh = icdf.mstarsat(fofcat['Mass'].compute()[-1])
    print('Mthresh = %0.2e'%Mthresh)
    nmbins = 50
    #cenpars = {sortcen=False, #EDITING HERE
    galmass = icdf.assigngalaxymass(galcat, fofcat['Mass'].compute(), nmbins=nmbins, Mthresh=Mthresh, 
                                    sortcen=sortcen, sortsat=sortsat, sigc=sigc, mstarc=mstarc)
    print('Galaxy masses assigned')

    galcat['Mass'] = galmass

    colsave = [cols for cols in galcat.columns]
    fname = 'galcat'
    if sortcen: fname+='-sortcen'
    if sortsat: fname+='-sortsat'
    if sigc !=0.212: fname += '-sigc%02d'%(sigc*100)
    print('Data saved at %s'%(ofolder + fname))
    galcat.save(ofolder+fname, colsave)




##
##
if __name__ == "__main__":

    #Set parameters here
#    proj = '/global/cscratch1/sd/chmodi/cosmo4d/data/'
    proj = '/global/cscratch1/sd/chmodi/cosmo4d/data/'
#    proj = '/global/cscratch1/sd/chmodi/cosmo4d/data/z05/'
#    proj = '/project/projectdirs/astro250/chmodi/cosmo4d/data/'
#    proj = '/project/projectdirs/m3035/chmodi/cosmo4d/data/'



##    #Set parameters here
    bs, nc = 400, 128
    nsteps = 5
    #numd = 5e-4

    #for seed in [800, 900, 1000]:
#    for seed in [800, 900, 1000, 1100, 1200]:
    for seed in np.arange(100, 1200, 100):        
        start = time()
        make_data(bs, nc, seed, nsteps, proj, z=0.0)
        end = time()
        print(end - start)

##    for l  in [[256, 40]]:
##        for bs in [200, 300]:
##            nc, nsteps = l
##            for seed in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]:
##                make_data(bs, nc, seed, nsteps, proj, z=0.5)


    #bs, nc = 300., 512
    #nsteps = 40
##    proj = '/global/cscratch1/sd/chmodi/cosmo4d/data/'
##    seed  = 600
##    for l in [[200, 64, 5], [200, 128, 40], [200, 256, 40], [200, 128, 5], [300, 64, 5], [300, 128, 5], [300, 256, 40]]:
##        bs, nc , nsteps = l
##        print("\n", l, "\n")
##        make_data(bs, nc, seed, nsteps, proj)


##    for seed in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
##        print('For seed = ', seed)
##        make_galcat(bs, nc, seed, nsteps, numd=numd, path=proj, z=0.0, sortcen=True, sortsat=True, sigc=0.2)
