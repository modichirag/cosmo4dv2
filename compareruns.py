import sys
sys.path.append('/global/u1/c/chmodi/Programs/cosmo4d/notebooks/')
from lab import *

ptup, fitp, bestp, datap, report, reportf = {}, {}, {}, {}, {}, {}


nsteps, seed = 5, 100
numd = 5e-4
numd1 = 1e-3 

ptup['200'] = (200, 64, nsteps, seed, numd*1e4)
ptup['300-64'] = (300, 64,  nsteps, seed, numd*1e4)
ptup['200-128'] = (200, 128, nsteps, seed, numd*1e4)
ptup['300'] = (300, 128, nsteps, seed, numd*1e4)
ptup['3001'] = (300, 128, nsteps, seed, numd1*1e4)
ptup['400'] = (400, 128, nsteps, seed, numd*1e4)
ptup['4001'] = (400, 128, nsteps, seed, numd1*1e4)
ptup['500'] = (500, 128, nsteps, seed, numd*1e4)
ptup['5001'] = (500, 128, nsteps, seed, numd1*1e4)

outpath = scratch + 'output/L%04d_N%04d_%02dstep-fof/lhd_S%04d/n%02d/version2/'
outpathold = scratch + 'output/L%04d_N%04d_%02dstep-fof/lhd_S%04d/n%02d/'
# #400
bs, nc = 400, 128

def loadfile(key, folder, ipath, subf = 'best-fit', mesh=False):
    try:
        print('Loading ', ipath)
        bestm = mapp.Observable.load(ipath + '%s/%s'%(folder, subf))
        datam = mapp.Observable.load(ipath + 'datap')
        fitm = mapp.Observable.load(ipath + 'fitp').mapp
        report[key] = dgrep.evaluate(bestm, datam)[:-2]
        reportf[key] = dgrep.evaluate1(fitm, datam.mapp)
        if mesh:
            bestp[key], datap[key], fitp[key] = bestm, datam, fitm
    except:
        print('\nFile does not exist, skipped!\n%s\n'%(ipath + '%s/%s'%(folder, subf)))

#########################################################



def plotsum(keys, l=400, ls='-', lw=1, alpha=1, fname='tmprep.pdf', suptitle=None):
    
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    
    for i, key in enumerate(keys):
        try:
            dgrep.summary(report['%d%s'%(l, key)], fig, ax, c='C%d'%(i%8), lw=lw, ls =ls, alpha=alpha, label='%s'%key)
        except:
            print ('\n%s not found and hence skipped'%key)


    for axis in ax.flatten():
        axis.set_xlim(3e-2, 0.5)
        axis.axhline(1, lw=0.5, color ='k')
        axis.grid(which='both', lw=0.5)

    for axis in ax[0, :]: axis.set_ylim(0.5, 1.05)
    for axis in ax[1, :]: axis.set_ylim(0.5, 1.5)
        
    ax[0, 2].legend(loc=3)
    ax[0, 1].legend_.remove()

    if suptitle is not None: fig.suptitle(suptitle, fontsize =14)
    
    fig.savefig('./figs/%s'%fname)


#########################################################

print('Loading files now')


###Log-loss-Halos
#
#loadfile('4001log-v06', 'M0_120/128-03-4.00', ipath=outpath%ptup['4001']+ 'opt_s999_iM12-sm3v06off/')
#loadfile('4001log-v60-n3dmlim120', 'M0_120/128-03-4.00', ipath=outpath%ptup['4001']+ 'opt_s999_iM12-sm3v06n3doffmlimn-120/')
#loadfile('4001log-v06-n3dmlim120-N1', 'M0_120/128-01-1.00', ipath=outpath%ptup['4001']+ 'opt_s999_iM12-sm3v06n3doffmlimn-120/')
loadfile('4001log-v06-n3dmlim120-N0', 'M0_120/128-01-0.00-init60', ipath=outpath%ptup['4001']+ 'opt_s999_iM12-sm3v06n3doffmlimn-120/')

loadfile('4001log-v25-fin', 'M0_12/128-03-0.00', ipath=outpathold%ptup['4001']+ 'opt_s999_MdowniM12-nasm3off/')
#loadfile('4001log-v25', 'M0_120/128-03-4.00', ipath=outpath%ptup['4001']+ 'opt_s999_iM12-sm3v25off/')
#loadfile('4001log-v25-mlim60', 'M0_120/128-03-4.00', ipath=outpath%ptup['4001']+ 'opt_s999_iM12-sm3v25n3doffmlimn-60/')

###M11
#
##loadfile('4001m11v20', 'M0_110/128-03-1.00', ipath=outpath%ptup['4001']+ 'opt_s999_iM12-sm3v20n3doffmlimn-69-restart/', subf='0060/fit_p/')
##loadfile('4001m11v60', 'M0_110/128-03-1.00', ipath=outpath%ptup['4001']+ 'opt_s999_iM12-sm3v06n3doffmlimn-120/', subf='0060/fit_p/')

##Ovd-loss-Halos
#
#loadfile('4001ovd-v25', 'M0_120/128-03-4.00', ipath=outpath%ptup['4001']+ 'overdensity/opt_s999_iM12-sm3v25off/')#, subf='0030/fit_p/')
loadfile('4001ovd-v25N1', 'M0_120/128-01-1.00', ipath=outpath%ptup['4001']+ 'overdensity/opt_s999_iM12-sm3v25off/', subf='0050/fit_p/')
#loadfile('4001ovd-v06', 'M0_120/128-03-4.00', ipath=outpath%ptup['4001']+ 'overdensity/opt_s999_iM12-sm3v06off/')#, subf='0030/fit_p/')
loadfile('4001ovd-v06N0', 'M0_120/128-01-0.00', ipath=outpath%ptup['4001']+ 'overdensity/opt_s999_iM12-sm3v06off/', subf='0120/fit_p/')
#loadfile('4001ovd-v06-now', 'M0_120/128-03-4.00', ipath=outpath%ptup['4001']+ 'overdensity/opt_s999_iM12-sm3v06off-now/', subf='0030/fit_p/')

#loadfile('4001ovd-v06', 'M0_120/128-01-4.00', ipath=outpath%ptup['4001']+ 'overdensity/opt_s999_iM12-sm3v06off/')#, subf='0030/fit_p/')
#loadfile('4001ovd-v06-n2dmlim-60', 'M0_120/128-01-4.00', ipath=outpath%ptup['4001']+ 'overdensity/opt_s999_iM12-sm3v06offmlimn-60/', subf='0040/fit_p/')
#loadfile('4001ovd-v06-n2dmlim00', 'M0_120/128-01-4.00', ipath=outpath%ptup['4001']+ 'overdensity/opt_s999_iM12-sm3v06offmlim00/', subf='0035/fit_p/')
#loadfile('4001ovd-v06-n2dmlim69', 'M0_120/128-01-4.00', ipath=outpath%ptup['4001']+ 'overdensity/opt_s999_iM12-sm3v06offmlimn69/', subf='0040/fit_p/')
#loadfile('4001ovd-v06-tol2', 'M0_120/128-01-4.00', ipath=outpath%ptup['4001']+ 'overdensity/opt_s999_iM12-tol2-sm3v06off/')#, subf='0030/fit_p/')


###Galaxies log-loss
#
##loadfile('400log-v25', 'M0_90/128-03-4.00', ipath=outpath%ptup['400']+ 'nngalaxies/opt_s999_iM9-sm3v25off/')
##loadfile('400log-v25-mlim00', 'M0_90/128-03-4.00', ipath=outpath%ptup['400']+ 'nngalaxies/opt_s999_iM9-sm3v25offmlimn00/')
##loadfile('400log-v25-n3dmlim00', 'M0_90/128-02-4.00', ipath=outpath%ptup['400']+ 'nngalaxies/opt_s999_iM9-sm3v25n3doffmlimn00/')
##loadfile('400log-v06', 'M0_90/128-03-4.00', ipath=outpath%ptup['400']+ 'nngalaxies/opt_s999_iM9-sm3v06off/', subf='0000/fit_p/')
##
####Galaxies ovd-loss
##loadfile('400ovd-v25', 'M0_90/128-03-4.00', ipath=outpath%ptup['400']+ 'nngalaxies/overdensity/opt_s999_iM9-sm3v25off/')
##loadfile('400ovd-v25-n3dmlim-60', 'M0_90/128-03-4.00', ipath=outpath%ptup['400']+ 'nngalaxies/overdensity/opt_s999_iM9-sm3v25n3doffmlimn-60/')
##loadfile('400ovd-v10', 'M0_90/128-01-4.00', ipath=outpath%ptup['400']+ 'nngalaxies/overdensity/opt_s999_iM9-sm3v10off/', subf='0050/fit_p/')
##loadfile('400ovd-v100-n3dmlim69', 'M0_90/128-01-4.00', ipath=outpath%ptup['400']+ 'nngalaxies/overdensity/opt_s999_iM9-sm3v100n3doffmlimn69/', subf='0050/fit_p/')
##loadfile('400ovd-v100-n3dmlim00', 'M0_90/128-03-4.00', ipath=outpath%ptup['400']+ 'nngalaxies/overdensity/opt_s999_iM9-sm3v100n3doffmlimn00/')
##
##


print('All files Loaded')

l = 400

#suptitle = 'log loss for M12-- different forms & values of noise (2d, 3d)'
#fname = 'logM12.png'

#suptitle = 'ovd loss for M12-- different forms & values of noise (2d, 3d)'
#fname = 'ovdM12.png'

#suptitle = 'compare log and ovd loss for M12-- different forms & values of noise (2d, 3d)'
#fname = 'lossM12.png'

suptitle = 'compare log and ovd loss for M12-- different forms & values of noise (2d, 3d)'
fname = 'lossnngal.png'

keys = [i[len(str(l)):] for i in report.keys()]
plotsum(keys, l=l, lw=2, ls='-', alpha=0.8, fname=fname,suptitle=suptitle)










#keys = ['m11-offsg20', 'm10-offsg20', 'm09-offsg20']
#plotsum(keys, lw=1, ls='--', alpha=1)
#
#keys = ['m11-offsg20', 'm10-offsg20', 'm09-offsg20']
#plotsum(keys, lw=1, ls='--', alpha=1)
#
#keys = ['m11-3dnoffsg20', 'm10-3dnoffsg20', 'm09-3dnoffsg20']
#plotsum(keys, lw=1, ls='-', alpha=1)
#
#keys = [ 'm11-4ft', 'm11-spl']
##plotsum(keys, lw=1, ls='-', alpha=1)
#
#keys = [ 'm11-off', 'm10-off']
#plotsum(keys, lw=1, ls=':', alpha=1)
#
#keys = [ 'm11-3dnoff', 'm10-3dnoff', 'm09-3dnoff']
#plotsum(keys, lw=1, ls='-.', alpha=1)
#
#key = 'm11-sg20'
#dgrep.summary(report['%d%s'%(l, key)], fig, ax, c='k', lw=1, ls = '--', alpha = 1, label='%s'%key)
#
