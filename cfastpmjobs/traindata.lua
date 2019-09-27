-- parameter file
------ Size of the simulation -------- 

-- For Testing

print(#args)    
if #args ~= 7 then
    print("usage: fastpm highres.lua boxsize nc random_seed output_prefix")
    os.exit(1)
end



boxsize = tonumber(args[1])
nc = tonumber(args[2])
random_seed = tonumber(args[3])
afin = tonumber(args[4])
prefix = args[5]
B = tonumber(args[6])
N = tonumber(args[7])

---print("boxsize")
---print(boxsize)
---print("nc")
---print(nc)
---print("random_seed")
---print(random_seed)
---print("afin")
---print(afin)
---print("prefix")
---print(prefix)
---print("B")
---print(B)
---print("N")
---print(N)
---


-------- Time Sequence ----
-- linspace: Uniform time steps in a
-- time_step = linspace(0.025, 1.0, 39)
-- logspace: Uniform time steps in loga

time_step = linspace(0.01, afin, N)
print(time_step)
--
aout = {afin}

-- Cosmology --
omega_m = 0.3175
h = 0.6711


-- Start with a power spectrum file
-- Initial power spectrum: k P(k) in Mpc/h units
-- Must be compatible with the Cosmology parameter
read_powerspectrum= "/global/u1/c/chmodi/Programs/cosmo4dv2/ics_matterpow_0.dat"

linear_density_redshift = 0.0
remove_cosmic_variance = false

-- linear_density_redshift = 0.0
-- random_seed= 100
-- remove_cosmic_variance = true

-------- Approximation Method ---------------
force_mode = "fastpm"

pm_nc_factor = B            -- Particle Mesh grid pm_nc_factor*nc per dimension in the beginning

np_alloc_factor= 2      -- Amount of memory allocated for particle

-------- Output ---------------

-- prefix = "/"
-- Dark matter particle outputs (all particles)
write_snapshot= prefix .. "/fastpm" 
write_lineark= prefix .. "/linear" 
particle_fraction = 1.00

write_fof     = prefix .. "/fastpm" 
fof_linkinglength = 0.200
fof_nmin = 16

-- 1d power spectrum (raw), without shotnoise correction
-- write_powerspectrum = prefix .. '/powerspec'

