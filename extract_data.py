'''
Script for extracting and truncating data from large HDF .h5 file to a more
tractable, constant particle number file with only useful paramters.

This is configured to separate particles by their PID, then sort them by
transverse momentum. For each type of particle, the first n highest
momentum particles are picked out and saved.
'''

import h5py
import numpy as np
import os

TRUNC = 10

# Separate by particle ID

path_qcd   = '/data/shared/Delphes/qcd_lepFilter_13TeV/pandas_joined/'
path_wjet  = '/data/shared/Delphes/wjets_lepFilter_13TeV/pandas_joined'
path_ttbar = '/data/shared/Delphes/ttbar_lepFilter_13TeV/pandas_joined/'

filenames_qcd   = [path_qcd   + fl for fl in next(os.walk(path_qcd))[2]   if '.h5' in fl]
filenames_wjet  = [path_wjet  + fl for fl in next(os.walk(path_wjet))[2]  if '.h5' in fl]
filenames_ttbar = [path_ttbar + fl for fl in next(os.walk(path_ttbar))[2] if '.h5' in fl]

filenames = filenames_qcd + filenames_wjet + filenames_ttbar

# Input Data
X = np.zeros( (len(filenames), 5, 10, 4) )

for i, fl in enumerate(filenames):
    f = h5py.File(fl, 'r')
    fldata = np.array(f['data']['block0_values'])
    # Keep Particle ID, Pt, Phi, Eta, Dxy
    useful_data = np.array([data[:, 4], data[:, 6], data[:, 7], data[:, 8], data[:, 9]]).T

    # Separate by Particle ID, then sort by Pt and take the highest 10
    X[i, 0] = np.sort(useful_data[useful_data[:, 0] == 22], axis=0)[-1:-TRUNC-1:-1, 1:]
    X[i, 1] = np.sort(useful_data[useful_data[:, 0] == 91], axis=0)[-1:-TRUNC-1:-1, 1:]
    X[i, 2] = np.sort(useful_data[useful_data[:, 0] == 90], axis=0)[-1:-TRUNC-1:-1, 1:]
    X[i, 3] = np.sort(useful_data[useful_data[:, 0] == 89], axis=0)[-1:-TRUNC-1:-1, 1:]
    X[i, 4] = np.sort(useful_data[useful_data[:, 0] == 83], axis=0)[-1:-TRUNC-1:-1, 1:]
    
    f.close()


# Ouput; Assign each event type a number
Y = np.empty( (len(filenames), ) )
Y[:len(filenames_qcd)]                    = 1
Y[len(filenames_qcd):len(filenames_wjet)] = 2
Y[len(filenames_wjet):]                   = 3


# Write X and Y to file
f = h5py.File("delphes_trunc_data.h5", "w")
f['X'] = X
f['Y'] = Y

f['X'].dims[0].label = 'events'
f['X'].dims[1].label = 'particles types'
f['X'].dims[2].label = 'particle parameters'

f.close()

