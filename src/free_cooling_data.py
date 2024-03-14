import numpy as np
import h5py

filename = "data/free_cooling/snapshot_000.hdf5"
f = h5py.File(filename, "r")
keys = list(f.keys())
# print(keys)
# config = f[keys[0]]
# print(config)
# print("Attributes in '/Config':")
# for attr_name in config.attrs:
#     print(f"{attr_name}: {config.attrs[attr_name]}")
# header = f[keys[1]]
# print(header)
# print("Attributes in '/Header':")
# for attr_name in header.attrs:
#     print(f"{attr_name}: {header.attrs[attr_name]}")
# parameters = f[keys[2]]
# print(parameters)
# print("Attributes in '/Parameters':")
# for attr_name in parameters.attrs:
#     print(f"{attr_name}: {parameters.attrs[attr_name]}")
parttype0 = f[keys[3]]
print(parttype0)
print(parttype0.keys())
print("Attributes in '/PartType0':")
for attr_name in parttype0.attrs:
    print(f"{attr_name}: {parttype0.attrs[attr_name]}")
members = list(parttype0.keys())

# Extracting data
distfunc = parttype0[members[0]][()]
print("distfunc:")
print(len(distfunc))
print(distfunc[0])
print(distfunc[32])
energy = parttype0[members[1]][()]
print("energy:")
print(len(energy))
print(energy[0])
print(energy[32])
number = parttype0[members[2]][()]
print("number:")
print(len(number))
print(number[0])
print(number[32])
slope = parttype0[members[3]][()]
print("slope:")
print(len(slope))
print(slope[0])
print(slope[32])
f.close()

filename = "data/free_cooling/snapshot_010.hdf5"
f = h5py.File(filename, "r")
keys = list(f.keys())
parttype0 = f[keys[3]]
members = list(parttype0.keys())

# Extracting data
distfunc = parttype0[members[0]][()]
print("distfunc:")
print(len(distfunc))
print(distfunc[0])
print(distfunc[32])
energy = parttype0[members[1]][()]
print("energy:")
print(len(energy))
print(energy[0])
print(energy[32])
number = parttype0[members[2]][()]
print("number:")
print(len(number))
print(number[0])
print(number[32])
slope = parttype0[members[3]][()]
print("slope:")
print(len(slope))
print(slope[0])
print(slope[32])
f.close()
