from distutils.util import change_root
from scipy.io import loadmat, savemat
import numpy as np

'''
data = loadmat('./dataset/rap2/RAP_annotation/RAP_annotation.mat')
print(data.keys())
print(type(data['RAP_annotation']),data['RAP_annotation'].shape)
print(data['RAP_annotation']['attribute'])
print(data['RAP_annotation']['selected_attribute'])
'''


data = loadmat('./dataset/peta/PETA_1.mat')
print(data['peta']['attribute'])


print(data['peta']['selected_attribute'])
attr_list = [i for i in range(1, 105)]
print(attr_list)
'''
attr_list = [5, 6, 11, 16, 17, 18, 21, 22, 23, 31, \
        36, 37, 38, 39, 40, 41, 42, 43, 44, 45, \
        46, 47, 48, 49, 50, 51, 52, 53, 54, 55, \
        56, 57, 58, 59, 60, 61, 62, 63, 64, 65, \
        66, 67, 68, 69, 70, 71, 72, 73, 74, 75, \
        76, 77, 78, 79, 83, 88, 95, 99]
'''
change_data = np.array(attr_list, dtype=np.uint8)
d = np.array(change_data)
print(d)
# # change_data = np.reshape(change_data, (1,1))
data['peta']['selected_attribute'][0, 0] = d

savemat('./dataset/peta/PETA.mat', data)
data = loadmat('./dataset/peta/PETA.mat')
#print(data['peta']['attribute'])
print(data['peta']['selected_attribute'])
