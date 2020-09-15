import os
import nibabel as nib
import numpy as np

from nilearn import plotting
from nilearn import image
from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker, NiftiMapsMasker
from nilearn.connectome import ConnectivityMeasure
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


def load_paths(root='./'):
	paths = []
	for r, dirs, names in os.walk(root):
		for name in names:
			if '.nii.gz' in name:
				paths += [os.path.join(r, name)]
	return paths

def load_csv(root='./'):
	lines = open(root, 'r').readlines()
	header = "wev,"+lines[0]
	headers = header.strip().split(',')

	labels = []
	for line in lines[1:]:
		conts = line.strip().split(',')
		label = {}
		for idx, key in enumerate(headers[1:]):
			label[key] = conts[idx]
		#print(label)
		labels.append(label)
	return labels

def load_data(img_root='./', csv='./'):
	img_paths = load_paths(img_root)
	labels = load_csv(csv)
	res = []
	# add img_path to labels
	for path in img_paths:
		name = path.split('/')[-1]
		uni_id = name.split('_')[0]
		sub_id = name.split('_')[1]
		file_name = uni_id + '_' + sub_id

		for idx, label in enumerate(labels):
			if label['FILE_ID'] == file_name:
				label['img_path'] = path
				res.append(label)
	return res

def single_uni(img_root, csv, uni_name='PITT'):
	datalist = load_data(img_root='../dataset/ABIDE', csv='../dataset/ABIDE/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv')
	unilist = []
	for dicts in datalist:
		if dicts['SITE_ID'] == uni_name:
			unilist += [dicts]
	return unilist

def __debug_show_img_info(data_info):
	img_path = data_info['img_path']
	img = nib.load(img_path)
	#print(img)
	'''
	# compute the voxel_wise of functional images across time
	# reducing the functional image from 4D to 3D
	mean_img = image.mean_img(img_path)

	# plot the data info
	plotting.plot_epi(mean_img, title='plot_epi')
	plotting.show()
	'''
	'''
	#loading yeo atlas with parcellations
	yeo = datasets.fetch_atlas_yeo_2011()
	connectome_measure = ConnectivityMeasure(kind='correlation')
	masker = NiftiLabelsMasker(labels_img=yeo['thick_17'], standardize=True, memory='nilearn_cache')

	# extract time series from this subject
	time_series = [masker.fit_transform(img_path)]
	correlation_matrices = connectome_measure.fit_transform(time_series)
	coordinates = plotting.find_parcellation_cut_coords(labels_img=yeo['thick_17'])
	plotting.plot_connectome(connectome_measure.mean_, coordinates, edge_threshold='80%', title='yeo atlas')
	'''

	atlas = datasets.fetch_atlas_msdl()
	#atlas = datasets.fetch_atlas_aal()
	atlas_filename = atlas['maps']
	labels = atlas['labels']
	masker = NiftiMapsMasker(maps_img=atlas_filename, standardize=True, memory='nilear_cache', verbose=5)
	time_series = masker.fit_transform(img_path)

	correlation_measure = ConnectivityMeasure(kind='correlation')
	correlation_matrix = correlation_measure.fit_transform([time_series])[0]

	np.fill_diagonal(correlation_matrix, 0)
	plotting.plot_matrix(correlation_matrix, labels=labels, colorbar=True, vmax=0.8, vmin=-0.8)

	coords = atlas.region_coords
	plotting.plot_connectome(correlation_matrix, coords, edge_threshold='80%', colorbar=True)

	#plotting.show()
	print(time_series.shape)
	plt.figure(figsize=(7,5))
	plt.plot(time_series[:150, :])
	plt.xlabel('Time [TRs]', fontsize=16)
	plt.ylabel('Intensity', fontsize=16)
	plt.xlim(0, 150)
	plt.subplots_adjust(bottom=.12, top=.95, right=.95, left=.12)
	plt.show()


def __debug_show_img_label(data_info):
	# this is the label for autisim and control groups
	print('Dx_group: %s' % ("autism" if data_info['DX_GROUP'] == '1' else 'Control'))
	# this is the more detail labels for more detail classification
	label2 = data_info['DSM_IV_TR']
	if label2 == '0': 
		print('Control')
	elif label2 == '1': 
		print('Autism')
	elif label2 == '2':
		print('Asperger')
	elif label2 == '3':
		print('PDDNOS')
	elif label2 == '4':
		print('Asperger or PDDNOS')

	print('AGE: %s' %(data_info['AGE_AT_SCAN']))
	print('Sex: %s' % ('male' if data_info['SEX']=='1' else 'female')) # 1 for male, 2 for female
	print('Hand: %s' % (data_info['HANDEDNESS_CATEGORY'])) # R & L & ambidextrous


if __name__ == '__main__':
	#labels = load_csv('../dataset/ABIDE/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv')
	#load_data(img_root='../dataset/ABIDE', csv='../dataset/ABIDE/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv')
	unilists = single_uni('../dataset/ABIDE', '../dataset/ABIDE/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv', uni_name='PITT')
	print(len(unilists))
	__debug_show_img_info(unilists[0])
	__debug_show_img_label(unilists[0])

