# preprocess dataset from abide
import os
import nibabel
import nilearn

def download(root='./ABIDE'):
    a = nilearn.datasets.fetch_abide_pcp(data_dir=root)

if __name__ == '__main__':
    download()

