import os
import logging
import numpy as np
import mne

from mne.coreg import Coregistration
from mne.io import read_info

from configs.configs_datasets import zic_datasets
from configs.configs_datasets import openneuro_datasets as on_datasets
from configs.configs_datasets import downstream_datasets as dt_datasets


def _auto_coregistration(fif_fname, trans_fname, subject, subjects_dir, fiducials):
    
    info = read_info(fif_fname)
    
    coreg = Coregistration(info, subject, subjects_dir, fiducials=fiducials)
    coreg.fit_fiducials(verbose=True)
    coreg.fit_icp(n_iterations=100, nasion_weight=5., verbose=True)
    coreg.omit_head_shape_points(distance=5. / 1000)
    coreg.fit_icp(n_iterations=50, nasion_weight=10., verbose=True)
    dists = coreg.compute_dig_mri_distances() * 1e3
    print(
        f"Distance between HSP and MRI (mean/min/max):\n{np.mean(dists):.2f} mm "
        f"/ {np.min(dists):.2f} mm / {np.max(dists):.2f} mm"
    )
    
    mne.write_trans(trans_fname, coreg.trans, overwrite=True)
    

DATA_ROOT_DICT = {
    "ds000117" : "./data/bids/ds000117/derivatives/meg_derivatives/SUBJECT/ses-meg/meg",
    "ds003633" : "./data/bids/ds003633/derivatives/preproc_meg-mne_mri-fmriprep/SUBJECT/ses-movie/meg",
    "ds003922" : "./data/bids/ds003922/SUBJECT/ses-01/meg",
    "ds004078" : "./data/bids/ds004078/derivatives/preprocessed_data/SUBJECT/MEG",
    "ds004107" : ["./data/bids/ds004107/SUBJECT/ses-01/meg","./data/bids/ds004107/SUBJECT/ses-02/meg"],
    "ds004212" : [f"./data/bids/ds004212/SUBJECT/ses-{i:02d}/meg" for i in range(1, 13)],
    "ds005356" : "./data/bids/ds005356/SUBJECT/ses-01/meg",
    "ds006012" : "./data/bids/ds006012/SUBJECT/meg",
    "ds006334" : "./data/bids/ds006334/SUBJECT/meg",
    "ywjaulg" : "./data/bids/ywjaulg/SUBJECT/ses-01/meg"
}


if __name__ == '__main__':

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )

    target_root = './data/bids'

    for dataset in (on_datasets + dt_datasets + zic_datasets):
        logging.info(f'--- Processing Dataset: {dataset} ---')
        bids_root = os.path.join(target_root, dataset)

        if dataset != 'ds003633':
            subjects_dir = os.path.join(bids_root, 'freesurfer')
        else:
            subjects_dir = os.path.join(bids_root, 'derivatives/preproc_meg-mne_mri-fmriprep/sourcedata/freesurfer')
            
        for subject in os.listdir(subjects_dir):
            if not subject.startswith('sub-'): continue
            
            if isinstance(DATA_ROOT_DICT[dataset], list):
                for data_root in DATA_ROOT_DICT[dataset]:
                    sub_root = data_root.replace('SUBJECT', subject)
                    trans_root = os.path.join(sub_root, 'trans')
                    if not os.path.exists(trans_root): os.mkdir(trans_root)
                    for fname in os.listdir(sub_root):
                        if not fname.endswith('_meg.fif'): continue
                        _auto_coregistration(
                            fif_fname=os.path.join(sub_root, fname),
                            trans_fname=os.path.join(trans_root, fname.replace('_meg.fif', '-trans.fif')),
                            subject=subject,
                            subjects_dir=subjects_dir,
                            fiducials='auto'
                        )
            else:
                sub_root = DATA_ROOT_DICT[dataset].replace('SUBJECT', subject)
                trans_root = os.path.join(sub_root, 'trans')
                if not os.path.exists(trans_root): os.mkdir(trans_root)
                for fname in os.listdir(sub_root):
                    if not fname.endswith('_meg.fif'): continue
                    _auto_coregistration(
                        fif_fname=os.path.join(sub_root, fname),
                        trans_fname=os.path.join(trans_root, fname.replace('_meg.fif', '-trans.fif')),
                        subject=subject,
                        subjects_dir=subjects_dir,
                        fiducials='auto'
                    )