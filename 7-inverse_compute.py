import os
import logging
import mne

from mne.io import read_info
from mne.minimum_norm import make_inverse_operator, write_inverse_operator

from configs.configs_datasets import zic_datasets
from configs.configs_datasets import openneuro_datasets as on_datasets
from configs.configs_datasets import downstream_datasets as dt_datasets


def _make_inv(fif_fname, fwd_fname, inv_fname, noise_cov):
    
    info = read_info(fif_fname)
    fwd = mne.read_forward_solution(fwd_fname)
    inv = make_inverse_operator(info, fwd, noise_cov, loose=0.2, depth=0.8, fixed='auto', rank='info')
    
    write_inverse_operator(inv_fname, inv, overwrite=True)


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
SPACING = 'oct6'
ICO = 5


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
                    fwd_root = os.path.join(sub_root, 'fwd')
                cov_root = os.path.join(sub_root, 'noise_cov')
                inv_root = os.path.join(sub_root, 'inv')
                if not os.path.exists(inv_root): os.mkdir(inv_root)
                
                noise_cov = mne.read_cov(os.path.join(cov_root, 'noise-cov.fif'))
                
                for fname in os.listdir(sub_root):
                    if not fname.endswith('_meg.fif'): continue
                    _make_inv(
                        fif_fname=os.path.join(sub_root, fname),
                        fwd_fname=os.path.join(fwd_root, fname.replace('_meg.fif', '-fwd.fif')),
                        inv_fname=os.path.join(inv_root, fname.replace('_meg.fif', '-inv.fif')),
                        noise_cov=noise_cov
                    )
            else:
                sub_root = DATA_ROOT_DICT[dataset].replace('SUBJECT', subject)
                fwd_root = os.path.join(sub_root, 'fwd')
                cov_root = os.path.join(sub_root, 'noise_cov')
                inv_root = os.path.join(sub_root, 'inv')
                if not os.path.exists(inv_root): os.mkdir(inv_root)
                
                noise_cov = mne.read_cov(os.path.join(cov_root, 'noise-cov.fif'))
                
                for fname in os.listdir(sub_root):
                    if not fname.endswith('_meg.fif'): continue
                    _make_inv(
                        fif_fname=os.path.join(sub_root, fname),
                        fwd_fname=os.path.join(fwd_root, fname.replace('_meg.fif', '-fwd.fif')),
                        inv_fname=os.path.join(inv_root, fname.replace('_meg.fif', '-inv.fif')),
                        noise_cov=noise_cov
                    )