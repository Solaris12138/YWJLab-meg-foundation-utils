import os
import argparse
import openneuro as on
import mne

from configs.configs_datasets import openneuro_datasets as on_datasets
from configs.configs_datasets import downstream_datasets as dt_datasets

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='The script to download data for pre-training and downstream tasks.')

    parser.add_argument('--n_jobs', default=5, help='The maximum number of downloads to run in parallel.')
    
    args = parser.parse_args()
    
    target_root = './data/bids'
    if not os.path.exists(target_root):
        os.makedirs(target_root, exist_ok=True)

    # Download pretrain datasets from OpenNeuro
    for dataset in on_datasets:
        bids_root = os.path.join(target_root, dataset)
        if not os.path.exists(bids_root):
            os.mkdir(bids_root)
        
        if dataset == 'ds004078':
            on.download(
                dataset=dataset,
                target_dir=bids_root,
                include=[
                    'README',
                    'dataset_description.json',
                    'participants.json',
                    'participants.tsv',
                    'derivatives/preprocessed_data/sub-*/MEG/**',
                    'sub-*/anat/**'
                ],
                max_concurrent_downloads=args.n_jobs
            )
        elif dataset == 'ds005356':
            on.download(
                dataset=dataset,
                target_dir=bids_root,
                include=[
                    'README',
                    'dataset_description.json',
                    'participants.json',
                    'participants.tsv',
                    'sub-*/ses-*/**',
                    'Code/anat/**',
                    'Code/sss_cal_3046_2009-04-29.dat',
                    'Code/ct_sparse_orion.fif'
                ],
                max_concurrent_downloads=args.n_jobs
            )
        elif dataset == 'ds003633':
            on.download(
                dataset=dataset,
                target_dir=bids_root,
                include=[
                    'README',
                    'dataset_description.json',
                    'participants.json',
                    'participants.tsv',
                    'derivatives/preproc_meg-mne_mri-fmriprep/**',
                    'sub-emptyroom/**'
                ],
                exclude=['derivatives/preproc_meg-mne_mri-fmriprep/sub-*/ses-movie/anat/**'],
                max_concurrent_downloads=args.n_jobs
            )
        elif dataset == 'ds004212':
            on.download(
                dataset=dataset,
                target_dir=bids_root,
                include=[
                    'README',
                    'dataset_description.json',
                    'participants.json',
                    'participants.tsv',
                    'task-main_events.json',
                    'derivatives/preprocessed/**',
                    'sub-*/**'
                ],
                max_concurrent_downloads=args.n_jobs
            )
        elif dataset == 'ds005279':
            on.download(
                dataset=dataset,
                target_dir=bids_root,
                include=[
                    'README',
                    'dataset_description.json',
                    'sub-*/ses-*/**'
                ],
                max_concurrent_downloads=args.n_jobs
            )
        elif dataset == 'ds006468':
            on.download(
                dataset=dataset,
                target_dir=bids_root,
                include=[
                    'README',
                    'dataset_description.json',
                    'participants.json',
                    'participants.tsv',
                    'derivatives/sub-*/coregistration/**',
                    'sub-*/**'
                ],
                max_concurrent_downloads=args.n_jobs
            )
        elif dataset == 'ds006502':
            on.download(
                dataset=dataset,
                target_dir=bids_root,
                include=[
                    'README',
                    'dataset_description.json',
                    'participants.json',
                    'participants.tsv',
                    'sub-*/**'
                ],
                max_concurrent_downloads=args.n_jobs
            )
        elif dataset == 'ds006334':
            on.download(
                dataset=dataset,
                target_dir=bids_root,
                include=[
                    'README',
                    'dataset_description.json',
                    'participants.tsv',
                    'sub-01/**',
                    'sub-04/**',
                    'sub-05/**',
                    'sub-06/**',
                    'sub-07/**',
                    'sub-08/**',
                    'sub-09/**',
                    'sub-10/**',
                    'sub-11/**',
                    'sub-12/**',
                    'sub-14/**',
                    'sub-17/**',
                    'sub-20/**',
                    'sub-21/**',
                    'sub-22/**',
                    'sub-23/**',
                    'sub-24/**',
                    'sub-25/**',
                    'sub-26/**',
                    'sub-28/**',
                    'sub-29/**',
                    'sub-30/**',
                ],
                max_concurrent_downloads=args.n_jobs
            )
        else:
            on.download(
                dataset=dataset,
                target_dir=bids_root,
                include=[
                    'README',
                    'dataset_description.json',
                    'participants.json',
                    'participants.tsv',
                    'sub-*/**'
                ],
                max_concurrent_downloads=args.n_jobs
            )
    
    # Download downstream datasets from OpenNeuro
    for dataset in dt_datasets:
        if not dataset.startswith('ds'):
            continue

        bids_root = os.path.join(target_root, dataset)
        if not os.path.exists(bids_root):
            os.mkdir(bids_root)

        if dataset == 'ds000117':
            on.download(
                dataset=dataset,
                target_dir=bids_root,
                include=[
                    'README',
                    'dataset_description.json',
                    'participants.json',
                    'participants.tsv',
                    'derivatives/meg_derivatives/sub-*/**',
                    'sub-*/ses-mri/anat/*_T1w.nii.gz'
                ],
                max_concurrent_downloads=args.n_jobs
            )
        elif dataset == 'ds003392':
            on.download(
                dataset=dataset,
                target_dir=bids_root,
                max_concurrent_downloads=args.n_jobs
            )
        elif dataset == 'ds005810':
            on.download(
                dataset=dataset,
                target_dir=bids_root,
                include=[
                    'README',
                    'dataset_description.json',
                    'participants.json',
                    'participants.tsv',
                    'derivatives/**',
                    'sub-*/ses-MRI/**',
                    'sub-emptyroom/**'
                ],
                max_concurrent_downloads=args.n_jobs
            )