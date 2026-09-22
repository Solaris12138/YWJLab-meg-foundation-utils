import os
import argparse

from utils.fs_processor import ParallelFreeSurferProcessor


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='The script to download data for pre-training and downstream tasks.')

    parser.add_argument('--n_jobs', default=16, help='Number of jobs to run in parallel.')
    
    args = parser.parse_args()

    N_JOBS = args.n_jobs

    target_root = './data/bids'

    # 'ds003633' and 'ywjaulg' provide FreeSurfer reconstructions
    # So we just skip them.
    datasets_config = {
        # Pre-train datasets
        'ds003922': {
            'input_pattern': os.path.join(target_root, 'ds003922/sub-*/ses-01/anat/sub-*_ses-01_T1w.nii.gz'),
            'output_dir': os.path.join(target_root, 'ds003922/freesurfer'),
            'subject_pattern': r"sub-[A-Za-z0-9]+"
        },
        'ds004107': {
            'input_pattern': os.path.join(target_root, 'ds004107/sub-*/ses-01/anat/sub-*_ses-01_T1w.nii.gz'),
            'output_dir': os.path.join(target_root, 'ds004107/freesurfer'),
            'subject_pattern': r"sub-[A-Za-z0-9]+"
        },
        'ds004078': {
            'input_pattern': os.path.join(target_root, 'ds004078/sub-*/anat/sub-*_run-02_T1w.nii.gz'),
            'output_dir': os.path.join(target_root, 'ds004078/freesurfer'),
            'subject_pattern': r"sub-\d+"
        },
        'ds005356': {
            'input_pattern': os.path.join(target_root, 'ds005356/Code/anat/sub-*_acq-highres_T1w.nii.gz'),
            'output_dir': os.path.join(target_root, 'ds005356/freesurfer'),
            'subject_pattern': r"sub-[A-Za-z0-9]+"
        },
        'ds004212': {
            'input_pattern': os.path.join(target_root, 'ds004212/sub-*/ses-MRI/anat/sub-*_ses-MRI_rec-deface_T1w.nii'),
            'output_dir': os.path.join(target_root, 'ds004212/freesurfer'),
            'subject_pattern': r"sub-[A-Za-z0-9]+"
        },
        'ds006012': {
            'input_pattern': os.path.join(target_root, 'ds006012/sub-*/anat/sub-*_T1w.nii.gz'),
            'output_dir': os.path.join(target_root, 'ds006012/freesurfer'),
            'subject_pattern': r"sub-\d+"
        },
        'ds006334': {
            'input_pattern': os.path.join(target_root, 'ds006334/sub-*/anat/sub-*_T1w.nii'),
            'output_dir': os.path.join(target_root, 'ds006334/freesurfer'),
            'subject_pattern': r"sub-\d+"
        },
        # Downstream datasets
        'ds000117': {
            'input_pattern': os.path.join(target_root, 'ds000117/sub-*/ses-mri/anat/sub-*_ses-mri_acq-mprage_T1w.nii.gz'),
            'output_dir': os.path.join(target_root, 'ds000117/freesurfer'),
            'subject_pattern': r"sub-\d+"
        }
    }
    
    processor = ParallelFreeSurferProcessor(
        freesurfer_home='/usr/local/freesurfer/8.1.0',
        max_n_jobs=N_JOBS
    )
    
    processor.run_sequential_datasets(datasets_config)