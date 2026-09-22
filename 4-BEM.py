import os
import logging
import mne

from mne.bem import make_watershed_bem, make_scalp_surfaces

from configs.configs_datasets import zic_datasets
from configs.configs_datasets import openneuro_datasets as on_datasets
from configs.configs_datasets import downstream_datasets as dt_datasets


# Make sure that BEM solution has been computed for fsaverage on your computer/remote-server.
# Asking the root user before running this script is recommended.

# Global Parameters
SPACING = 'oct6'
ICO = 5
SUBJECTS_DIRS_MAP = {
    "ds003633" : "derivatives/preproc_meg-mne_mri-fmriprep/sourcedata/freesurfer",
    "ywjaulg" : "freesurfer"
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

        if dataset not in list(SUBJECTS_DIRS_MAP.keys()):
            subjects_dir = os.path.join(bids_root, 'freesurfer')
        else:
            # Since 'ds003633' provides fsaverage (which is not owned by the root user of this computer/remote-server), 
            # so we need to additionally compute BEM solution here.
            # The same treatment for 'ywjaulg'.
            subjects_dir = os.path.join(bids_root, SUBJECTS_DIRS_MAP[dataset])

            for spacing in ['ico4', 'ico5', 'oct6']:
                subject = 'fsaverage'
                os.makedirs(os.path.join(subjects_dir, subject, 'bem'), exist_ok=True)
                
                src_fname = os.path.join(subjects_dir, subject, 'bem',
                                        f'{subject}-{spacing}-src.fif')

                # Create the surface source space
                src = mne.setup_source_space(subject, spacing, subjects_dir=subjects_dir)
                mne.write_source_spaces(src_fname, src, overwrite=True)
        
        for subject in os.listdir(subjects_dir):
            if not subject.startswith('sub-'): continue
            
            # Make BEMs using watershed bem
            bem_surf_fname = os.path.join(subjects_dir, subject, 'bem',
                                        f'{subject}-{SPACING}-ico{ICO}-bem.fif')
            bem_sol_fname = os.path.join(subjects_dir, subject, 'bem',
                                        f'{subject}-{SPACING}-ico{ICO}-bem-sol.fif')
            src_fname = os.path.join(subjects_dir, subject, 'bem',
                                    f'{subject}-{SPACING}-ico{ICO}-src.fif')
            
            make_watershed_bem(subject,
                                subjects_dir=subjects_dir,
                                overwrite=True,
                                show=False,
                                verbose=False)
            
            make_scalp_surfaces(subject=subject,
                                subjects_dir=subjects_dir,
                                force=True,
                                overwrite=True)

            # make BEM models
            bem_surf = mne.make_bem_model(
                subject,
                ico=ICO,
                conductivity=[0.3],
                subjects_dir=subjects_dir
            )
            mne.write_bem_surfaces(bem_surf_fname, bem_surf, overwrite=True)

            # make BEM solution
            bem_sol = mne.make_bem_solution(bem_surf)
            mne.write_bem_solution(bem_sol_fname, bem_sol, overwrite=True)

            # Create the surface source space
            src = mne.setup_source_space(subject, SPACING, subjects_dir=subjects_dir)
            mne.write_source_spaces(src_fname, src, overwrite=True)

        logging.info(f'--- Processing Dataset: {dataset} Completed ---')