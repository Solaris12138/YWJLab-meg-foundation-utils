import os
import math
import glob
import argparse
import logging
import pandas as pd
import numpy as np
import mne
import json

from mne.io import read_raw_fif, read_raw_ctf
from mne.preprocessing import annotate_muscle_zscore, ICA, maxwell_filter, maxwell_filter_prepare_emptyroom, find_bad_channels_maxwell
from mne_bids import BIDSPath, find_matching_paths, read_raw_bids
from mne_icalabel import label_components

from configs.configs_datasets import zic_datasets
from configs.configs_datasets import openneuro_datasets as on_datasets
from configs.configs_datasets import downstream_datasets as dt_datasets
from configs.configs_datasets import reject_criteria as reject


# Global Parameters
ICA_METHOD = 'infomax'
N_COMPONENTS = 20
ICALABEL_METHOD = 'megnet'
ICALABEL_FILTER = (1.0, 100.0)
RESAMPLE_SFREQ = 1000
LOWPASS_FILTER = 330.0
RANDOM_STATE = 1016


def _raw_MaxFilter(raw, calibration, cross_talk, if_tsss=True, coord_frame='head'):
    auto_noisy_chs, auto_flat_chs = find_bad_channels_maxwell(
        raw,
        calibration=calibration,
        cross_talk=cross_talk,
        coord_frame=coord_frame,
        return_scores=False,
        verbose=True
    )
    bads = raw.info['bads'] + auto_noisy_chs + auto_flat_chs
    raw.info['bads'] = bads
    
    if if_tsss:
        raw_ = maxwell_filter(
            raw,
            calibration=calibration,
            cross_talk=cross_talk,
            st_correlation=0.98,
            st_duration=10,
            coord_frame=coord_frame,
            verbose=True
        )
    else:
        raw_ = maxwell_filter(
            raw,
            calibration=calibration,
            cross_talk=cross_talk,
            st_correlation=0.98,
            st_duration=None,
            coord_frame=coord_frame,
            verbose=True
        )

    return raw_


def _pre_processing(raw, line_freq, n_jobs=None, if_resample=True, if_lowpass=True, picks_meg=None):
    if if_lowpass:
        raw = raw.filter(l_freq=None, h_freq=LOWPASS_FILTER, n_jobs=n_jobs, verbose=False)
    if if_resample:
        raw = raw.resample(sfreq=RESAMPLE_SFREQ, npad='auto', n_jobs=n_jobs)
    raw.load_data().notch_filter(np.arange(line_freq, line_freq * 5 + 1, line_freq))

    threshold_muscle = 10
    annotations_muscle, scores_muscle = annotate_muscle_zscore(
        raw, 
        ch_type='mag', 
        threshold=threshold_muscle, 
        min_length_good=0.2,
        filter_freq=[110, 140]
    )
    annotations_event = raw.annotations 
    raw.set_annotations(annotations_event + annotations_muscle)

    raw_filt = raw.copy().filter(*ICALABEL_FILTER, n_jobs=n_jobs, verbose=False)
    raw_filt.resample(sfreq=250, npad='auto')
    
    if not picks_meg:
        picks_meg = mne.pick_types(raw_filt.info, meg=True, exclude='bads')
    ica = ICA(n_components=N_COMPONENTS, method=ICA_METHOD, random_state=RANDOM_STATE, max_iter='auto')
    ica.fit(raw_filt, picks=picks_meg)

    label_res = label_components(raw_filt, ica, method=ICALABEL_METHOD)
    labels = label_res['labels']
    removed_by_model = [i for i, l in enumerate(labels) if l in ['eye blink', 'eye movement', 'heart beat']]

    final_removed = removed_by_model

    try: 
        ecg_inds, ecg_scores = ica.find_bads_ecg(raw_filt)
        final_removed += ecg_inds
    except Exception as e:
        print('Error when finding ECG components: ', str(e))
        pass

    try:
        eog_inds, eog_scores = ica.find_bads_eog(raw_filt, ch_name=None)
        final_removed += eog_inds
    except Exception as e:
        print('Error when finding EOG components: ', str(e))
        pass

    try:
        muscle_inds, muscle_scores = ica.find_bads_muscle(raw_filt)
        final_removed += muscle_inds
    except Exception as e:
        print('Error when finding muscle components: ', str(e))
        pass

    final_removed = sorted(set(final_removed))

    reconst_raw = raw.copy()
    ica.apply(reconst_raw, exclude=final_removed)

    return reconst_raw


if __name__ == '__main__':

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )
    
    parser = argparse.ArgumentParser(description='The script to pre-process data for pre-training and downstream tasks.')

    parser.add_argument('--n_jobs', default=32, help='Number of jobs to run in parallel.')
    parser.add_argument('-t', '--tsss', help='Apply spatiotemporal SSS.', action='store_true')
    
    args = parser.parse_args()

    N_JOBS = args.n_jobs
    IF_TSSS = args.tsss
    
    target_root = './data/bids'

    # Perform pre-processing procedures for pretrain datasets from OpenNeuro
    for dataset in on_datasets:
        logging.info(f'--- Processing Dataset: {dataset} ---')
        bids_root = os.path.join(target_root, dataset)
        
        # Perform MaxFilter, SSS and preprocessing
        failed_maxfilter = []
        if dataset == 'ds003922':
            line_freq = 50
            ## Perform MaxFilter on emptyroom recordings and compute noise covariance
            for sub in os.listdir(bids_root):
                if not sub.startswith('sub-') or 'sub-emptyroom' in sub: continue
                calibration = f'./data/bids/ds003922/{sub}/ses-01/meg/{sub}_ses-01_acq-calibration_meg.dat'
                cross_talk = f'./data/bids/ds003922/{sub}/ses-01/meg/{sub}_ses-01_acq-crosstalk_meg.fif'
                bids_path = BIDSPath(subject=sub.split('-')[1], session='01', run='01', 
                                     task='rest', datatype='meg', root=bids_root)
                raw = read_raw_bids(bids_path, extra_params=dict(allow_maxshield=True))
                raw_check = raw.copy()
                auto_noisy_chs, auto_flat_chs = find_bad_channels_maxwell(
                    raw_check,
                    cross_talk=cross_talk,
                    calibration=calibration,
                    return_scores=False,
                    verbose=True,
                )
                bads = raw.info['bads'] + auto_noisy_chs + auto_flat_chs
                raw.info['bads'] = bads
                ch_names = list()
                for c in raw.info['ch_names']:
                    if c.startswith('MEG'): ch_names.append(c)
                df = pd.read_csv(os.path.join(bids_root, sub, f'ses-01/{sub}_ses-01_scans.tsv'), sep='\t')
                date = df['acq_time'].values[0][:10].replace('-', '')
                er_fname = os.path.join(
                    bids_root,
                    f'sub-emptyroom/ses-{date}',
                    'meg',
                    f'sub-emptyroom_ses-{date}_task-noise_meg.fif'
                )
                if os.path.exists(er_fname):
                    raw_er = read_raw_fif(er_fname, allow_maxshield=True)
                    raw_er_prepared = maxwell_filter_prepare_emptyroom(
                        raw_er=raw_er,
                        raw=raw,
                        bads='from_raw',
                        emit_warning=False
                    )
                    raw_er_prepared = raw_er_prepared.load_data().filter(l_freq=None, h_freq=LOWPASS_FILTER, n_jobs=N_JOBS, verbose=False)
                    raw_er_prepared = raw_er_prepared.resample(sfreq=RESAMPLE_SFREQ, npad='auto', n_jobs=N_JOBS)
                    raw_er_prepared = _raw_MaxFilter(raw_er_prepared, calibration, cross_talk, IF_TSSS)
                    raw_er_prepared.load_data().notch_filter(np.arange(line_freq, line_freq * 7 + 1, line_freq))
                    noise_cov = mne.compute_raw_covariance(raw_er_prepared, 
                                                        tmin=0, tmax=None,
                                                        picks='meg', reject=reject,
                                                        method='auto', cv=5)
                else:
                    noise_cov = mne.Covariance(data=np.eye(len(ch_names)), names=ch_names, bads=bads, projs=[], nfree=1)
                cov_root = os.path.join(bids_root, f'{sub}/ses-01/meg/noise_cov')
                if not os.path.exists(cov_root):
                    os.mkdir(cov_root)
                mne.write_cov(os.path.join(cov_root, 'noise-cov.fif'), noise_cov, overwrite=True)
            ## Perform MaxFilter on task recordings
            bids_paths = find_matching_paths(
                bids_root, datatypes='meg', suffixes='meg', sessions=None, ignore_json=True
            )
            for bids_path in bids_paths:
                if 'calibration' in bids_path.basename or 'crosstalk' in bids_path.basename or 'sub-emptyroom' in bids_path.basename: continue
                sub, session = bids_path.basename.split('_')[0], bids_path.basename.split('_')[1]
                raw = read_raw_bids(bids_path, extra_params=dict(allow_maxshield=True, preload=True))
                try:
                    calibration = f'./data/bids/ds003922/{sub}/{session}/meg/{sub}_{session}_acq-calibration_meg.dat'
                    cross_talk = f'./data/bids/ds003922/{sub}/{session}/meg/{sub}_{session}_acq-crosstalk_meg.fif'
                    raw = _raw_MaxFilter(raw, calibration, cross_talk, IF_TSSS)
                except:
                    failed_maxfilter.append(f'./data/bids/ds003922/{sub}/{session}/meg/{bids_path.basename}')
                    continue
                raw = _pre_processing(raw, line_freq, N_JOBS)
                raw.save(
                    os.path.join(bids_root, f'{sub}/{session}/meg/{bids_path.basename}'),
                    overwrite=True
                )
            del_files = glob.glob(os.path.join(bids_root, '*/*/meg/**_acq-calibration_meg.dat')) + \
                        glob.glob(os.path.join(bids_root, '*/*/meg/**_acq-crosstalk_meg.fif')) + \
                        failed_maxfilter
            for file in del_files:
                os.system('rm -rf {0}'.format(file))
            ## We don't need emptyroom anymore
            cmd = f'rm -rf {os.path.join(bids_root, "sub-emptyroom")}'
            os.system(cmd)
        elif dataset == 'ds005356':
            ## This dataset provides no emptyroom recordings
            ## Perform MaxFilter only on task recordings
            line_freq = 60
            bids_paths = find_matching_paths(
                bids_root, datatypes='meg', suffixes='meg', sessions=None, ignore_json=True
            )
            for bids_path in bids_paths:
                if 'calibration' in bids_path.basename or 'crosstalk' in bids_path.basename: continue
                if 'split-02' in bids_path.basename:
                    failed_maxfilter.append(f'./data/bids/ds005356/{sub}/{session}/meg/{bids_path.basename}')
                    continue
                sub, session = bids_path.basename.split('_')[0], bids_path.basename.split('_')[1]
                raw = read_raw_bids(bids_path, extra_params=dict(allow_maxshield=True, preload=True))
                try:
                    if sub != 'sub-M87198523':
                        calibration = './data/bids/ds005356/Code/sss_cal_3046_2009-04-29.dat'
                        cross_talk = './data/bids/ds005356/Code/ct_sparse_orion.fif'
                        raw = _raw_MaxFilter(raw, calibration, cross_talk, IF_TSSS)
                    else:
                        calibration = f'./data/bids/ds005356/{sub}/{session}/meg/{sub}_{session}_acq-calibration_meg.dat'
                        cross_talk = f'./data/bids/ds005356/{sub}/{session}/meg/{sub}_{session}_acq-crosstalk_meg.fif'
                        raw = _raw_MaxFilter(raw, calibration, cross_talk, IF_TSSS)
                        failed_maxfilter.extend(
                            [
                                f'./data/bids/ds005356/{sub}/{session}/meg/{sub}_{session}_acq-calibration_meg.dat',
                                f'./data/bids/ds005356/{sub}/{session}/meg/{sub}_{session}_acq-crosstalk_meg.fif'
                            ]
                        )
                except:
                    failed_maxfilter.append(f'./data/bids/ds005356/{sub}/{session}/meg/{bids_path.basename}')
                    continue
                raw = _pre_processing(raw, line_freq, N_JOBS)
                raw.save(
                    os.path.join(bids_root, f'{sub}/{session}/meg/{bids_path.basename}'),
                    overwrite=True
                )
            del_files = failed_maxfilter
            for file in del_files:
                os.system('rm -rf {0}'.format(file))
            ## Since this dataset includes no emptyroom recordings, we could not estimate noise covariance using emptyroom data.
            ## Here, we used the time period before stimulus to compute noise covariance matrix.
            ## If the pre-stim period is too short, e.g. < 40 sec, we would use an identity matrix instead.
            ## This may not be a good option, but we have no better way.
            ch_names = list()
            for c in raw.info['ch_names']:
                if c.startswith('MEG'): ch_names.append(c)
            for sub in os.listdir(bids_root):
                if not sub.startswith('sub-'): continue
                tsv_fname = os.path.join(bids_root, f'{sub}/ses-01/meg/{sub}_ses-01_task-pst_run-1_events.tsv')
                if not os.path.exists(tsv_fname):
                    noise_cov = mne.Covariance(data=np.eye(len(ch_names)), names=ch_names, bads=[], projs=[], nfree=1)
                    cov_root = os.path.join(bids_root, f'{sub}/ses-01/meg/noise_cov')
                    if not os.path.exists(cov_root):
                        os.mkdir(cov_root)
                    mne.write_cov(os.path.join(cov_root, 'noise-cov.fif'), noise_cov, overwrite=True)
                    continue

                df = pd.read_csv(os.path.join(bids_root, f'{sub}/ses-01/meg/{sub}_ses-01_task-pst_run-1_events.tsv'), sep='\t')
                onset = df['onset'].values[0]
                onset = math.floor(onset) - 10

                raw_fname = None
                candidate_fnames = [
                    os.path.join(bids_root, f'{sub}/ses-01/meg/{sub}_ses-01_task-pst_run-1_meg.fif'),
                    os.path.join(bids_root, f'{sub}/ses-01/meg/{sub}_ses-01_task-pst_run-1_split-01_meg.fif')
                ]
                for fname in candidate_fnames:
                    if os.path.exists(fname):
                        raw_fname = fname
                        break                       
                if raw_fname is not None:
                    if onset > 40:
                        raw = read_raw_fif(raw_fname, preload=True).crop(tmin=0., tmax=onset)
                        try:
                            noise_cov = mne.compute_raw_covariance(raw, tmin=0, tmax=None, 
                                                                    picks='meg', reject=reject,
                                                                    method='auto', cv=5)
                        except ValueError:
                            print('All epochs were dropped! Use identity matrix instead.')
                            noise_cov = mne.Covariance(data=np.eye(len(ch_names)), names=ch_names, bads=[], projs=[], nfree=1)
                    else:
                        noise_cov = mne.Covariance(data=np.eye(len(ch_names)), names=ch_names, bads=[], projs=[], nfree=1)
                    cov_root = os.path.join(bids_root, f'{sub}/ses-01/meg/noise_cov')
                    if not os.path.exists(cov_root):
                        os.mkdir(cov_root)
                    mne.write_cov(os.path.join(cov_root, 'noise-cov.fif'), noise_cov, overwrite=True)
                else: continue
        elif dataset == 'ds006012':
            ## Perform MaxFilter on emptyroom recordings and compute noise covariance
            line_freq = 50
            for sub in os.listdir(bids_root):
                if not sub.startswith('sub-') or 'sub-emptyroom' in sub: continue
                calibration = f'./data/bids/ds006012/{sub}/meg/{sub}_acq-calibration_meg.dat'
                cross_talk = f'./data/bids/ds006012/{sub}/meg/{sub}_acq-crosstalk_meg.fif'
                raw = read_raw_fif(os.path.join(bids_root, f'{sub}/meg/{sub}_task-POGS_run-01_meg.fif'), allow_maxshield=True)
                ch_names = list()
                for c in raw.info['ch_names']:
                    if c.startswith('MEG'): ch_names.append(c)
                raw_check = raw.copy()
                auto_noisy_chs, auto_flat_chs = find_bad_channels_maxwell(
                    raw_check,
                    cross_talk=cross_talk,
                    calibration=calibration,
                    return_scores=False,
                    verbose=True,
                )
                bads = raw.info['bads'] + auto_noisy_chs + auto_flat_chs
                raw.info['bads'] = bads
                df = pd.read_csv(os.path.join(bids_root, sub, f'{sub}_scans.tsv'), sep='\t')
                date = df['acq_time'].values[0][:10].replace('-', '')
                er_fname = os.path.join(
                    bids_root,
                    f'sub-emptyroom/ses-{date}',
                    'meg',
                    f'sub-emptyroom_ses-{date}_task-noise_meg.fif'
                )
                if os.path.exists(er_fname):
                    raw_er = read_raw_fif(er_fname, allow_maxshield=True)
                    raw_er_prepared = maxwell_filter_prepare_emptyroom(
                        raw_er=raw_er,
                        raw=raw,
                        bads='from_raw',
                        emit_warning=False
                    )
                    raw_er_prepared = raw_er_prepared.load_data().filter(l_freq=None, h_freq=LOWPASS_FILTER, n_jobs=N_JOBS, verbose=False)
                    raw_er_prepared = raw_er_prepared.resample(sfreq=RESAMPLE_SFREQ, npad='auto', n_jobs=N_JOBS)
                    raw_er_prepared = _raw_MaxFilter(raw_er_prepared, calibration, cross_talk, IF_TSSS)
                    raw_er_prepared.load_data().notch_filter(np.arange(line_freq, line_freq * 7 + 1, line_freq))
                    noise_cov = mne.compute_raw_covariance(raw_er_prepared, 
                                                           tmin=0, tmax=None,
                                                           picks='meg', reject=reject,
                                                           method='auto', cv=5)
                else:
                    noise_cov = mne.Covariance(data=np.eye(len(ch_names)), names=ch_names, bads=bads, projs=[], nfree=1)
                cov_root = os.path.join(bids_root, f'{sub}/meg/noise_cov')
                if not os.path.exists(cov_root):
                    os.mkdir(cov_root)
                mne.write_cov(os.path.join(cov_root, 'noise-cov.fif'), noise_cov, overwrite=True)
            ## Perform MaxFilter on task recordings
            bids_paths = find_matching_paths(
                bids_root, datatypes='meg', suffixes='meg', sessions=None, ignore_json=True
            )
            for bids_path in bids_paths:
                if 'calibration' in bids_path.basename or 'crosstalk' in bids_path.basename: continue
                sub = bids_path.basename.split('_')[0]
                raw = read_raw_bids(bids_path, extra_params=dict(allow_maxshield=True, preload=True))
                try:
                    calibration = f'./data/bids/ds006012/{sub}/meg/{sub}_acq-calibration_meg.dat'
                    cross_talk = f'./data/bids/ds006012/{sub}/meg/{sub}_acq-crosstalk_meg.fif'
                    raw = _raw_MaxFilter(raw, calibration, cross_talk, IF_TSSS)
                except:
                    failed_maxfilter.append(f'./data/bids/ds006012/{sub}/meg/{bids_path.basename}')
                    continue
                raw = _pre_processing(raw, line_freq, N_JOBS)
                raw.save(
                    os.path.join(bids_root, f'{sub}/meg/{bids_path.basename}'),
                    overwrite=True
                )
            del_files = glob.glob(os.path.join(bids_root, '*/meg/**_acq-calibration_meg.dat')) + \
                        glob.glob(os.path.join(bids_root, '*/meg/**_acq-crosstalk_meg.fif')) + \
                        failed_maxfilter
            for file in del_files:
                os.system('rm -rf {0}'.format(file))
            ## We don't need emptyroom anymore
            cmd = f'rm -rf {os.path.join(bids_root, "sub-emptyroom")}'
            os.system(cmd)
        elif dataset == 'ds004107':
            ## This dataset provides emptyroom data and task data, both processed by MAXFilter and SSS.
            ## We only need to perform preprocessing and noise covariance estimation.
            line_freq = 60
            for sub in os.listdir(bids_root):
                if not sub.startswith('sub-') or 'sub-emptyroom' in sub: continue
                for session in ['01', '02']:
                    df = pd.read_csv(os.path.join(bids_root, sub, f'ses-{session}/{sub}_ses-{session}_scans.tsv'), sep='\t')
                    date = df['acq_time'].values[0][:10].replace('-', '')
                    er_fname = os.path.join(
                        bids_root,
                        f'sub-emptyroom/ses-{date}',
                        'meg',
                        f'sub-emptyroom_ses-{date}_task-noise_meg.fif'
                    )
                    raw_er = read_raw_fif(er_fname)
                    raw_er = raw_er.load_data().filter(l_freq=None, h_freq=LOWPASS_FILTER, n_jobs=N_JOBS, verbose=False)
                    raw_er = raw_er.resample(sfreq=RESAMPLE_SFREQ, npad='auto', n_jobs=N_JOBS)
                    raw_er.load_data().notch_filter(np.arange(line_freq, line_freq * 7 + 1, line_freq)) 
                    if date == '19130117':
                        raw_er.info['bads'].append('MEG 0311')                
                    noise_cov = mne.compute_raw_covariance(raw_er, 
                                                        tmin=0, tmax=None,
                                                        picks='meg', reject=reject,
                                                        method='auto', cv=5)
                    cov_root = os.path.join(bids_root, f'{sub}/ses-{session}/meg/noise_cov')
                    if not os.path.exists(cov_root):
                        os.mkdir(cov_root)
                    mne.write_cov(os.path.join(cov_root, 'noise-cov.fif'), noise_cov, overwrite=True)
            
            bids_paths = find_matching_paths(
                bids_root, datatypes='meg', suffixes='meg', sessions=None, ignore_json=True
            )
            for bids_path in bids_paths:
                if 'sub-emptyroom' in bids_path.basename: continue
                sub, session = bids_path.basename.split('_')[0], bids_path.basename.split('_')[1]
                raw = read_raw_bids(bids_path, extra_params=dict(preload=True))
                raw = _pre_processing(raw, line_freq, N_JOBS)
                raw.save(
                    os.path.join(bids_root, f'{sub}/{session}/meg/{bids_path.basename}'),
                    overwrite=True
                )
            ## We don't need emptyroom anymore
            cmd = f'rm -rf {os.path.join(bids_root, "sub-emptyroom")}'
            os.system(cmd)
        elif dataset == 'ds004078':
            ## This dataset provides only task data, processed by MAXFilter, SSS and preprocessing procedures.
            ## Here, we used the time period before stimulus to compute noise covariance matrix.
            ## However, each time period is too short (around 10s) to estimate noise covariance matrix (at least 47s needed).
            ## So, we would combine these time periods together for estimation.
            line_freq = 50
            bids_root = os.path.join(bids_root, 'derivatives/preprocessed_data')
            for sub in os.listdir(bids_root):
                raws = list()
                for run in range(1, 61):
                    df = pd.read_csv(os.path.join(bids_root, f'{sub}/MEG/{sub}_task-RDR_run-{run}_events.tsv'), sep='\t')
                    onset = df['onset'].values[np.where(df['trial_type'].values == 'Beg')[0]]
                    onset = math.floor(onset[0]) - 3
                    raw = read_raw_fif(
                        os.path.join(bids_root, f'{sub}/MEG/{sub}_task-RDR_run-{run}_meg.fif'),
                        preload=True
                    )
                    raw = raw.resample(sfreq=RESAMPLE_SFREQ, npad='auto', n_jobs=N_JOBS)
                    raw.load_data().notch_filter(np.arange(line_freq, line_freq * 7 + 1, line_freq))
                    raw.save(
                        os.path.join(bids_root, f'{sub}/MEG/{sub}_task-RDR_run-{run}_meg.fif'),
                        overwrite=True
                    )
                    raw = raw.crop(tmin=0., tmax=onset)
                    raws.append(raw)
                raws = mne.concatenate_raws(raws, on_mismatch='ignore')
                noise_cov = mne.compute_raw_covariance(raws, 
                                                       tmin=0, tmax=None,
                                                       picks='meg', reject=reject,
                                                       method='auto', cv=5, rank='info')
                cov_root = os.path.join(bids_root, f'{sub}/MEG/noise_cov')
                if not os.path.exists(cov_root):
                    os.mkdir(cov_root)
                mne.write_cov(os.path.join(cov_root, 'noise-cov.fif'), noise_cov, overwrite=True)
        elif dataset == 'ds003633':
            ## This dataset is a CTF dataset.
            ## Perform MaxFilter on emptyroom recordings and compute noise covariance
            line_freq = 50
            derivatives_root = os.path.join(bids_root, 'derivatives/preproc_meg-mne_mri-fmriprep')
            for sub in os.listdir(derivatives_root):
                if not sub.startswith('sub-'): continue
                df = pd.read_csv(os.path.join(derivatives_root, sub, f'ses-movie/{sub}_ses-movie_scans.tsv'), sep='\t')
                date = df['acq_time'].values[0][:10].replace('-', '')
                er_fname = os.path.join(
                    bids_root,
                    f'sub-emptyroom/ses-{date}',
                    'meg',
                    f'sub-emptyroom_ses-{date}_task-noise_meg.ds'
                )
                if os.path.exists(er_fname):
                    raw_er = read_raw_ctf(er_fname).apply_gradient_compensation(0)
                    raw_er = raw_er.load_data().filter(l_freq=None, h_freq=LOWPASS_FILTER, n_jobs=N_JOBS, verbose=False)
                    raw_er = raw_er.resample(sfreq=RESAMPLE_SFREQ, npad='auto', n_jobs=N_JOBS)
                    raw_er = _raw_MaxFilter(raw_er, calibration=None, cross_talk=None, if_tsss=IF_TSSS, coord_frame='meg')
                    raw_er.load_data().notch_filter(np.arange(line_freq, line_freq * 7 + 1, line_freq))
                    noise_cov = mne.compute_raw_covariance(raw_er, 
                                                           tmin=0, tmax=None,
                                                           picks='meg', reject=dict(mag=reject['mag']),
                                                           method='auto', cv=5)
                else:
                    raw = read_raw_fif(os.path.join(derivatives_root, f'{sub}/ses-movie/meg/{sub}_ses-movie_task-movie_run-01_meg.fif'))
                    ch_names = np.asarray(raw.info['ch_names'])[np.asarray(raw.get_channel_types()) == 'mag'].tolist()
                    noise_cov = mne.Covariance(data=np.eye(len(ch_names)), names=ch_names, bads=[], projs=[], nfree=1)
                cov_root = os.path.join(derivatives_root, f'{sub}/ses-movie/meg/noise_cov')
                if not os.path.exists(cov_root):
                    os.mkdir(cov_root)
                mne.write_cov(os.path.join(cov_root, 'noise-cov.fif'), noise_cov, overwrite=True)
            ## Perform notch filter on task recordings
            bids_paths = find_matching_paths(
                derivatives_root, datatypes='meg', suffixes='meg', sessions=None, ignore_json=True
            )
            for bids_path in bids_paths:
                if 'sub-emptyroom' in bids_path.basename: continue
                sub, session = bids_path.basename.split('_')[0], bids_path.basename.split('_')[1]
                raw = read_raw_bids(bids_path, extra_params=dict(preload=True))
                raw = _pre_processing(raw, line_freq, N_JOBS, if_resample=False, if_lowpass=False, picks_meg='mag')
                raw.save(
                    os.path.join(derivatives_root, f'{sub}/{session}/meg/{bids_path.basename}'),
                    overwrite=True
                )
            ## We don't need emptyroom anymore
            cmd = f'rm -rf {os.path.join(bids_root, "sub-emptyroom")}'
            os.system(cmd)
        elif dataset == 'ds004212':
            ## This dataset is a CTF dataset.
            ## THINGS-MEG provides preprocessed data and saved them as epoch files.
            ## However, they combined all the epochs together ignoring the head movement among trials.
            ## They also downsampled the data from 4800 Hz to 200 Hz and filtered it with a 40 Hz low-pass filter.
            ## In this situation, we would rather perform MaxFilter, SSS and preprocessing procedures on raw data.
            ## But they did not provide all the required emptyroom data; a significant amount of data is missing.
            ## One solution is to set the noise covariance as an identity matrix.
            ## Another solution is to use the baseline of epochs data for noise estimtion, ignoring the head movement among trials.
            ## Here we chose the former solution. 
            line_freq = 60
            raw = read_raw_ctf(os.path.join(bids_root, 'sub-emptyroom/ses-20190501/meg/sub-emptyroom_ses-20190501_run-01.ds'))
            ch_names = np.asarray(raw.info['ch_names'])[np.asarray(raw.get_channel_types()) == 'mag'].tolist()

            for sub in os.listdir(bids_root):
                if not sub.startswith('sub-') or 'sub-emptyroom' in sub: continue
                for session in os.listdir(os.path.join(bids_root, sub)):
                    if not session.startswith('ses-') or 'MRI' in session: continue
                    ## Perform MaxFilter on emptyroom recordings and compute noise covariance
                    df = pd.read_csv(os.path.join(bids_root, sub, f'{session}/{sub}_{session}_scans.tsv'), sep='\t')
                    date = df['acq_time'].values[0][:10].replace('-', '')
                    er_fname = os.path.join(
                        bids_root,
                        f'sub-emptyroom/ses-{date}',
                        'meg',
                        f'sub-emptyroom_ses-{date}_run-01.ds'
                    )
                    if os.path.exists(er_fname):
                        raw_er = read_raw_ctf(er_fname).apply_gradient_compensation(0)
                        raw_er = raw_er.load_data().filter(l_freq=None, h_freq=LOWPASS_FILTER, n_jobs=N_JOBS, verbose=False)
                        raw_er = raw_er.resample(sfreq=RESAMPLE_SFREQ, npad='auto', n_jobs=N_JOBS)
                        raw_er = _raw_MaxFilter(raw_er, calibration=None, cross_talk=None, if_tsss=IF_TSSS, coord_frame='meg')
                        raw_er.load_data().notch_filter(np.arange(line_freq, line_freq * 7 + 1, line_freq))
                        noise_cov = mne.compute_raw_covariance(raw_er, 
                                                               tmin=0, tmax=None,
                                                               picks='meg', reject=dict(mag=reject['mag']),
                                                               method='auto', cv=5)
                    else:
                        noise_cov = mne.Covariance(data=np.eye(len(ch_names)), names=ch_names, bads=[], projs=[], nfree=1)
                    cov_root = os.path.join(bids_root, f'{sub}/{session}/meg/noise_cov')
                    if not os.path.exists(cov_root):
                        os.mkdir(cov_root)
                    mne.write_cov(os.path.join(cov_root, 'noise-cov.fif'), noise_cov, overwrite=True)
                    ## Perform MaxFilter on task recordings
                    fnames = glob.glob(os.path.join(bids_root, f'{sub}/{session}/meg/{sub}_{session}_task-main_run-*_meg.ds'))
                    for fname in fnames:
                        basename = fname.split('/')[-1].replace('.ds', '.fif')
                        try:
                            raw = read_raw_ctf(fname).apply_gradient_compensation(0)
                            raw = raw.load_data().filter(l_freq=None, h_freq=LOWPASS_FILTER, n_jobs=N_JOBS, verbose=False)
                            raw = raw.resample(sfreq=RESAMPLE_SFREQ, npad='auto', n_jobs=N_JOBS)
                            raw = _raw_MaxFilter(raw, calibration=None, cross_talk=None, if_tsss=IF_TSSS, coord_frame='meg')
                        except:
                            continue
                        raw = _pre_processing(raw, line_freq, N_JOBS, picks_meg='mag')
                        raw.save(
                            os.path.join(bids_root, f'{sub}/{session}/meg/{basename}'),
                            overwrite=True
                        )
                    failed_maxfilter += fnames
            del_files = failed_maxfilter
            for file in del_files:
                os.system('rm -rf {0}'.format(file))
            ## We don't need emptyroom anymore
            cmd = f'rm -rf {os.path.join(bids_root, "sub-emptyroom")}'
            os.system(cmd)
        elif dataset == 'ds006334':
            ## This dataset provides task data processed by MAXFilter and SSS.
            ## We only need to perform preprocessing.
            ## This dataset provides no emptyroom recordings and the time period before stimulus is too short.
            ## Here, we would use an ientity matrix for noise control.
            ## ATTENTION: Some of the .fif files could not be read since they are incomplete!
            line_freq = 50
            raw = read_raw_fif(os.path.join(bids_root, 'sub-01/meg/sub-01_task-AVspeech_run-1_meg.fif'))
            ch_names = list()
            for c in raw.info['ch_names']:
                if c.startswith('MEG'): ch_names.append(c)

            for sub in os.listdir(bids_root):
                if not sub.startswith('sub-'): continue
                noise_cov = mne.Covariance(data=np.eye(len(ch_names)), names=ch_names, bads=[], projs=[], nfree=1)
                cov_root = os.path.join(bids_root, f'{sub}/meg/noise_cov')
                if not os.path.exists(cov_root):
                    os.mkdir(cov_root)
                mne.write_cov(os.path.join(cov_root, 'noise-cov.fif'), noise_cov, overwrite=True)
            # Perform preprocessing procedures for task recordings
            bids_paths = find_matching_paths(
                bids_root, datatypes='meg', suffixes='meg', sessions=None, ignore_json=True
            )
            for bids_path in bids_paths:
                sub = bids_path.basename.split('_')[0]
                try:
                    raw = read_raw_bids(bids_path, extra_params=dict(preload=True))
                except:
                    failed_maxfilter.append(f'./data/bids/ds006334/{sub}/meg/{bids_path.basename}')
                    continue
                if raw.times[-1] < 60:
                    failed_maxfilter.append(f'./data/bids/ds006334/{sub}/meg/{bids_path.basename}')
                    continue
                raw = _pre_processing(raw, line_freq, N_JOBS)
                raw.save(
                    os.path.join(bids_root, f'{sub}/meg/{bids_path.basename}'),
                    overwrite=True
                )
            del_files = failed_maxfilter
            for file in del_files:
                os.system('rm -rf {0}'.format(file))
        
        logging.info(f'--- Processing Dataset: {dataset} Completed ---')
    
    
    # Perform pre-processing procedures for pretrain datasets from Zhangjiang Imaging Center
    for dataset in zic_datasets:
        logging.info(f'--- Processing Dataset: {dataset} ---')
        bids_root = os.path.join(target_root, dataset)
        
        # Perform MaxFilter, SSS and preprocessing
        failed_maxfilter = []
        if dataset == 'ywjaulg':
            ## This dataset provides emptyroom data and task data, both processed by MAXFilter and tSSS.
            ## We only need to perform preprocessing and noise covariance estimation.
            line_freq = 50
            for sub in os.listdir(bids_root):
                if not sub.startswith('sub-') or 'sub-emptyroom' in sub: continue
                df = pd.read_csv(os.path.join(bids_root, sub, f'ses-01/{sub}_ses-01_scans.tsv'), sep='\t')
                date = df['acq_time'].values[0][:10].replace('-', '')
                er_fname = os.path.join(
                    bids_root,
                    f'sub-emptyroom/ses-{date}',
                    'meg',
                    f'sub-emptyroom_ses-{date}_task-noise_meg.fif'
                )
                raw_er = read_raw_fif(er_fname)
                raw_er = raw_er.load_data().filter(l_freq=None, h_freq=LOWPASS_FILTER, n_jobs=N_JOBS, verbose=False)
                raw_er = raw_er.resample(sfreq=RESAMPLE_SFREQ, npad='auto', n_jobs=N_JOBS)
                raw_er.load_data().notch_filter(np.arange(line_freq, line_freq * 7 + 1, line_freq))
                noise_cov = mne.compute_raw_covariance(raw_er, 
                                                        tmin=0, tmax=None,
                                                        picks='meg', reject=reject,
                                                        method='auto', cv=5)
                cov_root = os.path.join(bids_root, f'{sub}/ses-01/meg/noise_cov')
                if not os.path.exists(cov_root):
                    os.mkdir(cov_root)
                mne.write_cov(os.path.join(cov_root, 'noise-cov.fif'), noise_cov, overwrite=True)
            ## Perform preprocessing on task recordings
            bids_paths = find_matching_paths(
                bids_root, datatypes='meg', suffixes='meg', sessions=None, ignore_json=True
            )
            for bids_path in bids_paths:
                if 'sub-emptyroom' in bids_path.basename: continue
                sub, session = bids_path.basename.split('_')[0], bids_path.basename.split('_')[1]
                fname = os.path.join(bids_root, f'{sub}/{session}/meg/{bids_path.basename}')
                raw = read_raw_fif(fname, preload=True)
                raw = _pre_processing(raw, line_freq, N_JOBS)
                raw.save(
                    os.path.join(bids_root, f'{sub}/{session}/meg/{bids_path.basename}'),
                    overwrite=True
                )
            ## We don't need emptyroom anymore
            cmd = f'rm -rf {os.path.join(bids_root, "sub-emptyroom")}'
            os.system(cmd)
        
        logging.info(f'--- Processing Dataset: {dataset} Completed ---')


    # Perform pre-processing procedures for downstream datasets from OpenNeuro
    for dataset in dt_datasets:
        if not dataset.startswith('ds'): continue
        logging.info(f'--- Processing Dataset: {dataset} ---')
        bids_root = os.path.join(target_root, dataset)
        
        # Perform MaxFilter, SSS and preprocessing
        failed_maxfilter = []
        if dataset == 'ds000117':
            line_freq = 50
            bids_root = os.path.join(bids_root, 'derivatives/meg_derivatives')
            raw = read_raw_fif(os.path.join(bids_root, 'sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_proc-sss_meg.fif'))
            ch_names = list()
            for c in raw.info['ch_names']:
                if c.startswith('MEG'): ch_names.append(c)
            for sub in os.listdir(bids_root):
                if 'sub-emptyroom' in sub: continue
                # Estimation of noise covariance
                with open(os.path.join(bids_root, f'{sub}/ses-meg/{sub}_ses-meg_task-facerecognition_proc-sss_meg.json'), 'r') as file:
                    data = json.load(file)
                er_fname = os.path.join(bids_root, data['AssociatedEmptyRoom'])
                er_fname = er_fname.replace('_meg.fif', '_proc-sss_meg.fif')
                if os.path.exists(er_fname):
                    raw_er = read_raw_fif(er_fname)
                    raw_er = raw_er.load_data().filter(l_freq=None, h_freq=LOWPASS_FILTER, n_jobs=N_JOBS, verbose=False)
                    raw_er = raw_er.resample(sfreq=RESAMPLE_SFREQ, npad='auto', n_jobs=N_JOBS)
                    raw_er.load_data().notch_filter(np.arange(line_freq, line_freq * 7 + 1, line_freq))             
                    noise_cov = mne.compute_raw_covariance(raw_er, 
                                                           tmin=0, tmax=None,
                                                           picks='meg', reject=reject,
                                                           method='auto', cv=5)
                else:
                    noise_cov = mne.Covariance(data=np.eye(len(ch_names)), names=ch_names, bads=[], projs=[], nfree=1)
                cov_root = os.path.join(bids_root, f'{sub}/ses-meg/meg/noise_cov')
                if not os.path.exists(cov_root):
                    os.mkdir(cov_root)
                mne.write_cov(os.path.join(cov_root, 'noise-cov.fif'), noise_cov, overwrite=True)
            # Perform preprocessing procedures for task recordings
            bids_paths = find_matching_paths(
                bids_root, datatypes='meg', suffixes='meg', sessions=None, ignore_json=True
            )
            for bids_path in bids_paths:
                sub = bids_path.basename.split('_')[0]
                if 'sub-emptyroom' in sub: continue
                try:
                    raw = read_raw_bids(bids_path, extra_params=dict(preload=True))
                except:
                    failed_maxfilter.append(os.path.join(bids_root, f'{sub}/ses-meg/meg/{bids_path.basename}'))
                    continue
                raw = _pre_processing(raw, line_freq, N_JOBS, if_resample=False)
                raw.save(
                    os.path.join(bids_root, f'{sub}/ses-meg/meg/{bids_path.basename}'),
                    overwrite=True
                )
            del_files = failed_maxfilter
            for file in del_files:
                os.system('rm -rf {0}'.format(file))
            ## We don't need emptyroom anymore
            cmd = f'rm -rf {os.path.join(bids_root, "sub-emptyroom")}'
            os.system(cmd)

        logging.info(f'--- Processing Dataset: {dataset} Completed ---')