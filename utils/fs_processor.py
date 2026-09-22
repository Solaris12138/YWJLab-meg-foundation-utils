import os
import glob
import tempfile
import re
import subprocess
import time
import logging


class ParallelFreeSurferProcessor:
    """FreeSurfer processor for parallel processing of multiple datasets"""
    
    def __init__(self, 
                 freesurfer_home='/usr/local/freesurfer/8.1.0', 
                 max_n_jobs=16):
        """
        Initialize the processor

        Parameters
        ----------
        freesurfer_home : str
            The environment variable of 'FREESURFER_HOME'.
        max_n_jobs : int
            Maximum number of CPUs used.

        """
        self.freesurfer_home = freesurfer_home
        self.max_n_jobs = max_n_jobs
        self.setup_environment()
    
    def setup_environment(self):
        """Setting up the FreeSurfer basic environment"""
        os.environ['FREESURFER_HOME'] = self.freesurfer_home
        os.environ['FSFAST_HOME'] = f'{self.freesurfer_home}/fsfast'
        os.environ['MNI_DIR'] = f'{self.freesurfer_home}/mni'
        
        # Check FreeSurfer in PATH
        fs_bin = f'{self.freesurfer_home}/bin'
        if fs_bin not in os.environ['PATH']:
            raise RuntimeError(f'The FreeSurfer executable is not in the PATH: {fs_bin}')
    
    def process_dataset(self, dataset_name, dataset_config):
        """
        For a single dataset, use parallel tools to process all the subjects.

        Parameters
        ----------
        dataset_name : str
            The name of dataset.
        dataset_config : dict
            Dataset configuration, including:
                - input_pattern: Expressions of input files in glob mode.
                - output_dir: The output directory, corresponding to 'SUBJECTS_DIR'.
                - subject_pattern: Regular expression pattern for extracting subject ID from path.

        """
        # Set the SUBJECTS_DIR of the current dataset.
        subjects_dir = dataset_config['output_dir']
        os.environ['SUBJECTS_DIR'] = subjects_dir
        
        # Ensure the output directory exists.
        os.makedirs(subjects_dir, exist_ok=True)
        
        print(f'SUBJECTS_DIR is set to: {subjects_dir}')
        print(f"Finding T1 files in the pattern of: {dataset_config['input_pattern']}")
        
        t1ws = glob.glob(dataset_config['input_pattern'])
        
        if not t1ws:
            print('No T1 files found!')
            return False   
        print(f'Found {len(t1ws)} T1 files')
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmpf:
            for t1w in t1ws:
                if 'subject_pattern' in dataset_config:
                    match = re.search(dataset_config['subject_pattern'], t1w)
                    if match:
                        sid = match.group(0)
                    else:
                        print(f'Warning: Unable to extract subject ID from path: {t1w}')
                        continue
                else:
                    # Default extraction mode: sub-xxx
                    match = re.search(r"sub-\w+", t1w)
                    sid = match.group(0) if match else os.path.basename(t1w).split('_')[0]
                
                subject_output_dir = os.path.join(subjects_dir, sid)
                if os.path.exists(subject_output_dir):
                    print(f'Skip processed subjects: {sid}')
                    continue
                
                tmpf.write(f'{t1w},{sid}\n')
            tmp_path = tmpf.name
        
        with open(tmp_path, 'r') as f:
            lines = f.readlines()      
        if not lines:
            print('No new subject needs to be processed.')
            os.unlink(tmp_path)
            return True
        
        n_jobs = min(len(lines), self.max_n_jobs)
        print(f'Use {n_jobs} parallel jobs to process {len(lines)} subjects.')
        
        # Build parallel command
        parallel_cmd = (
            f"cat {tmp_path} | parallel --colsep ',' --jobs {n_jobs} --eta "
            '"recon-all -i {1} -s {2} -all"'
        )
        print(f'Running command: {parallel_cmd}')
        
        # Recording start time
        start_time = time.time()
        
        try:
            process = subprocess.Popen(
                parallel_cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            
            for line in process.stdout:
                print(line.strip())
            
            process.wait()
            
            if process.returncode == 0:
                end_time = time.time()
                total_time = (end_time - start_time) / 3600
                print(f'✓ Dataset: {dataset_name} Completed. Total time spent: {total_time:.2f} h')
                success = True
            else:
                print(f'✗ Dataset: {dataset_name} Failed. Return code: {process.returncode}')
                success = False
                
        except Exception as e:
            print(f'Error executing the parallel command: {e}')
            success = False
        
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
        return success
    
    def run_sequential_datasets(self, datasets_config):
        """
        Sequential processing of multiple datasets, with parallel processing within each dataset.
        
        Parameters
        ----------
        datasets_config : dict of dict
            Dataset configuration, including:
                - input_pattern: Expressions of input files in glob mode.
                - output_dir: The output directory, corresponding to 'SUBJECTS_DIR'.
                - subject_pattern: Regular expression pattern for extracting subject ID from path.

        """
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler()]
        )
        
        results = {}
        
        for dataset_name, dataset_config in datasets_config.items():
            logging.info(f'--- Processing Dataset: {dataset_name} ---')
            
            # Check system resources
            if not self._check_system_resources():
                response = input("System resources may be insufficient. Continue? (y/N): ")
                if response.lower() != 'y':
                    logging.info(f'Processing terminated.')
                    break
            
            # Process the current dataset
            start_time = time.time()
            success = self.process_dataset(dataset_name, dataset_config)
            end_time = time.time()
            
            processing_time = (end_time - start_time) / 3600
            
            results[dataset_name] = {
                'success': success,
                'processing_time': processing_time
            }
            
            if success:
                logging.info(f"Dataset '{dataset_name}' Completed. Total time spent: {processing_time:.2f} h")
            else:
                logging.info(f"Dataset '{dataset_name}' Failed.")
            
            # Brief pause to prepare for the next dataset.
            if dataset_name != list(datasets_config.keys())[-1]:
                logging.info('Preparing to process the next dataset...')
                time.sleep(10)

        logging.info('All datasets have been processed!')
        self._print_summary(results)
    
    def _check_system_resources(self):
        """System resource check"""
        try:
            statvfs = os.statvfs('/')
            free_disk_gb = (statvfs.f_bavail * statvfs.f_frsize) / (1024**3)
            
            if free_disk_gb < 20:
                print(f'Warning: Low disk space, only {free_disk_gb:.1f} GB remaining.')
                return False
            
            print(f'Disk space: {free_disk_gb:.1f} GB available')
            return True
            
        except Exception as e:
            print(f'Resource check failed: {e}')
            return True
    
    def _print_summary(self, results):
        """Output processing summary"""
        successful_datasets = [name for name, result in results.items() if result['success']]
        failed_datasets = [name for name, result in results.items() if not result['success']]
        
        print(f'Successfully processed: {len(successful_datasets)} datasets')
        print(f'Processing failed: {len(failed_datasets)} datasets')
        
        if successful_datasets:
            print('\nSuccessful datasets:')
            for name in successful_datasets:
                time_str = f"{results[name]['processing_time']:.2f} h"
                print(f'  ✓ {name} ({time_str})')
        
        if failed_datasets:
            print('\nFailed datasets:')
            for name in failed_datasets:
                print(f'  ✗ {name}')