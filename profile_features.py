import time
import numpy as np
import pandas as pd
import warnings
from typing import Dict, List
import os

# Suppress warnings
warnings.filterwarnings("ignore")

from eeg_feature_extraction.config import Config
from eeg_feature_extraction.features.time_domain import TimeDomainFeatures
from eeg_feature_extraction.features.frequency_domain import FrequencyDomainFeatures
from eeg_feature_extraction.features.connectivity import ConnectivityFeatures
from eeg_feature_extraction.features.network import NetworkFeatures
from eeg_feature_extraction.features.complexity import ComplexityFeatures
from eeg_feature_extraction.features.microstate import MicrostateFeatures
from eeg_feature_extraction.psd_computer import PSDComputer

try:
    import cupy as cp
    HAS_CUPY = True
    print(f"CuPy available. Device: {cp.cuda.runtime.getDeviceCount()} devices")
except ImportError:
    HAS_CUPY = False
    print("CuPy NOT available.")

def generate_dummy_data(n_channels=19, duration=10, fs=200):
    n_points = int(duration * fs)
    return np.random.randn(n_channels, n_points).astype(np.float32)

def profile_module(module_class, name, data, config, psd_result=None, **kwargs):
    # Instantiate
    try:
        extractor = module_class(config)
    except Exception as e:
        print(f"[{name}] Initialization failed: {e}")
        return None

    # Warmup (if GPU)
    if config.use_gpu and HAS_CUPY:
        try:
            extractor.compute(data, psd_result=psd_result, **kwargs)
            cp.cuda.Stream.null.synchronize()
        except Exception:
            pass

    # Profile
    start_time = time.time()
    try:
        extractor.compute(data, psd_result=psd_result, **kwargs)
        if config.use_gpu and HAS_CUPY:
            cp.cuda.Stream.null.synchronize()
    except Exception as e:
        print(f"[{name}] Computation failed: {e}")
        return None
    end_time = time.time()
    
    return end_time - start_time

def run_profile():
    # Standard 19 electrodes (10-20 system)
    channel_names = [
        'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 
        'T7', 'C3', 'CZ', 'C4', 'T8', 
        'P7', 'P3', 'PZ', 'P4', 'P8', 
        'O1', 'O2'
    ]
    n_channels = len(channel_names)
    duration = 10 # seconds
    fs = 200
    
    print(f"Generating dummy data: {n_channels} channels, {duration}s, {fs}Hz...")
    data = generate_dummy_data(n_channels, duration, fs)
    
    # Base Config
    config_cpu = Config()
    config_cpu.use_gpu = False
    config_cpu.sampling_rate = fs
    config_cpu.n_channels = n_channels
    config_cpu.update_from_electrode_names(channel_names)
    # Ensure complexity doesn't spawn too many threads during profiling
    config_cpu.complexity_n_workers = 1 
    
    config_gpu = Config()
    config_gpu.use_gpu = True
    config_gpu.sampling_rate = fs
    config_gpu.n_channels = n_channels
    config_gpu.update_from_electrode_names(channel_names)
    config_gpu.complexity_n_workers = 1

    results = []

    modules = [
        ("TimeDomain", TimeDomainFeatures),
        ("FrequencyDomain", FrequencyDomainFeatures),
        ("Connectivity", ConnectivityFeatures),
        ("Network", NetworkFeatures),
        ("Complexity", ComplexityFeatures),
        ("Microstate", MicrostateFeatures),
    ]

    # Pre-compute PSD for modules that need it
    print("Pre-computing PSD (CPU)...")
    psd_computer_cpu = PSDComputer(fs, use_gpu=False)
    psd_cpu = psd_computer_cpu.compute_psd(data)
    
    psd_gpu = None
    if HAS_CUPY:
        print("Pre-computing PSD (GPU)...")
        psd_computer_gpu = PSDComputer(fs, use_gpu=True)
        psd_gpu = psd_computer_gpu.compute_psd(data)

    print("\nStarting Profiling...")
    print(f"{'Module':<20} | {'CPU (s)':<10} | {'GPU (s)':<10} | {'Speedup':<10}")
    print("-" * 60)

    for name, module_cls in modules:
        # Profile CPU
        # Note: Network and Connectivity might need psd_result
        # Microstate might need analyzer or segments, we'll let it fit on current segment
        
        kwargs_cpu = {}
        if name in ["FrequencyDomain", "Connectivity", "Network", "Complexity"]:
            kwargs_cpu['psd_result'] = psd_cpu
        
        cpu_time = profile_module(module_cls, name, data, config_cpu, **kwargs_cpu)
        
        # Profile GPU
        gpu_time = None
        if HAS_CUPY:
            kwargs_gpu = {}
            if name in ["FrequencyDomain", "Connectivity", "Network", "Complexity"]:
                kwargs_gpu['psd_result'] = psd_gpu
            
            gpu_time = profile_module(module_cls, name, data, config_gpu, **kwargs_gpu)
        
        # Format output
        cpu_str = f"{cpu_time:.4f}" if cpu_time is not None else "Fail"
        gpu_str = f"{gpu_time:.4f}" if gpu_time is not None else "N/A"
        
        if cpu_time and gpu_time:
            speedup = f"{cpu_time / gpu_time:.2f}x"
        else:
            speedup = "-"
            
        print(f"{name:<20} | {cpu_str:<10} | {gpu_str:<10} | {speedup:<10}")
        
        results.append({
            "Module": name,
            "CPU": cpu_time,
            "GPU": gpu_time
        })

if __name__ == "__main__":
    run_profile()
