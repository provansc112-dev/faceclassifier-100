import os
import time
import torch
import torch.nn as nn
import gc
import psutil
import numpy as np
import matplotlib.pyplot as plt
import copy
import config
import model as model_lib

# --- 1. SETUP & UTILS ---
try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False

def get_file_size(file_path):
    if os.path.exists(file_path):
        return os.path.getsize(file_path) / (1024 * 1024)
    return 0.0

def get_ram_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def total_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

class EuclideanWrapper(nn.Module):
    def __init__(self, feature_extractor, db_size=10000, feature_dim=512, device='cpu'):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.register_buffer('db', torch.randn(db_size, feature_dim).to(device))
        
    def forward(self, x):
        emb = self.feature_extractor(x)
        if isinstance(emb, (tuple, list)):
            emb = emb[0]
        if emb.dim() == 3:
            emb = emb.mean(dim=1)
        elif emb.dim() == 4:
            emb = emb.mean(dim=[2, 3])
        if emb.shape[1] > self.db.shape[1]:
            emb = emb[:, :self.db.shape[1]]
        dist = torch.cdist(emb, self.db)
        return torch.min(dist, dim=1)

def print_stats_box(title, disk_size, params, flops, latency, fps, vram, ram):
    print(f"\n🚀 PROFILING {title}")
    print("="*45)
    print(f"📁 Model Disk Space  : {disk_size:.2f} MB")
    print(f"🛠️  Total Parameters  : {params:.2f} Million")
    flops_str = f"{flops:.2f} GFLOPs" if flops > 0 else "N/A"
    print(f"⚡ Complexity        : {flops_str}")
    print(f"⏱️  Inference Latency : {latency:.2f} ms / image")
    print(f"🚀 Throughput (FPS)  : {fps:.2f} images/sec")
    if vram > 0:
        print(f"📟 Peak VRAM (GPU)   : {vram:.2f} MB")
    print(f"💻 RAM Usage (CPU)   : {ram:.2f} MB")
    print("="*45)

def run_detailed_profiling(net, task_name, device, img_size, weights_path):
    device_str = str(device).split(':')[0]
    net.to(device)
    net.eval()
    inputs = torch.randn(1, 3, img_size, img_size).to(device)
    
    # Static Metrics
    disk_size = get_file_size(weights_path)
    params_count = sum(p.numel() for p in net.parameters()) / 1e6
    flops_count = 0.0
    if THOP_AVAILABLE and 'cpu' in device_str: 
        try:
            inputs_thop = inputs.clone().cpu()
            net_thop = copy.deepcopy(net).cpu()
            flops, _ = profile(net_thop, inputs=(inputs_thop, ), verbose=False)
            flops_count = flops / 1e9
            del net_thop, inputs_thop
        except: pass 

    # Benchmark
    total_cleanup()
    with torch.no_grad():
        for _ in range(5): _ = net(inputs) # Warmup
    
    start_time = time.time()
    iterations = 20 if device_str == 'cpu' else 50
    if 'cuda' in device_str: torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for _ in range(iterations):
            _ = net(inputs)
            if 'cuda' in device_str: torch.cuda.synchronize()
    
    end_time = time.time()
    latency = ((end_time - start_time) / iterations) * 1000
    fps = 1000 / latency if latency > 0 else 0
    ram_usage = get_ram_usage()
    vram_usage = torch.cuda.max_memory_allocated() / (1024 * 1024) if 'cuda' in device_str else 0.0

    full_title = f"{task_name} ({device_str.upper()})"
    print_stats_box(full_title, disk_size, params_count, flops_count, latency, fps, vram_usage, ram_usage)
    return latency

# --- 2. MAIN ANALYSIS LOGIC ---

def run_scalability_analysis():
    print("STARTING FULL BENCHMARK (Profiling: ALL | Plot: GPU Only)")
    
    db_sizes = [100, 1000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]
    
    # 1. TEST BOTH DEVICES (For Console Output)
    devices_to_test = ['cpu']
    if torch.cuda.is_available():
        devices_to_test.insert(0, 'cuda') 
    
    print("\nLoading Models...")
    try:
        model_inc = model_lib.load_trained_model('inception', load_best=False)
        model_swin = model_lib.load_trained_model('swin', load_best=False)
        path_inc = config.MODEL_CONFIGS['inception']['weights']
        path_swin = config.MODEL_CONFIGS['swin']['weights']
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    models_conf = [
        ('Inception-V1', model_inc, 160, path_inc),
        ('Swin-T', model_swin, 112, path_swin)
    ]

    results = {}

    # --- EXECUTION LOOP (Run for CPU & GPU) ---
    for device in devices_to_test:
        print(f"\n{'>'*20} TESTING ON {device.upper()} {'<'*20}")
        
        for name, net, img_size, w_path in models_conf:
            total_cleanup()
            
            # A. PROFILE E2E (Baseline)
            net_e2e = copy.deepcopy(net)
            lat_e2e = run_detailed_profiling(net_e2e, f"{name} E2E", device, img_size, w_path)
            del net_e2e
            total_cleanup()
            
            # B. PREPARE FEATURE EXTRACTOR
            feature_extractor = None
            if 'Inception' in name:
                feature_extractor = copy.deepcopy(net.base_model if hasattr(net, 'base_model') else net)
            elif 'Swin' in name:
                feature_extractor = copy.deepcopy(net)
                if hasattr(feature_extractor, 'reset_classifier'): feature_extractor.reset_classifier(0)

            feature_extractor.to(device).eval()
            
            # Detect Dimension
            dummy_in = torch.randn(1, 3, img_size, img_size).to(device)
            with torch.no_grad():
                out = feature_extractor(dummy_in)
                if isinstance(out, (tuple, list)): out = out[0]
                if out.dim() == 3: out = out.mean(dim=1)
                elif out.dim() == 4: out = out.mean(dim=[2,3])
                actual_dim = out.shape[1]

            # C. PROFILE EUCLIDEAN SEARCH
            wrapper = EuclideanWrapper(feature_extractor, db_size=10000, feature_dim=actual_dim, device=device)
            lat_euclidean_snap = run_detailed_profiling(wrapper, f"{name} EUCLIDEAN", device, img_size, w_path)
            
            # D. CALCULATION
            search_time_10k = max(0.001, lat_euclidean_snap - lat_e2e)
            unit_search_time = search_time_10k 
            
            total_latencies_euclidean = []
            total_latencies_e2e = [] 
            
            for n in db_sizes:
                search_component = unit_search_time * (n / 10000)
                total_latencies_euclidean.append(lat_e2e + search_component)
                total_latencies_e2e.append(lat_e2e) 
            
            # Key format: "Inception-V1 (CUDA)" or "Inception-V1 (CPU)"
            key = f"{name} ({device.upper()})"
            results[key] = {
                'e2e': total_latencies_e2e,
                'euclidean': total_latencies_euclidean
            }
            
            del wrapper, feature_extractor
            total_cleanup()

    # --- 4. PLOTTING (FILTER: GPU Only) ---
    print("\n🎨 Generating Scalability Plot (GPU Data Only)...")
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 8))
    
    colors = {'Inception-V1': '#1f77b4', 'Swin-T': '#2ca02c'}
    
    plotted_count = 0
    for key, data in results.items():
        # FILTER: Skip if CPU data
        if 'CPU' in key:
            continue
            
        # Parse name from "Inception-V1 (CUDA)" -> "Inception-V1"
        model_name = key.split('(')[0].strip()
        base_color = colors[model_name]
        plotted_count += 1
        
        # 1. E2E Baseline (Dashed)
        plt.plot(db_sizes, data['e2e'], 
                 label=f'{model_name} - E2E (Baseline)', 
                 color=base_color, linestyle='--', linewidth=2, alpha=0.8)
        
        # 2. Euclidean (Solid + Marker)
        plt.plot(db_sizes, data['euclidean'], 
                 label=f'{model_name} - Euclidean Search', 
                 color=base_color, linestyle='-', marker='o', linewidth=2.5)

    if plotted_count == 0:
        print("⚠️ Warning: No GPU results found to plot. (Did you run on a CPU-only machine?)")

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Database Size (Identities)', fontsize=14)
    plt.ylabel('Total Latency (ms) [Log Scale]', fontsize=14)
    plt.title('GPU Scalability: End-to-End vs Euclidean Search', fontsize=18)
    
    plt.legend(fontsize=12)
    plt.grid(True, which="both", alpha=0.3)
    
    save_path = os.path.join(config.BASE_DIR, 'scalability_gpu_only.png')
    plt.savefig(save_path)
    print(f"✅ Benchmark finished. Chart saved to: {save_path}")

if __name__ == "__main__":
    try:
        run_scalability_analysis()
    finally:
        total_cleanup()