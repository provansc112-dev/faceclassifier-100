import os
import time
import torch
import gc
import psutil
import numpy as np
import matplotlib.pyplot as plt
from thop import profile
import model as model_lib
import config

def total_cleanup():
    """Cleanup memory to prevent leaks during benchmarking."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    print("Memory cleaned.")

def run_full_profiling(net, task_name, device, img_size):
    print(f"\nProfiling {task_name} on {device.upper()}")
    net.to(device)
    net.eval()
    
    inputs = torch.randn(1, 3, img_size, img_size).to(device)
    
    # 1. FLOPs & Parameters
    if device == 'cpu':
        flops, params = profile(net, inputs=(inputs, ), verbose=False)
        print(f"Total FLOPs: {flops/1e9:.2f} G")
        print(f"Total Params: {params/1e6:.2f} M")

    # 2. Latency Measurement
    # Warmup
    with torch.no_grad():
        for _ in range(10): _ = net(inputs)
        
    start_time = time.time()
    iterations = 50
    with torch.no_grad():
        for _ in range(iterations):
            _ = net(inputs)
    
    latency = ((time.time() - start_time) / iterations) * 1000
    print(f"Avg Latency: {latency:.2f} ms")
    
    return latency

def run_scalability_analysis():
    print("\n🚀 Starting Scalability Analysis...")
    db_sizes = [100, 1000, 10000, 100000, 1000000, 10000000]
    device = config.DEVICE
    
    # Load Models
    model_inc = model_lib.load_trained_model('inception')
    model_swin = model_lib.load_trained_model('swin')
    
    models_to_test = {
        'Inception-V1': (model_inc, 160),
        'Swin-T': (model_swin, 112)
    }

    results = {}
    
    for name, (net, img_size) in models_to_test.items():
        # Baseline E2E
        latency_e2e = run_full_profiling(net, f"{name} E2E", device, img_size)
        
        test_n = 10000 # Adjust for Euclidean test
        feature = torch.randn(1, 512).to(device)
        db_sim = torch.randn(test_n, 512).to(device)
        
        with torch.no_grad():
            _ = torch.cdist(feature, db_sim) # Warmup
            start = time.time()
            for _ in range(30):
                dist = torch.cdist(feature, db_sim)
                _ = torch.min(dist, dim=1)
            unit_search_time = ((time.time() - start) / 30) * 1000

        euclidean_scaling = []
        for n in db_sizes:
            actual_search_time = unit_search_time * (n / test_n)
            euclidean_scaling.append(latency_e2e + actual_search_time)
            
        results[name] = {
            'e2e': [latency_e2e] * len(db_sizes),
            'euclidean': euclidean_scaling
        }
        total_cleanup()

    # Plotting Results
    plt.figure(figsize=(12, 7))
    colors = {'Inception-V1': '#1f77b4', 'Swin-T': '#2ca02c'}
    
    for name, data in results.items():
        plt.plot(db_sizes, data['e2e'], label=f'{name} Inference Only', color=colors[name], lw=2)
        plt.plot(db_sizes, data['euclidean'], label=f'{name} + Euclidean Search', 
                 color=colors[name], linestyle='--', marker='o')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Database Size (Class)')
    plt.ylabel('Latency (ms)')
    plt.title('Scalability Analysis: End-To-End vs Euclidean Distance')
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig('scalability_results.png')
    print("\nBenchmark finished. Chart saved as 'scalability_results.png'")
    plt.show()

if __name__ == "__main__":
    try:
        run_scalability_analysis()
    finally:
        total_cleanup()