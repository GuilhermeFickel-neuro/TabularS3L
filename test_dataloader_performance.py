#!/usr/bin/env python3
"""
Performance test script to monitor dataloader efficiency and GPU utilization
"""

import time
import torch
import argparse
import psutil
import threading
import gc
import os
import signal
import multiprocessing as mp
import resource
from collections import deque
from sklearn.model_selection import train_test_split

# Import your modules
from switchtab_matryoshka import (
    PaperExactSwitchTab, create_paper_exact_transformer_config
)
from ts3l.utils.embedding_utils import FTEmbeddingConfig
from ts3l.utils.switchtab_utils import SwitchTabConfig, SwitchTabDataset, SwitchTabFirstPhaseCollateFN
from ts3l.utils import TS3LDataModule, get_category_cardinality
from benchmark.datasets import load_higgs


def increase_file_limits():
    """Temporarily increase file descriptor limits for this process"""
    try:
        # Get current limits
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        print(f"Current file descriptor limits: soft={soft}, hard={hard}")
        
        # Set to maximum possible
        new_soft = min(hard, 4096)  # Try to increase to 4096 or hard limit
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
        
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        print(f"Updated file descriptor limits: soft={soft}, hard={hard}")
        return True
    except Exception as e:
        print(f"Failed to increase file limits: {e}")
        return False


def get_open_files_count():
    """Get current number of open file descriptors"""
    try:
        pid = os.getpid()
        return len(os.listdir(f'/proc/{pid}/fd'))
    except:
        return -1


def kill_orphaned_processes():
    """Kill any orphaned worker processes"""
    try:
        current_pid = os.getpid()
        parent = psutil.Process(current_pid)
        
        # Get all child processes
        children = parent.children(recursive=True)
        for child in children:
            try:
                if 'python' in child.name().lower():
                    print(f"Terminating orphaned process: {child.pid}")
                    child.terminate()
                    try:
                        child.wait(timeout=1)
                    except psutil.TimeoutExpired:
                        child.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except Exception as e:
        print(f"Warning during process cleanup: {e}")


def aggressive_cleanup(dataloader):
    """Aggressively clean up dataloader resources"""
    try:
        # First, try to access and shutdown any existing iterator
        if hasattr(dataloader, '_iterator') and dataloader._iterator is not None:
            try:
                # Get worker PIDs before shutdown
                worker_pids = []
                if hasattr(dataloader._iterator, '_workers'):
                    for worker in dataloader._iterator._workers:
                        if hasattr(worker, 'pid'):
                            worker_pids.append(worker.pid)
                
                dataloader._iterator._shutdown_workers()
                
                # Force kill any remaining worker processes
                for pid in worker_pids:
                    try:
                        os.kill(pid, signal.SIGTERM)
                        time.sleep(0.1)
                        os.kill(pid, signal.SIGKILL)
                    except (OSError, ProcessLookupError):
                        pass
                        
            except Exception as e:
                print(f"Warning during iterator shutdown: {e}")
            
            dataloader._iterator = None
            
        # Force iteration to stop if active
        if hasattr(dataloader, '_DataLoader__initialized'):
            dataloader._DataLoader__initialized = False
            
        # Clean up dataset references
        if hasattr(dataloader, 'dataset'):
            del dataloader.dataset
            
        # Clean up batch sampler
        if hasattr(dataloader, 'batch_sampler'):
            del dataloader.batch_sampler
            
        # Clean up sampler
        if hasattr(dataloader, 'sampler'):
            del dataloader.sampler
            
    except Exception as e:
        print(f"Warning during cleanup: {e}")
    
    # Kill any orphaned processes
    kill_orphaned_processes()
    
    # Force garbage collection
    del dataloader
    gc.collect()
    
    # Give system time to clean up
    time.sleep(0.5)


class GPUMonitor:
    """Monitor GPU utilization in a separate thread"""
    def __init__(self, interval=0.1):
        self.interval = interval
        self.running = False
        self.utilizations = deque(maxlen=1000)  # Keep last 1000 measurements
        self.thread = None
        
    def start(self):
        if torch.cuda.is_available():
            self.running = True
            self.thread = threading.Thread(target=self._monitor)
            self.thread.start()
            
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
            
    def _monitor(self):
        while self.running:
            try:
                # Get GPU utilization
                gpu_util = torch.cuda.utilization()
                self.utilizations.append(gpu_util)
                time.sleep(self.interval)
            except:
                time.sleep(self.interval)
                
    def get_stats(self):
        if not self.utilizations:
            return {"mean": 0, "max": 0, "min": 0, "idle_time": 100}
            
        utils = list(self.utilizations)
        mean_util = sum(utils) / len(utils)
        max_util = max(utils)
        min_util = min(utils)
        idle_time = sum(1 for u in utils if u < 10) / len(utils) * 100
        
        return {
            "mean": mean_util,
            "max": max_util, 
            "min": min_util,
            "idle_time": idle_time
        }


def benchmark_dataloader(dataloader, name, num_batches=50, warmup_batches=5):
    """Benchmark a dataloader's performance"""
    print(f"\n=== Benchmarking {name} ===")
    
    # Monitor file descriptors
    fds_before = get_open_files_count()
    if fds_before > 0:
        print(f"File descriptors before test: {fds_before}")
    
    # GPU monitoring
    gpu_monitor = GPUMonitor()
    gpu_monitor.start()
    
    # CPU monitoring
    cpu_times = []
    batch_times = []
    
    try:
        # Warmup
        print(f"Warming up...")
        for i, batch in enumerate(dataloader):
            if i >= warmup_batches:
                break
                
        print(f"Running benchmark for {num_batches} batches...")
        
        start_time = time.time()
        batch_start = time.time()
        
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
                
            # Simulate some GPU work
            if torch.cuda.is_available():
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    if isinstance(batch[0], torch.Tensor):
                        batch[0].cuda()
                    if isinstance(batch[1], torch.Tensor):
                        batch[1].cuda()
                
            batch_end = time.time()
            batch_times.append(batch_end - batch_start)
            cpu_times.append(psutil.cpu_percent(interval=None))
            batch_start = time.time()
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{num_batches} batches...")
                
        total_time = time.time() - start_time
        
    finally:
        gpu_monitor.stop()
    
    # Calculate statistics
    avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
    avg_cpu = sum(cpu_times) / len(cpu_times) if cpu_times else 0
    batches_per_sec = len(batch_times) / total_time if total_time > 0 else 0
    
    gpu_stats = gpu_monitor.get_stats()
    
    print(f"\n📊 {name} Performance Results:")
    print(f"  ⏱️  Average batch time: {avg_batch_time:.4f}s")
    print(f"  🚀 Batches per second: {batches_per_sec:.2f}")
    print(f"  🖥️  Average CPU usage: {avg_cpu:.1f}%")
    print(f"  🎮 GPU utilization:")
    print(f"     - Mean: {gpu_stats['mean']:.1f}%")
    print(f"     - Max: {gpu_stats['max']:.1f}%")
    print(f"     - Idle time: {gpu_stats['idle_time']:.1f}%")
    
    return {
        'avg_batch_time': avg_batch_time,
        'batches_per_sec': batches_per_sec,
        'avg_cpu': avg_cpu,
        'gpu_stats': gpu_stats
    }


def create_dataloader(num_workers, prefetch_factor, X_train, y_train, config, continuous_cols, category_cols):
    """Create a single dataloader configuration"""
    # Create dataset
    train_ds = SwitchTabDataset(X_train, y_train, config, 
                              continuous_cols=continuous_cols, 
                              category_cols=category_cols, 
                              is_second_phase=False)
    val_ds = SwitchTabDataset(X_train[:1000], y_train[:1000], config,
                            continuous_cols=continuous_cols,
                            category_cols=category_cols,
                            is_second_phase=False)
    
    # Create dataloader with reduced persistent_workers for high worker counts
    persistent = num_workers <= 8  # Only use persistent workers for lower worker counts
    
    dl = TS3LDataModule(train_ds, val_ds,
                      batch_size=128,
                      n_jobs=num_workers,
                      train_sampler="random",
                      train_collate_fn=SwitchTabFirstPhaseCollateFN(),
                      valid_collate_fn=SwitchTabFirstPhaseCollateFN(),
                      prefetch_factor=prefetch_factor,
                      persistent_workers=persistent,
                      pin_memory=True)
    
    dl.setup("fit")
    name = f"Workers:{num_workers}, Prefetch:{prefetch_factor}"
    return dl.train_dataloader(), name


def main():
    parser = argparse.ArgumentParser(description='Test dataloader performance')
    parser.add_argument('--max_workers', type=int, default=16, help='Maximum number of workers to test')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for testing')
    parser.add_argument('--num_batches', type=int, default=50, help='Number of batches to benchmark')
    args = parser.parse_args()
    
    print("🚀 Dataloader Performance Benchmark")
    print("=" * 50)
    
    # Increase file descriptor limits
    increase_file_limits()
    
    # Check initial file descriptor usage
    initial_fds = get_open_files_count()
    if initial_fds > 0:
        print(f"Initial file descriptors: {initial_fds}")
    
    # Load data
    print("Loading dataset...")
    data, label, continuous_cols, category_cols, output_dim, metric_name, metric_hparams = load_higgs()
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)
    X_train = X_train.iloc[:5000]  # Use subset for testing
    y_train = y_train.iloc[:5000]
    
    print(f"Using {len(X_train)} training samples")
    print(f"System info: {psutil.cpu_count()} CPU cores, GPU: {torch.cuda.is_available()}")
    
    # Create config
    embedding_config = FTEmbeddingConfig(
        input_dim=X_train.shape[1],
        emb_dim=256,
        cont_nums=len(continuous_cols),
        cat_cardinality=get_category_cardinality(X_train, category_cols),
        required_token_dim=2
    )
    backbone_config = create_paper_exact_transformer_config(d_model=256)
    
    config = SwitchTabConfig(
        task="classification",
        embedding_config=embedding_config,
        backbone_config=backbone_config,
        output_dim=output_dim,
        corruption_rate=0.3,
        loss_fn="CrossEntropyLoss",
        metric=metric_name,
    )
    
    # Test configurations
    cpu_count = psutil.cpu_count()
    num_workers_list = [x for x in range(0, min(args.max_workers, cpu_count), 2)]
    prefetch_factors = [1, 2, 3, 4, 5, 6, 7, 8]
    
    print(f"\n🧪 Testing worker counts: {num_workers_list}")
    print(f"🧪 Testing prefetch factors: {prefetch_factors}")
    
    # Benchmark each configuration
    results = []
    total_configs = len(num_workers_list) * len(prefetch_factors)
    current_config = 0
    
    for num_workers in num_workers_list:
        for prefetch_factor in prefetch_factors:
            current_config += 1
            print(f"\n🔄 Testing configuration {current_config}/{total_configs}: Workers={num_workers}, Prefetch={prefetch_factor}")
            
            # Monitor file descriptors before creating dataloader
            fds_before = get_open_files_count()
            if fds_before > 0:
                print(f"File descriptors before creation: {fds_before}")
                
                # Safety check - if we're getting close to limit, do extra cleanup
                if fds_before > 800:
                    print("⚠️  High file descriptor usage - doing extra cleanup")
                    kill_orphaned_processes()
                    gc.collect()
                    time.sleep(1)
            
            try:
                # Create dataloader on demand
                dataloader, name = create_dataloader(
                    num_workers, prefetch_factor,
                    X_train, y_train.values, config, 
                    continuous_cols, category_cols
                )
                
                result = benchmark_dataloader(dataloader, name, args.num_batches)
                result['name'] = name
                results.append(result)
                
            except Exception as e:
                print(f"❌ Failed to test {name}: {e}")
                # Do emergency cleanup
                kill_orphaned_processes()
                gc.collect()
                time.sleep(1)
                continue
                
            finally:
                # Aggressive cleanup
                try:
                    aggressive_cleanup(dataloader)
                except:
                    pass
                
                # Monitor file descriptors after cleanup
                fds_after = get_open_files_count()
                if fds_after > 0:
                    print(f"File descriptors after cleanup: {fds_after}")
                
                # Extra delay for higher worker counts
                delay = 3 if num_workers >= 12 else 2
                time.sleep(delay)
    
    # Final cleanup
    kill_orphaned_processes()
    gc.collect()
    
    # Print summary
    print("\n" + "=" * 60)
    print("📋 PERFORMANCE SUMMARY")
    print("=" * 60)
    
    if not results:
        print("❌ No successful tests completed!")
        return
    
    # Sort by batches per second
    results.sort(key=lambda x: x['batches_per_sec'], reverse=True)
    
    print(f"{'Configuration':<25} {'Batch/s':<10} {'GPU Mean%':<12} {'GPU Idle%':<12}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['name']:<25} {result['batches_per_sec']:<10.2f} "
              f"{result['gpu_stats']['mean']:<12.1f} {result['gpu_stats']['idle_time']:<12.1f}")
    
    # Find best configuration
    best = results[0]
    print(f"\n🏆 Best configuration: {best['name']}")
    print(f"   Achieves {best['batches_per_sec']:.2f} batches/sec")
    print(f"   GPU utilization: {best['gpu_stats']['mean']:.1f}% (idle: {best['gpu_stats']['idle_time']:.1f}%)")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if best['gpu_stats']['idle_time'] > 50:
        print("   - GPU is idle >50% of time - increase num_workers or prefetch_factor")
    elif best['gpu_stats']['idle_time'] < 10:
        print("   - Excellent GPU utilization!")
    else:
        print("   - Good GPU utilization, minor tweaks possible")
        
    if best['name'].startswith('Workers:0'):
        print("   - Single-threaded loading is fastest - your data processing is very light")
    else:
        workers = int(best['name'].split(',')[0].split(':')[1])
        print(f"   - Use {workers} workers for optimal performance")


if __name__ == "__main__":
    main() 
