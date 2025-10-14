import argparse
import time
import multiprocessing
import sys
import psutil
from threading import Thread, Event
from pynvml import *
import torch
import signal


class SystemLoadGenerator:
    """
    SystemLoadGenerator: generate CPU, memory, and GPU load for testing.

    Args:
        memory_mb (int): Amount of RAM to allocate in megabytes.
        cpu_cores (int): Number of CPU worker processes to spawn.
        gpu_calc_intensity (int): Intensity level for GPU compute load (1-10).
        gpu_mem_intensity (int): Intensity level for GPU memory allocation (1-10).
    """

    def __init__(self, memory_mb=0, cpu_cores=0, gpu_calc_intensity=0, gpu_mem_intensity=0):
        self.memory_mb = memory_mb
        self.cpu_cores = cpu_cores
        self.gpu_calc_intensity = gpu_calc_intensity or 0
        self.gpu_mem_intensity = gpu_mem_intensity or 0

        self.stop_event = Event()
        self.processes = []
        self.threads = []
        self.global_memory = []

    # -------------------------
    # Workload generators
    # -------------------------
    def memory_load(self, mb):
        """Allocate a bytearray of the requested size (MB)."""
        bytes_amount = mb * 1024 * 1024
        return bytearray(bytes_amount)

    def cpu_worker(self, stop_event):
        """CPU-bound worker: runs a busy computation loop until stop_event is set."""
        while not stop_event.is_set():
            sum(i * i for i in range(10**7))

    def gpu_worker(self, intensity, mem_gb):
        """
        GPU worker: allocate GPU memory blocks and perform matrix multiplies
        to generate sustained GPU utilization until stop_event is set.

        Args:
            intensity (int): compute intensity multiplier.
            mem_gb (float): amount of GPU memory to allocate in GB.
        """
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            chunk_size = 2**20  # allocate in 1 MiB-ish chunks
            memory_blocks = []

            # Cap allocation to 90% of 12GB if requested value is too large
            mem_bytes = min(int(12 * 0.9 * 1024**3), int(mem_gb * 1024**3))
            for _ in range(0, mem_bytes, chunk_size):
                block = torch.zeros(chunk_size // 4, dtype=torch.float32, device=device)
                memory_blocks.append(block)

            # Matrix sizes scale with intensity
            size = max(64, 500 * intensity)
            a = torch.randn(size, size, device=device)
            b = torch.randn(size, size, device=device)

            while not self.stop_event.is_set():
                c = torch.mm(a, b)
                # Add small random noise to avoid deterministic optimization
                a = c * 0.9 + torch.randn_like(c) * 0.1
                b = c.t() * 0.9 + torch.randn_like(c.t()) * 0.1
                torch.cuda.synchronize()

        except ImportError:
            print("PyTorch is required for GPU load: pip install torch")
        except Exception as e:
            print(f"GPU worker error: {e}")

    # -------------------------
    # System sampling & utilities
    # -------------------------
    def get_system_stats(self):
        """Return a snapshot of CPU, memory and GPU utilization."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        sys_mem = psutil.virtual_memory()
        mem_used = sys_mem.used / (1024 ** 3)
        mem_total = sys_mem.total / (1024 ** 3)

        gpu_info = []
        try:
            nvmlInit()
            for i in range(nvmlDeviceGetCount()):
                handle = nvmlDeviceGetHandleByIndex(i)
                util = nvmlDeviceGetUtilizationRates(handle)
                mem = nvmlDeviceGetMemoryInfo(handle)
                gpu_info.append({
                    "gpu_util": util.gpu,
                    "mem_util": mem.used / mem.total * 100,
                    "mem_used": mem.used / (1024 ** 3),
                    "mem_total": mem.total / (1024 ** 3)
                })
            nvmlShutdown()
        except NVMLError:
            pass

        return {
            "cpu": cpu_percent,
            "memory": f"{mem_used:.1f}/{mem_total:.1f}GB ({sys_mem.percent}%)",
            "gpu": gpu_info
        }

    def get_available_resources(self):
        """
        Return available system resources:
          - available memory (MB)
          - approximate CPU idle percentage (0-100)
          - free GPU memory (MB) for the first GPU (if available)
        """
        mem = psutil.virtual_memory()
        mem_available_mb = mem.available / 1024 / 1024  # MB

        # CPU idle percent (approx)
        cpu_idle_percent = 100 - psutil.cpu_percent(interval=0.1)

        gpu_free_mem_list = []
        try:
            nvmlInit()
            for i in range(nvmlDeviceGetCount()):
                handle = nvmlDeviceGetHandleByIndex(i)
                mem_info = nvmlDeviceGetMemoryInfo(handle)
                free_mem_mb = mem_info.free / 1024 / 1024
                gpu_free_mem_list.append(free_mem_mb)
            nvmlShutdown()
        except NVMLError:
            gpu_free_mem_list = []

        first_gpu_free = gpu_free_mem_list[0] if gpu_free_mem_list else 0
        return {
            "mem_avail": round(mem_available_mb * 0.9, 2),
            "cpu_avail": round(cpu_idle_percent * 0.9, 2),
            "gpu_avail": round(first_gpu_free * 0.9, 2)
        }

    # -------------------------
    # Monitoring & sampling
    # -------------------------
    def monitor(self):
        """Continuously print a live system status until stopped."""
        try:
            while not self.stop_event.wait(1):
                stats = self.get_system_stats()
                # Clear the terminal and print a status header
                print("\033c", end="")

                print("=== Real-time System Monitor ===")
                print(f"CPU usage: {stats['cpu']}%")
                print(f"Memory usage: {stats['memory']}")
                if stats['gpu']:
                    for i, gpu in enumerate(stats['gpu']):
                        print(f"GPU{i} util: {gpu['gpu_util']}% | VRAM: {gpu['mem_used']:.1f}/{gpu['mem_total']:.1f}GB ({gpu['mem_util']:.1f}%)")
                else:
                    print("GPU info: not available")
                print("\nPress Ctrl+C to exit")
        except KeyboardInterrupt:
            self.stop_event.set()

    def _collect_resource_samples(self, resource_type='gpu', duration=10, interval=1.0):
        """
        Collect resource samples over a duration and return average values.

        Returns:
            (avg_resource, avg_memory_percent)
        """
        resource_samples = []
        memory_samples = []

        iterations = max(1, int(duration / interval))
        for _ in range(iterations):
            stats = self.get_system_stats()

            if resource_type == 'cpu':
                resource_val = stats['cpu']
            elif resource_type == 'gpu':
                if stats['gpu']:
                    # use the largest per-GPU memory usage as the representative value
                    resource_val = max(g['mem_used'] for g in stats['gpu'])
                else:
                    resource_val = 0
            else:
                raise ValueError("resource_type must be 'cpu' or 'gpu'")

            # parse memory percent from the string "X/YGB (Z%)"
            mem_percent = float(stats['memory'].split('(')[-1].strip('%) '))
            resource_samples.append(resource_val)
            memory_samples.append(mem_percent)
            time.sleep(interval)

        avg_resource = sum(resource_samples) / len(resource_samples)
        avg_memory = sum(memory_samples) / len(memory_samples)
        return avg_resource, avg_memory

    def detect_fluctuation_continuous(self, resource_type='gpu', duration=10, interval=10, threshold=0.1, notify_func=None):
        """
        Continuous resource fluctuation detector.

        Samples resource usage for `duration` seconds, waits `interval` seconds,
        and compares the new average to the previous average. If both the resource
        and memory change exceed `threshold` (fractional, e.g. 0.1 = 10%), a notification is triggered.

        Args:
            resource_type (str): 'cpu' or 'gpu'
            duration (int): sampling duration in seconds
            interval (int): seconds to wait between comparisons
            threshold (float): fractional threshold for change detection
            notify_func (callable): optional callback called as notify_func(changed: bool, message: str)
        """
        def monitor_loop():
            prev_resource, prev_memory = self._collect_resource_samples(resource_type, duration)

            while not self.stop_event.is_set():
                time.sleep(interval)
                curr_resource, curr_memory = self._collect_resource_samples(resource_type, duration)

                delta_resource = abs(curr_resource - prev_resource)
                delta_memory = abs(curr_memory - prev_memory)

                # Avoid division by zero: use prev_memory as denominator if non-zero, otherwise 1
                denom = prev_memory if prev_memory != 0 else 1.0
                resource_changed = (delta_resource / denom) >= threshold
                memory_changed = (delta_memory / denom) >= threshold

                print("prev_resource:", prev_resource, "curr_resource:", curr_resource)
                print("prev_memory:", prev_memory, "curr_memory:", curr_memory)
                print("resource_changed:", resource_changed, "memory_changed:", memory_changed)

                if resource_changed and memory_changed:
                    message = (
                        f"Resource fluctuation detected:\n"
                        f"  {resource_type.upper()}: {prev_resource:.1f} -> {curr_resource:.1f} (Δ {delta_resource:.1f})\n"
                        f"  Memory: {prev_memory:.1f}% -> {curr_memory:.1f}% (Δ {delta_memory:.1f})"
                    )
                    if notify_func:
                        notify_func(True, message)
                    else:
                        print(message)
                else:
                    if notify_func:
                        notify_func(False, "No significant change")
                    else:
                        print("No significant resource fluctuation detected")

                prev_resource, prev_memory = curr_resource, curr_memory

        fluctuation_thread = Thread(target=monitor_loop, daemon=True)
        fluctuation_thread.start()

    # -------------------------
    # Runner & cleanup
    # -------------------------
    def run(self):
        """Start the monitoring loop and spawn configured load workers."""
        def handle_exit(signum, frame):
            print(f"\nReceived signal {signum}, cleaning up and exiting")
            self.cleanup()
            sys.exit(0)

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, handle_exit)   # Ctrl+C
        signal.signal(signal.SIGTERM, handle_exit)  # kill -15

        monitor_thread = Thread(target=self.monitor, daemon=True)
        monitor_thread.start()

        try:
            # Allocate memory if requested
            if self.memory_mb:
                self.global_memory.append(self.memory_load(self.memory_mb))
                print(f"Allocated {self.memory_mb} MB memory")

            # Spawn CPU worker processes if requested
            if self.cpu_cores:
                for _ in range(self.cpu_cores):
                    p = multiprocessing.Process(target=self.cpu_worker, args=(self.stop_event,))
                    p.start()
                    self.processes.append(p)
                print(f"Started {self.cpu_cores} CPU worker processes")

            # Start GPU worker thread if requested
            if self.gpu_calc_intensity or self.gpu_mem_intensity:
                MEMORY_LEVELS = {
                    1: 1,  2: 2,  3: 3,  4: 4,  5: 6,
                    6: 8,  7: 9,  8: 10, 9: 11, 10: 10.8
                }
                mem_gb = MEMORY_LEVELS.get(self.gpu_mem_intensity, 0)
                t = Thread(target=self.gpu_worker, args=(self.gpu_calc_intensity, mem_gb), daemon=True)
                t.start()
                self.threads.append(t)
                print(f"Started GPU worker (compute intensity: {self.gpu_calc_intensity}, memory: {mem_gb} GB)")

            # Keep the main thread alive until stop_event is set
            while not self.stop_event.is_set():
                time.sleep(0.5)

        except KeyboardInterrupt:
            self.cleanup()
            sys.exit(0)

    def cleanup(self):
        """Stop workers and free allocated resources."""
        print("\nCleaning up resources...")
        self.stop_event.set()

        # Terminate CPU worker processes
        for p in self.processes:
            try:
                p.terminate()
            except Exception:
                pass

        # Join GPU threads (they are daemon threads, but attempt a short join)
        for t in self.threads:
            t.join(timeout=2)

        # Free allocated memory buffers
        self.global_memory.clear()

        # Release CUDA memory if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        import gc
        gc.collect()

        print("Cleanup complete")


# -------------------------
# Command-line interface
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="System load generator (class-based)")
    parser.add_argument('--memory', type=int, help='Memory load in MB')
    parser.add_argument('--cpu', type=int, help='Number of CPU worker processes')
    parser.add_argument('--gpu-calc', type=int, help='GPU compute intensity (1-10)')
    parser.add_argument('--gpu-mem', type=int, help='GPU memory intensity (1-10)')
    args = parser.parse_args()

    generator = SystemLoadGenerator(
        memory_mb=args.memory or 0,
        cpu_cores=args.cpu or 0,
        gpu_calc_intensity=args.gpu_calc or 0,
        gpu_mem_intensity=args.gpu_mem or 0
    )
    generator.run()
