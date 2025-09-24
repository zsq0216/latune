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
    def __init__(self, memory_mb=0, cpu_cores=0, gpu_calc_intensity=0, gpu_mem_intensity=0):
        self.memory_mb = memory_mb
        self.cpu_cores = cpu_cores
        self.gpu_calc_intensity = gpu_calc_intensity or 0
        self.gpu_mem_intensity = gpu_mem_intensity or 0

        self.stop_event = Event()
        self.processes = []
        self.threads = []
        self.global_memory = []

    def memory_load(self, mb):
        bytes_amount = mb * 1024 * 1024
        return bytearray(bytes_amount)

    def cpu_worker(self, stop_event):
        while not stop_event.is_set():
            sum(i * i for i in range(10**7))

    def gpu_worker(self, intensity, mem_gb):
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            chunk_size = 2**20
            memory_blocks = []

            mem_bytes = min(int(12 * 0.9 * 1024**3), mem_gb * 1024**3)
            for _ in range(0, mem_bytes, chunk_size):
                block = torch.zeros(chunk_size // 4, dtype=torch.float32, device=device)
                memory_blocks.append(block)

            size = 500 * intensity
            a = torch.randn(size, size, device=device)
            b = torch.randn(size, size, device=device)

            while not self.stop_event.is_set():
                c = torch.mm(a, b)
                a = c * 0.9 + torch.randn_like(c) * 0.1
                b = c.T * 0.9 + torch.randn_like(c.T) * 0.1
                torch.cuda.synchronize()

        except ImportError:
            print("需要安装PyTorch: pip install torch")
        except Exception as e:
            print(f"GPU错误: {e}")

    def get_system_stats(self):
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
        返回当前系统的可用资源信息，包括：
        - 可用内存（MB）
        - CPU 空闲百分比（0-100）
        - 每张 GPU 的剩余显存（MB）
        """
        # 获取内存信息
        mem = psutil.virtual_memory()
        mem_available_mb = mem.available / 1024 / 1024  # MB

        # 获取 CPU 空闲率（100 - 当前使用率）
        cpu_idle_percent = 100 - psutil.cpu_percent(interval=0.1)

        # 获取 GPU 可用显存
        gpu_free_mem_list = []
        try:
            nvmlInit()
            for i in range(nvmlDeviceGetCount()):
                handle = nvmlDeviceGetHandleByIndex(i)
                mem_info = nvmlDeviceGetMemoryInfo(handle)
                free_mem_mb = mem_info.free / 1024 / 1024  # MB
                gpu_free_mem_list.append(free_mem_mb)
            nvmlShutdown()
        except NVMLError:
            gpu_free_mem_list = []

        return {
            "mem_avail": round(mem_available_mb*0.9, 2),
            "cpu_avail": round(cpu_idle_percent*0.9, 2),
            "gpu_avail": gpu_free_mem_list[0]*0.9
        }

    def monitor(self):
        try:
            while not self.stop_event.wait(1):
                stats = self.get_system_stats()
                print("\033c", end="")

                print("=== 实时系统监控 ===")
                print(f"CPU使用率: {stats['cpu']}%")
                print(f"内存使用: {stats['memory']}")
                if stats['gpu']:
                    for i, gpu in enumerate(stats['gpu']):
                        print(f"GPU{i} 使用率: {gpu['gpu_util']}% | 显存: {gpu['mem_used']:.1f}/{gpu['mem_total']:.1f}GB ({gpu['mem_util']:.1f}%)")
                else:
                    print("GPU信息: 不可用")
                print("\n按 Ctrl+C 退出")
        except KeyboardInterrupt:
            self.stop_event.set()

    def _collect_resource_samples(self, resource_type='gpu', duration=10, interval=1.0):
        """
        采集资源样本，返回 (资源平均值, 内存平均值)
        """
        resource_samples = []
        memory_samples = []

        for _ in range(int(duration / interval)):
            stats = self.get_system_stats()
            
            if resource_type == 'cpu':
                resource_val = stats['cpu']
            elif resource_type == 'gpu':
                if stats['gpu']:
                    resource_val = max(g['mem_used'] for g in stats['gpu'])
                else:
                    resource_val = 0
            else:
                raise ValueError("资源类型必须是 'cpu' 或 'gpu'")

            # 统一获取内存使用率
            mem_percent = float(stats['memory'].split('(')[-1].strip('% )'))
            resource_samples.append(resource_val)
            memory_samples.append(mem_percent)
            time.sleep(interval)

        avg_resource = sum(resource_samples) / len(resource_samples)
        avg_memory = sum(memory_samples) / len(memory_samples)
        return avg_resource, avg_memory

    def detect_fluctuation_continuous(self, resource_type='gpu', duration=10, interval=10, threshold=0.1, notify_func=None):
        """
        持续运行的资源波动检测器。
        每 interval 秒采样一次资源，采样 duration 秒并对比前一次的平均值；
        若 CPU/GPU 与 Memory 使用率同时变化超过 threshold%，则触发通知。

        :param resource_type: 'cpu' 或 'gpu'
        :param duration: 每次采样持续时间（秒）
        :param interval: 每 interval 秒对比一次（实际采样周期 = interval + duration）
        :param threshold: 平均值变动阈值（百分比）
        :param notify_func: 回调函数，参数 (changed: bool, message: str)
        """
        def monitor_loop():
            prev_resource, prev_memory = self._collect_resource_samples(resource_type, duration)

            while not self.stop_event.is_set():
                time.sleep(interval)
                curr_resource, curr_memory = self._collect_resource_samples(resource_type, duration)

                delta_resource = abs(curr_resource - prev_resource)
                delta_memory = abs(curr_memory - prev_memory)

                resource_changed = delta_resource/prev_memory >= threshold
                memory_changed = delta_memory/prev_memory >= threshold

                print("prev_resource:", prev_resource, "curr_resource:", curr_resource)
                print("prev_memory:", prev_memory, "curr_memory:", curr_memory)
                print("resource_changed:", resource_changed, "memory_changed:", memory_changed)

                if resource_changed and memory_changed:
                    message = (f"⚠️ 资源变动检测:\n"
                               f"  {resource_type.upper()}: {prev_resource:.1f}% → {curr_resource:.1f}% "
                               f"(Δ{delta_resource:.1f}%)\n"
                               f"  内存: {prev_memory:.1f}% → {curr_memory:.1f}% "
                               f"(Δ{delta_memory:.1f}%)")
                    if notify_func:
                        notify_func(True, message)
                    else:
                        print(message)
                else:
                    if notify_func:
                        notify_func(False, "无显著变动")
                    else:
                        print("✅ 无显著资源变动")

                prev_resource, prev_memory = curr_resource, curr_memory

        fluctuation_thread = Thread(target=monitor_loop)
        fluctuation_thread.daemon = True
        fluctuation_thread.start()

    def run(self):
        def handle_exit(signum, frame):
            print(f"\n🛑 收到信号 {signum}，触发清理")
            self.cleanup()
            sys.exit(0)  # 确保退出整个程序

        signal.signal(signal.SIGINT, handle_exit)   # Ctrl+C
        signal.signal(signal.SIGTERM, handle_exit)  # kill -15 pid
        
        monitor_thread = Thread(target=self.monitor)
        monitor_thread.start()

        try:
            if self.memory_mb:
                self.global_memory.append(self.memory_load(self.memory_mb))
                print(f"✅ 已分配 {self.memory_mb}MB 内存")

            if self.cpu_cores:
                for _ in range(self.cpu_cores):
                    p = multiprocessing.Process(target=self.cpu_worker, args=(self.stop_event,))
                    p.start()
                    self.processes.append(p)
                print(f"✅ 已启动 {self.cpu_cores} 个CPU负载进程")

            if self.gpu_calc_intensity or self.gpu_mem_intensity:
                MEMORY_LEVELS = {
                    1: 1,  2: 2,  3: 3,  4: 4,  5: 6,
                    6: 8,  7: 9,  8: 10, 9: 11, 10: 10.8
                }
                mem_gb = MEMORY_LEVELS.get(self.gpu_mem_intensity, 0)
                t = Thread(target=self.gpu_worker, args=(self.gpu_calc_intensity, mem_gb))
                t.daemon = True
                t.start()
                self.threads.append(t)
                print(f"✅ 已启动GPU负载 [计算:{self.gpu_calc_intensity}/显存:{mem_gb}GB]")

            while not self.stop_event.is_set():
                time.sleep(0.5)

        except KeyboardInterrupt:
            self.cleanup()
            sys.exit(0)

    def cleanup(self):
        print("\n🛑 正在清理资源...")
        self.stop_event.set()

        for p in self.processes:
            p.terminate()

        for t in self.threads:
            t.join(timeout=2)

        self.global_memory.clear()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        import gc
        gc.collect()

        print("✅ 清理完成")

# 命令行入口
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="系统负载生成器V2（类封装版）")
    parser.add_argument('--memory', type=int, help='内存负载大小（MB）')
    parser.add_argument('--cpu', type=int, help='CPU负载核数')
    parser.add_argument('--gpu-calc', type=int, help='GPU计算强度（1-10）')
    parser.add_argument('--gpu-mem', type=int, help='GPU显存强度（1-10）')
    args = parser.parse_args()

    generator = SystemLoadGenerator(
        memory_mb=args.memory or 0,
        cpu_cores=args.cpu or 0,
        gpu_calc_intensity=args.gpu_calc or 0,
        gpu_mem_intensity=args.gpu_mem or 0
    )
    generator.run()



