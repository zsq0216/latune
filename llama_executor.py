import subprocess
import re
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import requests
import psutil
import numpy as np
import pandas as pd
import csv
import itertools
import logging
import ctypes

class LlamaExecutor:
    def __init__(self, 
                 param_types: Dict[str, str],
                 model_path: Optional[str] = "./../models/qwen3-8b-q4.gguf",
                 device: str = "gpu",
                 hardware: str = "rtx3060",
                 default_prompt: str = "I believe the meaning of life is",
                 ):
        """
        :param param_types: 参数字典，记录参数类型（用于构建命令行参数）
        :param model_path: 模型文件默认路径
        :param default_prompt: 默认使用的提示语
        """
        self.param_types = param_types
        self.default_prompt = default_prompt
        self.model_path = model_path
        self.device = device
        self.hardware = hardware
        self.stop_event = None
        self.server_process = None
        self.prompt_list = [
            "How much should I charge for web design by hour with profit if I need to make $8000 a month",
            "Please introduce yourself", 
            "How should CSP headers be set in a php website in the interest of web application security"
            "jeffrey and Sarah found one last CRT TV he didn\u2019t sell online, and they deepen their commitment to each other while playing retro games on it",
            "how to grant select method to a user on a table in oracle?",
            "what other ways do I have to identify what table to write to instead of dlp api?",
            "I need your help writing an article. I will provide you with some background information to begin with. And then I will provide you with directions to help me write the article.",\
            "Write linkedin post from the University of Manchester wishing ramadan kareem",
            "What could be a cool teamname for the Continuous Engineering Unit."
            "what if I use column names and order to determine the table?",
            "what exactly is breath control?",
            "suggest images for the post",
            "Generate some popular hashtags",
            # "what could CEU Stand for in Spain",
            # "What could it stand for releated to IT development",
            # "What is a short name for Continuous Engineering Unit",
            # "Is there another cool way",
            # "What political party does Narendra Modi belong to?"
        ]
        print(f"Using device: {self.device}")

        
    def generate_configs(self, performance_params, n_samples=100):
        """生成随机配置样本"""
        configs = []
        for _ in range(n_samples):
            config = {}
            # 遍历字典的键值对 (name, param_info)
            for name, param_info in performance_params.items():  # 关键修改点
                param_type = param_info['type']  # 从 param_info 获取类型
                if param_type == 'boolean':
                    config[name] = np.random.choice([True, False])
                elif param_type == 'integer':
                    config[name] = np.random.randint(param_info['values']['min'], param_info['values']['max'] + 1)
                elif param_type == 'enum':
                    config[name] = np.random.choice(param_info['values'])
                elif param_type == 'float':
                    config[name] = np.random.uniform(param_info['values']['min'], param_info['values']['max'])
            # config = self.handle_dependency(config)
            configs.append(config)
        return configs
    
    def generate_configs_fixed(self, performance_params, n_samples=100, seed=42):
        """生成随机配置样本（可复现）"""
        # 固定随机种子：每次调用都会从相同的随机序列开始
        rng = np.random.default_rng(seed)

        configs = []
        # 可选：为了让“随机数消耗顺序”也稳定，按 key 排序（避免 dict 构造顺序影响）
        items = sorted(performance_params.items())

        for _ in range(n_samples):
            config = {}
            for name, param_info in items:
                param_type = param_info['type']
                if param_type == 'boolean':
                    # rng.integers 更快，也可用 rng.choice([True, False])
                    config[name] = bool(rng.integers(0, 2))
                elif param_type == 'integer':
                    lo = param_info['values']['min']
                    hi = param_info['values']['max'] + 1  # 上界开区间
                    config[name] = int(rng.integers(lo, hi))
                elif param_type == 'enum':
                    config[name] = rng.choice(param_info['values'])
                elif param_type == 'float':
                    lo = param_info['values']['min']
                    hi = param_info['values']['max']
                    config[name] = float(rng.uniform(lo, hi))
                else:
                    raise ValueError(f"未知类型: {param_type}")
            configs.append(config)
        return configs

    def handle_dependency(self, config):
        #config['grp-attn-w']=config['grp-attn-n']的整数倍
        # if config['grp-attn-n'] absent
        if 'grp-attn-n' not in config:
            return config
        
        max_multiplier = 2048 // config['grp-attn-n'] # 2048是grp-attn-w最大值
        config['grp-attn-w'] = config['grp-attn-n'] * max_multiplier
        return config

    def _build_server_command(self, config: Dict, model_path: str) -> List[str]:
        """构建命令行参数列表"""
        cmd = [
            f"./../llama.cpp/build{self.device}/bin/llama-server",
            "-m", model_path,
            "--host", config.get("host", "127.0.0.1"),
            "--port", str(config.get("port", 8080))
        ]
        
        for param, value in config.items():
            param_type = self.param_types.get(param, "unknown")
            if param_type == "boolean":
                if value:
                    cmd.append(f"--{param}")
            else:
                cmd.extend([f"--{param}", str(value)])
        return cmd

    @staticmethod
    def _run_with_realtime_output(cmd: List[str]) -> Tuple[str, str]:
        """执行子进程并捕获输出"""
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        stdout_output, stderr_output = process.communicate()
        return stdout_output, stderr_output

    @staticmethod
    def _metrics_parser(output: str) -> float:
        """解析性能指标输出"""
        # 匹配 eval time
        eval_time_match = re.search(
            r"llama_perf_context_print:\s+eval time =\s+([\d.]+) ms /\s+\d+ runs\s+\(\s+([\d.]+) ms",
            output, 
            re.MULTILINE
        )
        
        if not eval_time_match:
            raise ValueError("Failed to parse metrics from output")
            
        eval_time_per_token = float(eval_time_match.group(2))
        print(1 / eval_time_per_token * 1000)
        return 1 / eval_time_per_token * 1000  # 转换为tokens/s

    def _start_server_and_metrics_thread(self, config, model_path):
        try:
            self.server_process = self._run_llama_server(config = config, model_path = model_path)
        except TimeoutError as e:
            print(e)
            raise ValueError(e)

        resource_metrics = []
        from threading import Thread, Event
        self.stop_event = Event()

        def collect_metrics():
            while not self.stop_event.is_set():
                metrics = self.collect_server_metrics(self.server_process)
                metrics['timestamp'] = time.time()
                resource_metrics.append(metrics)
                time.sleep(0.5)

        thread = Thread(target=collect_metrics)
        thread.start()

        return thread, resource_metrics

    def _send_request_batch(self, prompts, config):
        tps_list, pps_list = [], []
        for prompt in prompts:
            try:
                tps, pps = self.send_request_to_server(
                    prompt=prompt,
                    port=config.get('port', 8080),
                    max_tokens=64
                )
                tps_list.append(tps)
                pps_list.append(pps)
            except Exception as e:
                print(f"Request failed: {str(e)}")
        return tps_list, pps_list

    def _summarize_metrics(self, resource_metrics, tps_list, pps_list):
        cpu = [m.get('cpu_percent', 0) for m in resource_metrics]
        mem = [m.get('memory_mb', 0) for m in resource_metrics]
        gpu = [m.get('gpu_mem', 0) for m in resource_metrics]

        return {
            'tps_avg': np.mean(tps_list) if tps_list else 0,
            'pps_avg': np.mean(pps_list) if pps_list else 0,
            'cpu_avg': np.mean(cpu) if cpu else 0,
            'mem_avg': np.mean(mem) if mem else 0,
            'gpu_avg': np.mean(gpu) if gpu else 0
        }

    def run_server_performance_test(self, config: Dict, num_requests: int = 3,
                                    model_path: Optional[str] = None) -> Dict:
        model_path = model_path or self.model_path
        thread = None
        try:
            try:
                thread, metrics = self._start_server_and_metrics_thread(config, model_path)
            except (TimeoutError, ValueError) as e:
                return {
                    'tps_avg': 0,
                    'gpu_avg': {"m4": 9000.0, "rtx3060": 11000.0, "rtx4090": 22000.0, "orin": 5000.0}[self.hardware]
                }
            prompts = self.prompt_list[:num_requests]
            tps_list, pps_list = self._send_request_batch(prompts, config)
            time.sleep(0.5)
        finally:
            if self.stop_event:
                self.stop_event.set()
            if thread is not None:
                thread.join()
            if hasattr(self, 'server_process') and self.server_process is not None:
                self.server_process.terminate()

        return self._summarize_metrics(metrics, tps_list, pps_list)
    
    def _run_llama_server(self, config: Dict, model_path: Optional[str] = None) -> subprocess.Popen:
        """启动llama.cpp服务器进程"""
        model_path = model_path
        cmd = self._build_server_command(config=config, model_path = model_path)
        # print(cmd)
            
        server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8"
        )

        # server_process.communicate()这个会阻塞，只能测试的时候用
        # stdout_output, stderr_output = server_process.communicate()
        # print("stdout:", stdout_output)
        # print("stderr:", stderr_output)
        
        # 等待服务器启动
        self._wait_for_server_ready(config.get("port", 8080), timeout=30)
        return server_process
        

    def extract_meta_feature(self, config: Dict, model_path: Optional[str] = None) -> subprocess.Popen:
        """启动 llama.cpp 服务器进程，收集 stdout/stderr 后立即 kill 掉"""
        cmd = self._build_server_command(config=config, model_path=model_path)

        server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8"
        )

        try:
            # 等待一段时间读取输出（例如 10 秒），可按需调整
            stdout_output, stderr_output = server_process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            # 如果 10 秒内没结束，就强杀并再收集残余输出
            server_process.kill()
            stdout_output, stderr_output = server_process.communicate()
        except Exception as e:
            logging.exception("Error while running llama server: %s", e)
            server_process.kill()
            stdout_output, stderr_output = "", str(e)

        # 这里可以根据需要保存/打印输出
        # print("stdout:", stdout_output)
        # print("stderr:", stderr_output)

        # 确保进程被杀
        if server_process.poll() is None:
            server_process.kill()

        return stderr_output
            

    def send_request_to_server(self, prompt: str, port: int = 8080, 
                            max_tokens: int = 64) -> Tuple[float, int]:
        """发送HTTP请求到服务器并收集指标"""
        url = f"http://localhost:{port}/v1/completions"
        headers = {"Content-Type": "application/json"}
        
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            
            if response.status_code != 200:
                raise RuntimeError(f"Request failed: {response.text}")

            timings = response.json()['timings']
            TPS = timings['predicted_per_second']
            PPS = timings['prompt_per_second']
            return TPS, PPS

        except requests.exceptions.Timeout:
            # 超过2分钟未响应，返回1,1
            return 1, 1

    def collect_server_metrics(self, process: subprocess.Popen) -> Dict:
        """收集服务器资源指标"""
        try:
            proc = psutil.Process(process.pid)
            proc.cpu_percent(interval=None)  
            time.sleep(0.1) 
            if self.hardware == "m4": 
                gpu_mem = self._get_mac_gpu_memory_usage()
            elif self.hardware == "orin":
                gpu_mem = self._get_orin_cuda_used_mb()-2800
            else:
                gpu_mem = self._get_gpu_memory_usage(proc.pid)
            return {
                "cpu_percent": proc.cpu_percent()/psutil.cpu_count(logical=True),
                "system_cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory_mb": proc.memory_info().rss / 1024 / 1024,
                "gpu_mem": gpu_mem
            }
        except psutil.NoSuchProcess:
            return {"error": "Process not found"}
    
    def _get_gpu_memory_usage(self, pid):
        """
        通过 `nvidia-smi` 和 `grep` 直接过滤 PID 的 GPU 显存使用量（单位：MiB）
        返回：显存使用量（整数），若未找到或出错则返回 0
        """
        try:
            # 将命令拼接为字符串，使用管道符连接 grep
            cmd = f"nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv | grep {pid}"
            output = subprocess.check_output(
                cmd, 
                shell=True,          # 启用 shell 以解析管道符
                universal_newlines=True,
                stderr=subprocess.DEVNULL  # 忽略错误输出
            )
            
            # 解析输出并累加显存
            total_mem = 0
            for line in output.strip().split("\n"):
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) >= 2:
                    mem_str = parts[1].strip().split()[0]  # 提取数值（如 "2500 MiB" -> 2500）
                    total_mem += int(mem_str)
            return total_mem
        except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
            # 处理无匹配项、命令执行失败、格式错误等异常
            return 0
        
    def _get_mac_gpu_memory_usage(self):
        """
        估算 Mac 上 GPU 显存占用（实际上是 wired memory 的一部分）
        返回：占用的 MB
        """
        try:
            output = subprocess.check_output(["vm_stat"], universal_newlines=True)
            match = re.search(r"Pages wired down:\s+(\d+)", output)
            if match:
                pages = int(match.group(1))
                # 每页大小（macOS 默认 4096 bytes = 4KB）
                vram_mb = pages * 4096 / 1024 / 1024
                return int(vram_mb)
            return 0
        except Exception:
            return 0

    # def _get_orin_ram_used_mb(self) -> int:
    #     """
    #     使用 tegrastats 获取当前系统 RAM 已用量（MB）。
    #     返回：整数 MB
    #     """
    #     try:
    #         # 启动 tegrastats，只取一行
    #         proc = subprocess.Popen(
    #             ["tegrastats", "--interval", "100"],
    #             stdout=subprocess.PIPE,
    #             stderr=subprocess.DEVNULL,
    #             universal_newlines=True
    #         )
    #         line = proc.stdout.readline()
    #         proc.terminate()

    #         # 解析 "RAM 3958/15849MB ..." -> used=3958
    #         m = re.search(r"RAM\s+(\d+)\s*/\s*(\d+)MB", line)
    #         if m:
    #             return int(m.group(1))  # 已用 MB
    #     except Exception:
    #         pass
    #     return 0
    
    # def _get_orin_ram_used_mb(self, pid: int) -> int:
    #     """
    #     返回指定进程的 VmRSS (MB)。
    #     VmRSS = 进程实际占用的物理内存（驻留集大小）。
    #     """
    #     try:
    #         with open(f"/proc/{pid}/status", "r") as f:
    #             for line in f:
    #                 if line.startswith("VmRSS:"):
    #                     parts = line.split()
    #                     if len(parts) >= 2:
    #                         # VmRSS 默认单位是 kB
    #                         return int(parts[1]) // 1024
    #     except FileNotFoundError:
    #         raise ValueError(f"进程 {pid} 不存在")
    #     except Exception as e:
    #         raise RuntimeError(f"读取 VmRSS 出错: {e}")
    #     return 0

    def _get_orin_cuda_used_mb(self) -> int:
        """
        返回当前 GPU（设备）已用内存，单位 MB（非 per-PID，全局视角）。
        适用于 Jetson（无 nvidia-smi/NVML 的场景）。
        """
        try:
            lib = ctypes.CDLL("libcuda.so")

            cuInit = lib.cuInit
            cuInit.argtypes = [ctypes.c_uint]
            if cuInit(0) != 0:
                return 0

            cuDeviceGet = lib.cuDeviceGet
            cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
            dev = ctypes.c_int()
            if cuDeviceGet(ctypes.byref(dev), 0) != 0:
                return 0

            # 创建上下文（有些 Jetson 固件只导出 v2 版本）
            cuCtxCreate = getattr(lib, "cuCtxCreate_v2", getattr(lib, "cuCtxCreate"))
            cuCtxCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
            ctx = ctypes.c_void_p()
            if cuCtxCreate(ctypes.byref(ctx), 0, dev.value) != 0:
                return 0

            # 读取 free/total
            cuMemGetInfo = getattr(lib, "cuMemGetInfo_v2", getattr(lib, "cuMemGetInfo"))
            cuMemGetInfo.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
            free_b = ctypes.c_size_t()
            total_b = ctypes.c_size_t()
            ret = cuMemGetInfo(ctypes.byref(free_b), ctypes.byref(total_b))

            # 释放上下文
            try:
                cuCtxDestroy = getattr(lib, "cuCtxDestroy_v2", getattr(lib, "cuCtxDestroy"))
                cuCtxDestroy.argtypes = [ctypes.c_void_p]
                cuCtxDestroy(ctx)
            except Exception:
                pass

            if ret != 0 or total_b.value == 0:
                return 0

            used_mb = (total_b.value - free_b.value) // (1024 * 1024)
            return int(used_mb)
        except Exception:
            return 0

    def _wait_for_server_ready(self, port: int, timeout: int = 30):
        """检测服务器就绪状态"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                if requests.get(f"http://localhost:{port}/health").ok:
                    return
            except requests.ConnectionError:
                time.sleep(1)
        raise TimeoutError("Server startup timed out")
        
if __name__ == "__main__":
    param_types_instance ={'gpu-layers': 'integer',
                           'ctx-size': 'integer',
                           'no-kv-offload': 'boolean',
                           'defrag-thold': 'float',
                           'ubatch-size': 'integer',}
    config_list =[
        {"gpu-layers":25, "no-kv-offload": True, "ctx-size": 41, "ubatch-size": 512, "defrag-thold": 0.3},
        {"gpu-layers":25, "no-kv-offload": True, "ctx-size": 41, "ubatch-size": 512, "defrag-thold": -0.1}
    ]

    executor = LlamaExecutor(param_types=param_types_instance,
                             model_path="./../models/phimoe-mini-q8.gguf",
                              device="gpu")
    results = []
    for config in config_list:
        print(config)
        result = executor.run_server_performance_test(config)
        # 把config拼接到result中
        # result.update(config)
        results.append(result)
        print(result)


