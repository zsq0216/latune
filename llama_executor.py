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
    """
    LlamaExecutor: utility class to run a llama.cpp server, send requests,
    and collect resource/performance metrics.

    Args:
        param_types (Dict[str, str]): Mapping of parameter name -> type (used to build CLI args).
        model_path (Optional[str]): Default path to model file.
        device (str): Device type to use (e.g., "gpu" or "cpu").
        hardware (str): Host hardware identifier used for certain fallbacks.
        default_prompt (str): Default prompt used for inference tests.
    """

    def __init__(
        self,
        param_types: Dict[str, str],
        model_path: Optional[str] = "./../models/qwen3-8b-q4.gguf",
        device: str = "gpu",
        hardware: str = "rtx3060",
        default_prompt: str = "I believe the meaning of life is",
    ):
        """
        Initialize an executor instance.

        :param param_types: Dictionary that records parameter types (used to construct CLI args).
        :param model_path: Default model file path.
        :param default_prompt: Default prompt for inference requests.
        """
        self.param_types = param_types
        self.default_prompt = default_prompt
        self.model_path = model_path
        self.device = device
        self.hardware = hardware
        self.stop_event = None
        self.server_process = None

        # A small set of prompts used for throughput/latency tests
        self.prompt_list = [
            "How much should I charge for web design by hour with profit if I need to make $8000 a month",
            "Please introduce yourself",
            "How should CSP headers be set in a php website in the interest of web application security"
            "jeffrey and Sarah found one last CRT TV he didn\u2019t sell online, and they deepen their commitment to each other while playing retro games on it",
            "how to grant select method to a user on a table in oracle?",
            "what other ways do I have to identify what table to write to instead of dlp api?",
            "I need your help writing an article. I will provide you with some background information to begin with. And then I will provide you with directions to help me write the article.",
            "Write linkedin post from the University of Manchester wishing ramadan kareem",
            "What could be a cool teamname for the Continuous Engineering Unit."
            "what if I use column names and order to determine the table?",
            "what exactly is breath control?",
            "suggest images for the post",
            "Generate some popular hashtags",
        ]
        print(f"Using device: {self.device}")

    # ---------------------------------------------------------------------
    # Configuration sampling
    # ---------------------------------------------------------------------

    def generate_configs(self, performance_params, n_samples=100):
        """Generate random configuration samples.

        Args:
            performance_params (dict): Parameter definitions (each has 'type' and 'values').
            n_samples (int): Number of configs to generate.

        Returns:
            list[dict]: Randomly generated configs.
        """
        configs = []
        for _ in range(n_samples):
            config = {}
            # Iterate over (name, param_info)
            for name, param_info in performance_params.items():
                param_type = param_info["type"]
                if param_type == "boolean":
                    config[name] = np.random.choice([True, False])
                elif param_type == "integer":
                    config[name] = np.random.randint(
                        param_info["values"]["min"], param_info["values"]["max"] + 1
                    )
                elif param_type == "enum":
                    config[name] = np.random.choice(param_info["values"])
                elif param_type == "float":
                    config[name] = np.random.uniform(
                        param_info["values"]["min"], param_info["values"]["max"]
                    )
            # Optionally apply inter-parameter dependency handlers
            # config = self.handle_dependency(config)
            configs.append(config)
        return configs

    def generate_configs_fixed(self, performance_params, n_samples=100, seed=42):
        """Generate reproducible random configuration samples.

        Uses a numpy Generator with a fixed seed and sorts keys to avoid
        dict insertion-order variation affecting random sequence consumption.

        Args:
            performance_params (dict): Parameter definitions.
            n_samples (int): Number of configurations to produce.
            seed (int): RNG seed for reproducibility.

        Returns:
            list[dict]: Generated configurations.
        """
        rng = np.random.default_rng(seed)
        configs = []
        # Sort items so the sampling order is deterministic
        items = sorted(performance_params.items())

        for _ in range(n_samples):
            config = {}
            for name, param_info in items:
                param_type = param_info["type"]
                if param_type == "boolean":
                    config[name] = bool(rng.integers(0, 2))
                elif param_type == "integer":
                    lo = param_info["values"]["min"]
                    hi = param_info["values"]["max"] + 1  # upper-exclusive
                    config[name] = int(rng.integers(lo, hi))
                elif param_type == "enum":
                    config[name] = rng.choice(param_info["values"])
                elif param_type == "float":
                    lo = param_info["values"]["min"]
                    hi = param_info["values"]["max"]
                    config[name] = float(rng.uniform(lo, hi))
                else:
                    raise ValueError(f"Unknown parameter type: {param_type}")
            configs.append(config)
        return configs

    # ---------------------------------------------------------------------
    # Dependency handling
    # ---------------------------------------------------------------------

    def handle_dependency(self, config):
        """
        Handle dependencies between parameters.

        Example: enforce grp-attn-w to be an integer multiple of grp-attn-n.

        Args:
            config (dict): Candidate configuration.

        Returns:
            dict: Updated configuration with dependencies satisfied.
        """
        if "grp-attn-n" not in config:
            return config

        # 2048 is the upper bound for grp-attn-w
        max_multiplier = 2048 // config["grp-attn-n"]
        config["grp-attn-w"] = config["grp-attn-n"] * max_multiplier
        return config

    # ---------------------------------------------------------------------
    # Server command construction and process management
    # ---------------------------------------------------------------------

    def _build_server_command(self, config: Dict, model_path: str) -> List[str]:
        """Build the command-line argument list to start the llama server.

        Args:
            config (Dict): Configuration values to translate into CLI args.
            model_path (str): Path to the model file.

        Returns:
            List[str]: Command and arguments for subprocess.
        """
        cmd = [
            f"./../llama.cpp/build{self.device}/bin/llama-server",
            "-m",
            model_path,
            "--host",
            config.get("host", "127.0.0.1"),
            "--port",
            str(config.get("port", 8080)),
        ]

        for param, value in config.items():
            param_type = self.param_types.get(param, "unknown")
            if param_type == "boolean":
                # For boolean flags, add flag only when value is True
                if value:
                    cmd.append(f"--{param}")
            else:
                cmd.extend([f"--{param}", str(value)])
        return cmd

    @staticmethod
    def _run_with_realtime_output(cmd: List[str]) -> Tuple[str, str]:
        """Run a subprocess and capture stdout/stderr (blocking).

        This helper runs the process to completion and returns the outputs.

        Args:
            cmd (List[str]): Command with arguments.

        Returns:
            Tuple[str, str]: (stdout, stderr) text contents.
        """
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

    # ---------------------------------------------------------------------
    # Metrics parsing and collection
    # ---------------------------------------------------------------------

    @staticmethod
    def _metrics_parser(output: str) -> float:
        """
        Parse performance metrics from server output.

        Looks for lines like:
        llama_perf_context_print: eval time = <x> ms / <y> runs ( <z> ms

        Returns:
            float: tokens per second derived from the parsed metric.

        Raises:
            ValueError: when parsing fails.
        """
        eval_time_match = re.search(
            r"llama_perf_context_print:\s+eval time =\s+([\d.]+) ms /\s+\d+ runs\s+\(\s+([\d.]+) ms",
            output,
            re.MULTILINE,
        )

        if not eval_time_match:
            raise ValueError("Failed to parse metrics from output")

        eval_time_per_token = float(eval_time_match.group(2))
        # convert ms/token to tokens/s
        return 1 / eval_time_per_token * 1000

    def _start_server_and_metrics_thread(self, config, model_path):
        """
        Start the llama server process and a background thread to collect metrics.

        Returns:
            tuple(threading.Thread, list): (metrics collection thread, resource metrics list)
        """
        try:
            self.server_process = self._run_llama_server(config=config, model_path=model_path)
        except TimeoutError as e:
            print(e)
            raise ValueError(e)

        resource_metrics = []
        from threading import Thread, Event

        self.stop_event = Event()

        def collect_metrics():
            # Periodically collect resource metrics until stop_event is set.
            while not self.stop_event.is_set():
                metrics = self.collect_server_metrics(self.server_process)
                metrics["timestamp"] = time.time()
                resource_metrics.append(metrics)
                time.sleep(0.5)

        thread = Thread(target=collect_metrics)
        thread.start()

        return thread, resource_metrics

    def _send_request_batch(self, prompts, config):
        """Send a batch of prompts to the server and collect throughput stats.

        Returns:
            tuple(list, list): (tps_list, pps_list)
        """
        tps_list, pps_list = [], []
        for prompt in prompts:
            try:
                tps, pps = self.send_request_to_server(
                    prompt=prompt, port=config.get("port", 8080), max_tokens=64
                )
                tps_list.append(tps)
                pps_list.append(pps)
            except Exception as e:
                print(f"Request failed: {str(e)}")
        return tps_list, pps_list

    def _summarize_metrics(self, resource_metrics, tps_list, pps_list):
        """Aggregate collected metrics into summary statistics."""
        cpu = [m.get("cpu_percent", 0) for m in resource_metrics]
        mem = [m.get("memory_mb", 0) for m in resource_metrics]
        gpu = [m.get("gpu_mem", 0) for m in resource_metrics]

        return {
            "tps_avg": np.mean(tps_list) if tps_list else 0,
            "pps_avg": np.mean(pps_list) if pps_list else 0,
            "cpu_avg": np.mean(cpu) if cpu else 0,
            "mem_p95": np.percentile(mem, 95) if mem else 0,
            "gpu_p95": np.percentile(gpu, 95) if gpu else 0,
        }

    # ---------------------------------------------------------------------
    # High-level test runner
    # ---------------------------------------------------------------------

    def run_server_performance_test(
        self, config: Dict, num_requests: int = 3, model_path: Optional[str] = None
    ) -> Dict:
        """
        Run a short performance test against a llama server started with `config`.

        It starts the server, collects resource metrics in a background thread,
        sends a small batch of requests, then tears down the server and summarizes metrics.

        Returns:
            dict: Summary metrics (tps_avg, gpu_p95, cpu_avg, mem_p95, pps_avg).
        """
        model_path = model_path or self.model_path
        thread = None
        try:
            try:
                thread, metrics = self._start_server_and_metrics_thread(config, model_path)
            except (TimeoutError, ValueError) as e:
                # Return a default "failed" result with a hardware-specific fallback GPU value.
                return {
                    "tps_avg": 0,
                    "gpu_p95": {
                        "m4": 9000.0,
                        "rtx3060": 11000.0,
                        "rtx4090": 22000.0,
                        "orin": 5000.0,
                    }[self.hardware],
                }
            prompts = self.prompt_list[:num_requests]
            tps_list, pps_list = self._send_request_batch(prompts, config)
            time.sleep(0.5)
        finally:
            # Signal metrics thread to stop and join it
            if self.stop_event:
                self.stop_event.set()
            if thread is not None:
                thread.join()
            # Terminate server process if still alive
            if hasattr(self, "server_process") and self.server_process is not None:
                self.server_process.terminate()

        return self._summarize_metrics(metrics, tps_list, pps_list)

    def _run_llama_server(self, config: Dict, model_path: Optional[str] = None) -> subprocess.Popen:
        """Start the llama.cpp server subprocess and wait for it to become ready."""
        model_path = model_path
        cmd = self._build_server_command(config=config, model_path=model_path)

        server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )

        # Wait for server HTTP health endpoint to be ready
        self._wait_for_server_ready(config.get("port", 8080), timeout=30)
        return server_process

    # ---------------------------------------------------------------------
    # Metadata extraction helper
    # ---------------------------------------------------------------------

    def extract_meta_feature(self, config: Dict, model_path: Optional[str] = None) -> subprocess.Popen:
        """
        Start the llama.cpp server process, read stdout/stderr for a short time,
        then kill the process and return collected stderr output.

        This is useful to capture startup logs or static vocabulary/model metadata.
        """
        cmd = self._build_server_command(config=config, model_path=model_path)

        server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )

        try:
            # Wait up to 10 seconds for the process to emit logs
            stdout_output, stderr_output = server_process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            # If not finished in time, kill and gather remaining output
            server_process.kill()
            stdout_output, stderr_output = server_process.communicate()
        except Exception as e:
            logging.exception("Error while running llama server: %s", e)
            server_process.kill()
            stdout_output, stderr_output = "", str(e)

        # Ensure the process is terminated
        if server_process.poll() is None:
            server_process.kill()

        return stderr_output

    # ---------------------------------------------------------------------
    # HTTP request helper
    # ---------------------------------------------------------------------

    def send_request_to_server(self, prompt: str, port: int = 8080, max_tokens: int = 64) -> Tuple[float, int]:
        """
        Send an HTTP request to the running server and read throughput timings.

        Returns:
            Tuple[float, int]: (TPS, PPS). On timeout, returns (1, 1).
        """
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

            timings = response.json()["timings"]
            TPS = timings["predicted_per_second"]
            PPS = timings["prompt_per_second"]
            return TPS, PPS

        except requests.exceptions.Timeout:
            # Return trivial low throughput on timeout
            return 1, 1

    # ---------------------------------------------------------------------
    # Resource collection helpers
    # ---------------------------------------------------------------------

    def collect_server_metrics(self, process: subprocess.Popen) -> Dict:
        """Collect process-level CPU/memory and approximate GPU usage for the server."""
        try:
            proc = psutil.Process(process.pid)
            proc.cpu_percent(interval=None)
            # small sleep to allow cpu_percent to compute
            time.sleep(0.1)
            if self.hardware == "m4":
                gpu_mem = self._get_mac_gpu_memory_usage()
            elif self.hardware == "orin":
                # subtract some baseline offset if necessary
                gpu_mem = self._get_orin_cuda_used_mb() - 2800
            else:
                gpu_mem = self._get_gpu_memory_usage(proc.pid)
            return {
                "cpu_percent": proc.cpu_percent() / psutil.cpu_count(logical=True),
                "system_cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory_mb": proc.memory_info().rss / 1024 / 1024,
                "gpu_mem": gpu_mem,
            }
        except psutil.NoSuchProcess:
            return {"error": "Process not found"}

    def _get_gpu_memory_usage(self, pid):
        """
        Query `nvidia-smi` and grep for the PID's GPU memory usage (MiB).

        Returns:
            int: total GPU memory used by the PID across processes (MiB), or 0 on error.
        """
        try:
            # Use a shell pipeline to query and filter by PID
            cmd = f"nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv | grep {pid}"
            output = subprocess.check_output(
                cmd,
                shell=True,  # use shell so pipe (|) and grep work
                universal_newlines=True,
                stderr=subprocess.DEVNULL,
            )

            total_mem = 0
            for line in output.strip().split("\n"):
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) >= 2:
                    mem_str = parts[1].strip().split()[0]  # e.g. "2500 MiB" -> "2500"
                    total_mem += int(mem_str)
            return total_mem
        except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
            # Handle no match, command failure, parsing errors
            return 0

    def _get_mac_gpu_memory_usage(self):
        """
        Estimate GPU memory usage on macOS by reading wired memory pages.

        Note: macOS doesn't expose GPU VRAM easily; this is an approximation.
        Returns memory in MB.
        """
        try:
            output = subprocess.check_output(["vm_stat"], universal_newlines=True)
            match = re.search(r"Pages wired down:\s+(\d+)", output)
            if match:
                pages = int(match.group(1))
                # macOS page size is typically 4096 bytes
                vram_mb = pages * 4096 / 1024 / 1024
                return int(vram_mb)
            return 0
        except Exception:
            return 0

    def _get_orin_cuda_used_mb(self) -> int:
        """
        Return used GPU memory in MB for Jetson/Orin-like environments where NVML isn't available.

        This function attempts to use libcuda to query free/total memory.
        Returns:
            int: used memory in MB, or 0 on failure.
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

            # Create context (use v2 symbol if available)
            cuCtxCreate = getattr(lib, "cuCtxCreate_v2", getattr(lib, "cuCtxCreate"))
            cuCtxCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
            ctx = ctypes.c_void_p()
            if cuCtxCreate(ctypes.byref(ctx), 0, dev.value) != 0:
                return 0

            # Query free/total memory
            cuMemGetInfo = getattr(lib, "cuMemGetInfo_v2", getattr(lib, "cuMemGetInfo"))
            cuMemGetInfo.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
            free_b = ctypes.c_size_t()
            total_b = ctypes.c_size_t()
            ret = cuMemGetInfo(ctypes.byref(free_b), ctypes.byref(total_b))

            # Destroy context if possible
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

    # ---------------------------------------------------------------------
    # Server readiness probe
    # ---------------------------------------------------------------------

    def _wait_for_server_ready(self, port: int, timeout: int = 30):
        """
        Poll the server /health endpoint until it's ready or timeout elapses.

        Raises:
            TimeoutError: when the server health endpoint did not respond in time.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                if requests.get(f"http://localhost:{port}/health").ok:
                    return
            except requests.ConnectionError:
                time.sleep(1)
        raise TimeoutError("Server startup timed out")


# ---------------------------------------------------------------------
# Example usage (executed when running this file directly)
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Example parameter types definition
    param_types_instance = {
        "gpu-layers": "integer",
        "ctx-size": "integer",
        "no-kv-offload": "boolean",
        "defrag-thold": "float",
        "ubatch-size": "integer",
    }

    # Example configuration list for testing
    config_list = [
        {"gpu-layers": 25, "no-kv-offload": True, "ctx-size": 41, "ubatch-size": 512, "defrag-thold": 0.3},
        {"gpu-layers": 25, "no-kv-offload": True, "ctx-size": 41, "ubatch-size": 512, "defrag-thold": -0.1},
    ]

    executor = LlamaExecutor(
        param_types=param_types_instance,
        model_path="./../models/phimoe-mini-q8.gguf",
        device="gpu",
    )

    results = []
    for config in config_list:
        print("Running test for config:", config)
        result = executor.run_server_performance_test(config)
        results.append(result)
        print("Result:", result)
