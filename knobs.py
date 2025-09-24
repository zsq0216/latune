import json
import random

class Knobs:
    def __init__(self, json_file, knobs_num, random=False):
        """
        初始化 Knobs 类，从 JSON 文件中加载配置。
        
        参数:
            json_file (str): 包含配置的 JSON 文件路径。
        """
        self.configs = self._load_json(json_file)
        # knobs_num = len(list(self.configs.keys())) #TODO:暂时行为，如果要设计成可调节的，后续考虑
        if random:
            self.knobs = self._parse_configs_random(self.configs, knobs_num)
        else:
            self.knobs = self._parse_configs(self.configs, knobs_num)

    def _load_json(self, json_file):
        """
        从 JSON 文件中加载配置。
        
        参数:
            json_file (str): JSON 文件路径。
        
        返回:
            dict: 加载的 JSON 数据。
        """
        try:
            with open(json_file, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"文件 {json_file} 未找到。")
        except json.JSONDecodeError:
            raise ValueError(f"文件 {json_file} 格式错误。")

    def _parse_configs(self, configs, knobs_num):
        """
        解析配置数据，将每个配置项存储为类的属性。
        
        参数:
            configs (dict): 加载的 JSON 数据。
            knobs_num (int): 需要解析的配置项数量。
        
        返回:
            dict: 解析后的配置项。
        """
        knobs = {}
        count = 0  # 用于计数已解析的配置项数量
        for knob_name, knob_info in configs.items():
            if count >= knobs_num:  # 如果已达到指定数量，停止解析
                break
            knobs[knob_name] = knob_info
            count += 1  # 每解析一个配置项，计数加1
        return knobs

    def _parse_configs_random(self, configs, knobs_num):
        """
        解析配置数据，从配置项中随机抽取指定数量，将每个配置项存储为类的属性。
        
        参数:
            configs (dict): 加载的 JSON 数据。
            knobs_num (int): 需要随机抽取的配置项数量。
        
        返回:
            dict: 解析后的配置项。
        """
        knobs = {}
        keys = list(configs.keys())
        num = min(knobs_num, len(keys))  # 确定实际抽取的数量（避免超过总项数）
        
        if num <= 0:
            return knobs  # 处理边界情况
        
        # 随机抽取键（若 num < 总项数则抽样，否则直接取全部）
        selected_keys = random.sample(keys, num) if num < len(keys) else keys
        
        # 根据选中的键填充结果
        for key in selected_keys:
            knobs[key] = configs[key]
        
        return knobs

    def get_knob_info(self, knob_name):
        """
        获取某个配置项的详细信息。
        
        参数:
            knob_name (str): 配置项名称。
        
        返回:
            dict: 配置项的详细信息。
        """
        if knob_name in self.knobs:
            return self.knobs[knob_name]
        else:
            raise KeyError(f"未找到配置项 '{knob_name}'。")

    def __repr__(self):
        return f"Knobs(configs={self.knobs})"


# 示例用法
if __name__ == "__main__":
    # 假设 JSON 文件路径为 "knobs_config.json"
    json_file = "knobs_files/cli_knobs_shap.json"
    knobs = Knobs(json_file, 5, random=True)

    # 获取某个配置项的信息
    knob_name = "threads"
    knob_info = knobs.get_knob_info(knob_name)
    # print(f"配置项 '{knob_name}' 的信息：")
    # print(knob_info)
    print(knobs.knobs)
