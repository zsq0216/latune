import json
import random


class Knobs:
    def __init__(self, json_file, knobs_num, random=False):
        """
        Initialize the Knobs class by loading configuration data from a JSON file.

        Args:
            json_file (str): Path to the JSON configuration file.
            knobs_num (int): Number of configuration entries to parse.
            random (bool): If True, randomly select configuration entries. Defaults to False.
        """
        self.configs = self._load_json(json_file)
        if random:
            self.knobs = self._parse_configs_random(self.configs, knobs_num)
        else:
            self.knobs = self._parse_configs(self.configs, knobs_num)

    def _load_json(self, json_file):
        """
        Load configuration data from a JSON file.

        Args:
            json_file (str): Path to the JSON file.

        Returns:
            dict: Parsed JSON data.

        Raises:
            FileNotFoundError: If the JSON file does not exist.
            ValueError: If the JSON file is not formatted correctly.
        """
        try:
            with open(json_file, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{json_file}' not found.")
        except json.JSONDecodeError:
            raise ValueError(f"File '{json_file}' contains invalid JSON format.")

    def _parse_configs(self, configs, knobs_num):
        """
        Parse a specified number of configuration entries.

        Args:
            configs (dict): Loaded JSON configuration data.
            knobs_num (int): Number of configuration entries to parse.

        Returns:
            dict: A dictionary containing the parsed configuration entries.
        """
        knobs = {}
        count = 0  # Track the number of parsed configurations
        for knob_name, knob_info in configs.items():
            if count >= knobs_num:
                break
            knobs[knob_name] = knob_info
            count += 1
        return knobs

    def _parse_configs_random(self, configs, knobs_num):
        """
        Randomly select and parse a specified number of configuration entries.

        Args:
            configs (dict): Loaded JSON configuration data.
            knobs_num (int): Number of configuration entries to randomly select.

        Returns:
            dict: A dictionary containing the randomly selected configuration entries.
        """
        knobs = {}
        keys = list(configs.keys())
        num = min(knobs_num, len(keys))  # Ensure we don't exceed the total number of available entries

        if num <= 0:
            return knobs  # Handle edge cases (e.g., knobs_num <= 0)

        # Randomly select keys (sample if fewer, otherwise use all)
        selected_keys = random.sample(keys, num) if num < len(keys) else keys

        # Fill the dictionary with the selected entries
        for key in selected_keys:
            knobs[key] = configs[key]

        return knobs

    def get_knob_info(self, knob_name):
        """
        Retrieve detailed information for a specific configuration entry.

        Args:
            knob_name (str): Name of the configuration entry.

        Returns:
            dict: Detailed information about the configuration entry.

        Raises:
            KeyError: If the specified configuration entry is not found.
        """
        if knob_name in self.knobs:
            return self.knobs[knob_name]
        else:
            raise KeyError(f"Configuration entry '{knob_name}' not found.")

    def __repr__(self):
        """
        Return a string representation of the Knobs instance.

        Returns:
            str: String representation showing parsed configuration entries.
        """
        return f"Knobs(configs={self.knobs})"


# Example usage
if __name__ == "__main__":
    # Example JSON file path
    json_file = "knobs_files/cli_knobs_shap.json"

    # Initialize the Knobs class (randomly select 5 configurations)
    knobs = Knobs(json_file, 5, random=True)

    # Retrieve information for a specific knob
    knob_name = "threads"
    knob_info = knobs.get_knob_info(knob_name)

    # Print parsed knob data
    print(knobs.knobs)
