import re
import copy
import json
import hashlib
import itertools
import numpy as np
from pathlib import Path
from typing import Dict, List
from yaml import load, FullLoader


class MultivalueYaml:
    variable_template = r"{{(.*)}}"
    range_template = r"range\((.*),(.*),(.*)\)"
    list_template = r"\[(.*)\]"
    boolean_template = r"^((true)|(false)|,)*$"

    def parse(self, yaml_file: Path) -> List[Dict]:
        with open(yaml_file, "r") as yamlfile:
            data = load(yamlfile, Loader=FullLoader)
            multiples = self.find_multiples(data)
            return self.generate_configs(data=data, multiples=multiples)

    def find_multiples(self, data: Dict) -> Dict:
        multiples = {}
        for key, val in data.items():
            if isinstance(val, str):
                result = self.evaluate_val(val)
                if result is not None:
                    multiples[key] = result
            if isinstance(val, dict):
                child_multiples = self.find_multiples(val)
                for mul_key, mul_val in child_multiples.items():
                    multiples[f"{key}.{mul_key}"] = mul_val
        return multiples

    def evaluate_val(self, val: str):
        matches = re.search(self.variable_template, val)
        if matches is None:
            return None
        match = matches[1].strip()
        range_matches = re.search(self.range_template, match)
        if range_matches is not None:
            start, end, step = range_matches[1], range_matches[2], range_matches[3]
            if "." in f"{start}{end}{step}":
                map_fun = float
            else:
                map_fun = int
            return np.arange(map_fun(start), map_fun(end), map_fun(step)).tolist()
        list_matches = re.search(self.list_template, match)
        if list_matches is not None:
            list_input = list_matches[1]
            if re.match(r"^[0-9\,\.\-]*$", list_input):
                if "." in list_input:
                    return list(map(float, list_input.split(",")))
                else:
                    return list(map(int, list_input.split(",")))
            else:
                return list(map(str.strip, list_input.split(",")))
        bool_matches = re.search(self.boolean_template, match)
        if bool_matches is not None:
            bools = []
            if bool_matches[2] == "true":
                bools += [True]
            if bool_matches[3] == "false":
                bools += [False]
            return bools

    def generate_configs(self, data, multiples) -> List[Dict]:
        if len(multiples) == 0:
            return [data]
        keys, vals = zip(*multiples.items())
        configs = []
        for vals in list(itertools.product(*vals)):
            configs += [self.update_yaml(data, dict(zip(keys, vals)))]
        return configs

    def update_yaml(self, data: Dict, updates: Dict) -> Dict:
        clone = copy.deepcopy(data)
        update_key = hashlib.sha256(json.dumps(updates).encode("utf-8")).hexdigest()
        prev_key = clone["key"]
        clone["key"] = f"{prev_key}_{update_key}"
        for key, val in updates.items():
            parent = clone
            key_parts = key.split(".")
            for key_part in key_parts[:-1]:
                parent = parent[key_part]
            parent[key_parts[-1]] = val
        return clone
