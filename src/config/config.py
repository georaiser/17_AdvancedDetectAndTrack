# -*- coding: utf-8 -*-
import yaml
import os

class ConfigManager:
    def __init__(self, config_dir="src/config", main_config="config.yaml"):
        """Loads the main configuration file and dynamically loads all referenced config files."""
        self.config_dir = config_dir
        self.config = self._load_yaml(os.path.join(config_dir, main_config))

        # Load all referenced config files
        self.sub_configs = {}
        for key, filename in self.config.get("config_files", {}).items():
            file_path = os.path.join(config_dir, filename)
            self.sub_configs[key] = self._load_yaml(file_path)

    def _load_yaml(self, file_path):
        """Helper function to load a YAML file."""
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                return yaml.safe_load(file) or {}
        return {}

    def get(self, section, key, default=None):
        """Get a value from any loaded YAML file."""
        for config in [self.config] + list(self.sub_configs.values()):
            if section in config and key in config[section]:
                return config[section][key]
        return default

    def set(self, section, key, value):
        """exit if value is None"""
        if value is None:
            return False
        """Set a value in the appropriate config."""
        for name, config in self.sub_configs.items():
            if section in config:
                config[section][key] = value
                return True
        return False
    
    def save(self):
        """Save updated configurations back to their respective YAML files."""
        for name, filename in self.config.get("config_files", {}).items():
            file_path = os.path.join(self.config_dir, filename)
            with open(file_path, "w") as file:
                yaml.safe_dump(self.sub_configs[name], file)