import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from configs import Config
from utils import dataclass_from_file, upload_checkpoints

config_path = input("Enter the path to the config file: ")

config = dataclass_from_file(Config, config_path)

run_id = input("Enter the run ID: ")

upload_checkpoints(config, run_id)
