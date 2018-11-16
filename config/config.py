import os
from os.path import join, dirname
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), '../.env')
load_dotenv(dotenv_path)

debug = eval(os.environ.get("DEBUG"))

data_dir_path = os.environ.get("DATA_DIR_PATH")
fig_dir_path = os.environ.get("FIG_DIR_PATH")