from lfads_tf2.utils import restrict_gpu_usage
restrict_gpu_usage(gpu_ix=0)
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from os import path

from lfads_tf2.models import LFADS
from lfads_tf2.utils import load_data, merge_chops, load_posterior_averages
from lfads_tf2.defaults import get_cfg_defaults, DEFAULT_CONFIG_DIR

# create and train the LFADS model
cfg_path = path.join('/snel/home/brianna/projects/deep_learning_project/ltc_dynamics/ltc_config.yaml')
model = LFADS(cfg_path=cfg_path)
model.train()

# Read config to load data for evalution
cfg = get_cfg_defaults()
cfg.merge_from_file(cfg_path)
cfg.freeze()

# perform posterior sampling
model.sample_and_average()