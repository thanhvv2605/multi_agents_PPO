import sys
import os
import torch
import random
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from controller.ippo.IPPO import IPPO
import numpy as np
from rl_env.WRSN import WRSN
import yaml



torch.backends.cudnn.deterministic = True
import multiprocessing



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set spawn 
    multiprocessing.set_start_method('spawn', force=True)

    env = WRSN(scenario_path="physical_env/network/network_scenarios/50_target_train.yaml"
                ,agent_type_path="physical_env/mc/mc_types/default.yaml"
                ,num_agent=3, map_size=100, density_map=True)

    with open("alg_args/ippo.yaml", 'r') as file:
        args = yaml.safe_load(file)
    controller = IPPO(args['alg_args'], env=env, device=device)
    controller.train(100000, save_folder="save_model/ippo_50")

