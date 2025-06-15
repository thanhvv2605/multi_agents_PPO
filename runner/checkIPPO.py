import sys
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from rl_env.WRSN import WRSN
from utils import draw_heatmap_state
from controller.ippo.IPPO import IPPO
import yaml
import torch
import random
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def log(net, mcs):
    # If you want to print something, just put it here. Do not fix the core code.
    while True:
        if net.env.now % 100 == 0:
            print(net.env.now)
        yield net.env.timeout(1.0)

list_map = ["hanoi_50", "hanoi_100", "hanoi_150", "hanoi_200"]

list_result = []
for map in list_map:
    results = []
    scenario_path = f"physical_env/network/network_scenarios/{map}.yaml"
    print("Map Path", scenario_path)

    network = WRSN(scenario_path=scenario_path
                ,agent_type_path="physical_env/mc/mc_types/default.yaml"
                ,num_agent=3, map_size=100,density_map=True)



    with open("alg_args/ippo.yaml", 'r') as file:
        args = yaml.safe_load(file)


    for i in range(1,41):
        iteration_ckpt = i
        path = f"save_model/ippo_50/{iteration_ckpt}"
        controller = IPPO(args['alg_args'], env=network, device=device, model_path=path)
        print("Iteration training:", iteration_ckpt )
        request = network.reset()
        for id, _ in enumerate(network.net.targets_active):
            if _ == 0:
                print(id)
        cnt = 0
        while not request["terminal"]:
            cnt += 1
            # print(request["agent_id"], request["action"], request["terminal"])
            action, _ = controller.get_action(request["agent_id"], request["state"])
            request = network.step(request["agent_id"], action)
            if cnt % 50 == 0:
                print(network.net.env.now)
        object_result = {}
        object_result["iteration"] = iteration_ckpt
        object_result["map"] = map
        object_result["life_time"] = network.net.env.now
        results.append(object_result)
        print("The network life time",network.net.env.now)
        print("Object result", results)
    list_result.append(results)
    

    # Luu vao file json 
    if not os.path.exists("result/ippo_50_target"):
        os.makedirs("result/ippo_50_target") 
    with open(f"result/ippo_50_target/{map}.json", 'w') as f:
        json.dump(results, f)

