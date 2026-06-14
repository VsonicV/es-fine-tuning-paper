import numpy as np


def z_score(seeds_perf, mean_reward, std_reward):
    for k in seeds_perf:
        seeds_perf[k]["norm_reward"] = (seeds_perf[k]["avg_reward"] - mean_reward) / (
            std_reward + 1e-8
        )
        print(f"Seed {k}; avg reward: {seeds_perf[k]["avg_reward"]}; normalized reward: {seeds_perf[k]['norm_reward']}")

    return seeds_perf
