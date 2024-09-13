import os
import numpy as np
import torch
import pickle
import functools


def parse_task_code(task_code):
    task_code_list = task_code.split("++")
    process_key = f"{task_code_list[0]}/{task_code_list[1]}"
    task_type = task_code_list[2]
    task_range_str_list = task_code_list[3].split("_")
    task_range = (int(task_range_str_list[0]), int(task_range_str_list[1]))
    task_main_object = task_code_list[4]
    return process_key, task_type, task_range, task_main_object
