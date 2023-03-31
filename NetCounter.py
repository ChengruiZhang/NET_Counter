# This file aims at profile the input, weight memory usage 
# of NN.Module based networks, and provides the function of 
# FLOPs counting of these networks 

# Requirement: only analyze the basic function module of NN.Module, 
# like Conv, Linear, Dropout, etc. 
# Therefore, please ensure that network operations are operated 
# by NN.Module as much as possible 

import torch
from torch import nn

from ptflops import get_model_complexity_info
# from ptflops.pytorch_ops import *
from FLOPs_Counter_Func import *
import pandas as pd


def _init(): 
    global Memory_Count
    global Missing_Module
    # global FLOPs_count

    Memory_Count = pd.DataFrame(columns=["NetName", "InputMemoryUsage-M", "InputShape", "WeightShape", \
        "OutputShape", "WeightMemoryUsage-M", "FLOPsCount-G"])
    Missing_Module = []
    # FLOPs_count = pd.DataFrame(columns=["NetName", "FLOPs_count"])


def Add_Input_Value(value):
    Memory_Count.loc[len(Memory_Count)] = value
    

def Add_Missing_Module(value):
    Missing_Module.append(value)


def get_value(string):
    try:
        return eval(string)
    except:
        print('Load'+'Fail\r\n')


# compute memory and weight usage
def Input_Memory_Hook(network, input, output):
    if type(input) == tuple:
        input = input[0]
    try:
        new_row = {
                    "NetName": network._get_name(),
                    "InputMemoryUsage-M": int(torch.prod(torch.tensor(input.shape))) / 1024 / 1024,
                    "WeightMemoryUsage-M": int(torch.prod(torch.tensor(network.weight.shape))) / 1024 / 1024,
                }
        Add_Input_Value(new_row)
    except:
        new_row = {
                    "NetName": network._get_name(),
                    "InputMemoryUsage-M": int(torch.prod(torch.tensor(input.shape))) / 1024 / 1024,
                    "WeightMemoryUsage-M": 0,
                }
        Add_Input_Value(new_row)


# print inner shape of tuple
def decouple(input):
    input_shape = []
    try:
        if type(input) != tuple:
            return list(input.shape)
        # for tuple
        if input.__len__() > 1:
            for i in input:
                input_shape.append(decouple(i))
        else:
            input_shape.append(input[0].shape)
    except:
        return None
    
    return input_shape


# compute memory, weight usage and FLOPs of each layer
def Input_Memory_FLOPs_Hook(network, input, output):
    
    if network._get_name() == 'MultiheadAttention':
        new_row = {
                    "NetName": network._get_name(),
                    # "InputMemoryUsage-M": int(torch.prod(torch.tensor(input.shape))),
                    "InputMemoryUsage-M": sum(p.numel() for p in input) / 1024 / 1024,
                    "InputShape": [list(p.shape) for p in input],
                    "OutputShape": list(decouple(output)),
                    "WeightShape": [list(p.shape) for p in network.parameters()],
                    "WeightMemoryUsage-M": sum(p.numel() for p in network.parameters() if p.requires_grad) / 1024 / 1024
                }
        new_row["FLOPsCount-G"] = MODULES_MAPPING[type(network)](network, (input[0], input[0], input[0]), output) / 1024 / 1024 / 1024
        Add_Input_Value(new_row)
        return None
        
    # for memory and weight usage
    try:
        
        new_row = {
                    "NetName": network._get_name(),
                    "InputMemoryUsage-M": sum(p.numel() for p in input) / 1024 / 1024,
                    "InputShape": [list(p.shape) for p in input],
                    "OutputShape": list(decouple(output)),
                    "WeightShape": [list(p.shape) for p in network.parameters()],
                    "WeightMemoryUsage-M": sum(p.numel() for p in network.parameters() if p.requires_grad) / 1024 / 1024
                }
    except:
        new_row = {
                    "NetName": network._get_name(),
                    "InputMemoryUsage-M": sum(p.numel() for p in input) / 1024 / 1024,
                    "InputShape": [list(p.shape) for p in input],
                    "OutputShape": list(decouple(output)),
                    "WeightShape": [list(p.shape) for p in network.parameters()],
                    "WeightMemoryUsage-M": 0,
                }
    
    # bias
    try:
        if network.bias is not None:
            new_row["WeightMemoryUsage-M"] += int(torch.prod(torch.tensor(output.shape[2:]))) / 1024 / 1024
    except:
        pass
    
    # for FLOPs counting
    try:
        if type(network) in MODULES_MAPPING or type(network) in CUSTOM_MODULES_MAPPING:
            new_row["FLOPsCount-G"] = MODULES_MAPPING[type(network)](network, input, output) / 1024 / 1024 / 1024
        else:
            # print(f"Missing modules {type(network)}")
            if type(network) in Missing_Module:
                pass
            else:
                Add_Missing_Module(type(network))
            new_row["FLOPsCount-G"] = 0
            new_row["WeightMemoryUsage-M"] = 0
            new_row["WeightShape"] = []
    except:
        new_row["FLOPsCount-G"] = 0
        
    # add value
    Add_Input_Value(new_row)


# add hooks for each layer
def Add_Input_Hook(network: nn.Module): #, Support_module: dict):
    for i in network._modules:
        if isinstance(network._modules[i], nn.Module):
            network._modules[i] = Add_Input_Hook(network._modules[i])# , Support_module)
        else:
            try:
                # network._modules[i].register_forward_hook(Input_Memory_Hook)
                network._modules[i].register_forward_hook(Input_Memory_FLOPs_Hook)
                return network
            except:
                return network
            
    # network.register_forward_hook(Input_Memory_Hook)
    network.register_forward_hook(Input_Memory_FLOPs_Hook)
    
    return network  


 
