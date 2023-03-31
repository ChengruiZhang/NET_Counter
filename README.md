# NET_Counter
FLOPs and memory usage (Input and weight) of each network layer.  
Some functions come from ptflops package, in this file we ensure that every submodule can be visited.  

Please notice that we only analyze the memory usage and FLOPs for the basic function module of NN.Module, like Conv, Linear, Dropout, etc.   
Therefore, please ensure that network operations are operated by NN.Module as much as possible  

****Please contact me (zhangchr@shanghaitech.edu.cn) if you have any questions.****

## Demo

from FLOPs_Counter_Func import *  
import pandas as pd  
import NetCounter
NetCounter._init() # initialize global var
  
net = NetworkCreate() # create a new network  
net2 = NetCounter.Add_Input_Hook(net) # add hooks in each layer  
  
net2(SelfBuildInput) # forward  
getattr(NetCounter, "Memory_Count").to_csv("{}_MemoryCount.csv".format(net._get_name())) # save  
print(net._get_name())  
print(NetCounter.Missing_Module)  
  
## Output
pd.DataFrame(columns=["NetName", "InputMemoryUsage-M", "InputShape", "WeightShape", \  
        "OutputShape", "WeightMemoryUsage-M", "FLOPsCount-G"])  
  
Resnet50
  
|Num|NetName|InputMemoryUsage-M|InputShape|WeightShape|OutputShape|WeightMemoryUsage-M|FLOPsCount-G|
|-|-|-|-|-|-|-|-|
|0|Conv2d|0.143554688|[[1, 3, 224, 224]]|[[64, 3, 7, 7]]|[1, 64, 112, 112]|0.008972168|0.109909058|
|1|BatchNorm2d|0.765625|[[1, 64, 112, 112]]|[[64], [64]]|[1, 64, 112, 112]|0.012084961|0.001495361|
|2|ReLU|0.765625|[[1, 64, 112, 112]]|[]|[1, 64, 112, 112]|0|0.000747681|
|3|MaxPool2d|0.765625|[[1, 64, 112, 112]]|[]|[1, 64, 56, 56]|0|0.000747681|
|4|Conv2d|0.19140625|[[1, 64, 56, 56]]|[[64, 64, 1, 1]]|[1, 64, 56, 56]|0.00390625|0.011962891|
|5|BatchNorm2d|0.19140625|[[1, 64, 56, 56]]|[[64], [64]]|[1, 64, 56, 56]|0.003112793|0.00037384|
|6|ReLU|0.19140625|[[1, 64, 56, 56]]|[]|[1, 64, 56, 56]|0|0.00018692|
|7|Conv2d|0.19140625|[[1, 64, 56, 56]]|[[64, 64, 3, 3]]|[1, 64, 56, 56]|0.03515625|0.107666016|
|8|BatchNorm2d|0.19140625|[[1, 64, 56, 56]]|[[64], [64]]|[1, 64, 56, 56]|0.003112793|0.00037384|
|9|ReLU|0.19140625|[[1, 64, 56, 56]]|[]|[1, 64, 56, 56]|0|0.00018692|
|10|Conv2d|0.19140625|[[1, 64, 56, 56]]|[[256, 64, 1, 1]]|[1, 256, 56, 56]|0.015625|0.047851563|
|11|BatchNorm2d|0.765625|[[1, 256, 56, 56]]|[[256], [256]]|[1, 256, 56, 56]|0.003479004|0.001495361|
|12|Conv2d|0.19140625|[[1, 64, 56, 56]]|[[256, 64, 1, 1]]|[1, 256, 56, 56]|0.015625|0.047851563|
|13|BatchNorm2d|0.765625|[[1, 256, 56, 56]]|[[256], [256]]|[1, 256, 56, 56]|0.003479004|0.001495361|
|14|Sequential|0.19140625|[[1, 64, 56, 56]]|[]|[1, 256, 56, 56]|0|0|
|15|ReLU|0.765625|[[1, 256, 56, 56]]|[]|[1, 256, 56, 56]|0|0.000747681|
|16|Bottleneck|0.19140625|[[1, 64, 56, 56]]|[]|[1, 256, 56, 56]|0|0|
|17|Conv2d|0.765625|[[1, 256, 56, 56]]|[[64, 256, 1, 1]]|[1, 64, 56, 56]|0.015625|0.047851563|
|18|BatchNorm2d|0.19140625|[[1, 64, 56, 56]]|[[64], [64]]|[1, 64, 56, 56]|0.003112793|0.00037384|
|19|ReLU|0.19140625|[[1, 64, 56, 56]]|[]|[1, 64, 56, 56]|0|0.00018692|
|20|Conv2d|0.19140625|[[1, 64, 56, 56]]|[[64, 64, 3, 3]]|[1, 64, 56, 56]|0.03515625|0.107666016|
