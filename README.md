# NET_Counter
FLOPs and memory usage (Input and weight) of each network layer 
Some functions come from ptflops package, in this file we ensure that every submodule can be visited.

## Demo

from FLOPs_Counter_Func import *  
import pandas as pd  
  
net = NetworkCreate() # create a new network  
net2 = NetCounter.Add_Input_Hook(net) # add hooks in each layer  
  
net2(SelfBuildInput) # forward  
getattr(NetCounter, "Memory_Count").to_csv("{}_MemoryCount.csv".format(net._get_name())) # save  
  
## Output
pd.DataFrame(columns=["NetName", "InputMemoryUsage-M", "InputShape", "WeightShape", \  
        "OutputShape", "WeightMemoryUsage-M", "FLOPsCount-G"])  
  
Resnet50
  
|Num|NetName|InputMemoryUsage-M|InputShape|WeightShape|OutputShape|WeightMemoryUsage-M|FLOPsCount-G|
|-|-|-|-|-|-|-|-|
|0|Conv2d|0.143554688|[[1, 3, 224, 224]]|[[768, 3, 16, 16], [768]]|[1, 768, 14, 14]|0.563419342|0.107806206|
|1|Dropout|0.144287109|[[1, 197, 768]]|[]|[1, 197, 768]|0|0|
|2|LayerNorm|0.144287109|[[1, 197, 768]]|[[768], [768]]|[1, 197, 768]|0.002197266|0.000281811|
|3|MultiheadAttention|0.432861328|[[1, 197, 768], [1, 197, 768], [1, 197, 768]]|[[2304, 768], [2304], [768, 768], [768]]|[[1, 197, 768], None]|2.252929688|0.489516299|
|4|Dropout|0.144287109|[[1, 197, 768]]|[]|[1, 197, 768]|0|0|
|5|LayerNorm|0.144287109|[[1, 197, 768]]|[[768], [768]]|[1, 197, 768]|0.002197266|0.000281811|
|6|Linear|0.144287109|[[1, 197, 768]]|[[3072, 768], [3072]]|[1, 197, 3072]|2.255859375|0.432864189|
|7|GELU|0.577148438|[[1, 197, 3072]]|[]|[1, 197, 3072]|0|0.000563622|
|8|Dropout|0.577148438|[[1, 197, 3072]]|[]|[1, 197, 3072]|0|0|
|9|Linear|0.577148438|[[1, 197, 3072]]|[[768, 3072], [768]]|[1, 197, 768]|2.251464844|0.432862043|
|10|Dropout|0.144287109|[[1, 197, 768]]|[]|[1, 197, 768]|0|0|
|11|MLPBlock|0.144287109|[[1, 197, 768]]|[]|[1, 197, 768]|0|0|
|12|EncoderBlock|0.144287109|[[1, 197, 768]]|[]|[1, 197, 768]|0|0|
|13|LayerNorm|0.144287109|[[1, 197, 768]]|[[768], [768]]|[1, 197, 768]|0.002197266|0.000281811|
|14|MultiheadAttention|0.432861328|[[1, 197, 768], [1, 197, 768], [1, 197, 768]]|[[2304, 768], [2304], [768, 768], [768]]|[[1, 197, 768], None]|2.252929688|0.489516299|
|15|Dropout|0.144287109|[[1, 197, 768]]|[]|[1, 197, 768]|0|0|
|16|LayerNorm|0.144287109|[[1, 197, 768]]|[[768], [768]]|[1, 197, 768]|0.002197266|0.000281811|
|17|Linear|0.144287109|[[1, 197, 768]]|[[3072, 768], [3072]]|[1, 197, 3072]|2.255859375|0.432864189|
|18|GELU|0.577148438|[[1, 197, 3072]]|[]|[1, 197, 3072]|0|0.000563622|
|19|Dropout|0.577148438|[[1, 197, 3072]]|[]|[1, 197, 3072]|0|0|
|20|Linear|0.577148438|[[1, 197, 3072]]|[[768, 3072], [768]]|[1, 197, 768]|2.251464844|0.432862043|
