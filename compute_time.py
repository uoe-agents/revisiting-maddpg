import gradient_estimators as GE
import torch
import timeit
import numpy as np
import scipy.stats as st

dims = [3,5,10,50,100,1000]

estimators = {
    "STGS": GE.STGS(1.0),
    "MCK1": GE.GRMCK(1.0, 1),
    "MK10": GE.GRMCK(1.0, 10),
    "MK50": GE.GRMCK(1.0, 100),
    "GST": GE.GST(1.0),
}

REPEATS = 5
NUMBER = 10_000

first_time = True
for dim in dims:
    logits = torch.randn([1,dim])
    if first_time:
        print(str(dim) + ",x",end=",")
        first_time = False
    else:
        print(str(dim) + ",",end=",")
    
    baseline_mean = -1
    multiplier = -1

    for ee in estimators.items():
        times = [time / NUMBER for time in timeit.repeat(lambda: ee[1](logits), repeat=REPEATS, number=NUMBER)]
        time_mean = np.mean(times)
        time_error = st.t.interval(0.95, len(times)-1, loc=time_mean, scale=st.sem(times))

        if ee[0] == "STGS":
            baseline_mean = time_mean
            multiplier = 1.00
        else:
            multiplier = np.round(time_mean / baseline_mean, 2)
        
        mean = np.round(time_mean * 1e6, 2)
        error =  np.round((time_error[1] - time_mean) * 1e6, 2)
        print(mean,error,multiplier,sep=",",end=",")

    print()