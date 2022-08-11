import gradient_estimators as GE
import torch
import timeit
import numpy as np
import scipy.stats as st

logits = torch.randn([100,3])

estimators = {
    "STGS": GE.STGS(1.0),
    "GRMCK1": GE.GRMCK(1.0, 1),
    "GRMCK10": GE.GRMCK(1.0, 10),
    "GRMCK50": GE.GRMCK(1.0, 100),
    "GST": GE.GST(1.0),
}

REPEATS = 5
NUMBER = 500

for ee in estimators.items():
    times = [time / NUMBER for time in timeit.repeat(lambda: ee[1](logits), repeat=REPEATS, number=NUMBER)]
    time_mean = np.mean(times)
    time_error = st.t.interval(0.95, len(times)-1, loc=time_mean, scale=st.sem(times))

    print(f"{ee[0]}: {time_mean * 1e6} ± {(time_error[1] - time_mean) * 1e6} us")