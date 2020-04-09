from pypapi import events, papi_high as high
import numpy as np
from pytranskit.optrans.continuous.radoncdt import RadonCDT
import sys
from skimage.transform import rotate


input_size = int(sys.argv[1])


eps = 1e-6
x0_range = [0, 1]
x_range = [0, 1]
Rdown = 4  # downsample radon projections (w.r.t. angles)
theta = np.linspace(0, 176, 180 // Rdown)
radoncdt = RadonCDT(theta)

I = np.random.rand(input_size, input_size)
template = np.ones(I.shape, dtype=I.dtype)

high.start_counters([events.PAPI_FP_OPS,])
rcdt = radoncdt.forward(x0_range, template / np.sum(template), x_range, I / np.sum(I), False)
print(high.stop_counters()[0]/1e9)




def discrete_radon_transform(image, steps):
    R = np.zeros((steps, image.shape[0]), dtype=np.float32)
    for s in range(steps):
        rotation = rotate(image, -s*180/steps).astype(np.float32)
        R[s] = np.sum(rotation, axis=1)
    return R

I = np.random.rand(input_size, input_size)
high.start_counters([events.PAPI_FP_OPS,])
radon = discrete_radon_transform(I, 90)
print(high.stop_counters()[0]/1e9)
