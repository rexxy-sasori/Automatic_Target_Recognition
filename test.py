from nn.models import ConvLite,ConvNet
import torch
from nn.utils import Profiler
import numpy as np

num_channels = [9,18,27,36,45,54,64]

for c in num_channels:
	m = ConvLite(num_channels=c)
	x = torch.randn(1,9,32,32,device="cpu")
	m = m.to("cpu")
	profiler = Profiler(m)
	profiler_result = profiler.profile(inputs=x)
	
	per_compute_layer_complexity = np.array(profiler_result.per_compute_layer_complexity)
	per_compute_layer_complexity = per_compute_layer_complexity[:, 1::]
	per_compute_layer_complexity = per_compute_layer_complexity.astype(np.float32)
	
	max_act = np.max(per_compute_layer_complexity[:, 2])
	rep_cost = 4 * (profiler_result.num_params + max_act)
	print(rep_cost)

	comp_cost = 0
	prec = 23
	for l in per_compute_layer_complexity:
		mul = l[-1]*l[-2]*prec*prec
		add = l[-2] * (l[-1]-1) * (prec+prec +np.log2(l[-1])-1)
		comp_cost += (mul+add)
	print("{:e}".format(comp_cost))

m = ConvNet(in_channels=9, dropout=0.5)
x = torch.randn(1,9,32,32,device="cpu")
m = m.to("cpu")
profiler = Profiler(m)
profiler_result = profiler.profile(inputs=x)

per_compute_layer_complexity = np.array(profiler_result.per_compute_layer_complexity)
per_compute_layer_complexity = per_compute_layer_complexity[:, 1::]
per_compute_layer_complexity = per_compute_layer_complexity.astype(np.float32)

max_act = np.max(per_compute_layer_complexity[:, 2])
rep_cost = 4 * (profiler_result.num_params + max_act)
print(rep_cost)

comp_cost = 0
prec = 23
for l in per_compute_layer_complexity:
	mul = l[-1]*l[-2]*prec*prec
	add = l[-2] * (l[-1]-1) * (prec+prec +np.log2(l[-1])-1)
	comp_cost += (mul+add)
print("{:e}".format(comp_cost))