---
title: Attention Kernel Optimization - YOLO as context
draft: false
tags:
---
## Backstory
I was performing object detection by running the YOLO inference/prediction method on an image. The images used were approximately 6000 × 3000 pixels in resolution. To achieve optimal performance, I employed the highest-performing models available — YOLOv11-L and YOLOv11-X. However, I soon encountered memory issues, with the traceback indicating that the error occurred during the forward pass of PyTorch’s `nn/Block` module, specifically within one of the following lines -

```python
attn = (q.transpose(-2, -1) @ k) * scale
attn = attn.softmax(dim=-1)
x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + pe(v.reshape(B, C, H, W))
```

## Problem statement
The `ultralytics/nn/modules/block.py` forward call constitutes a critical component of the YOLO inference process. Our objective is to optimize this operation for both memory usage and runtime performance. The optimization strategy specifically targets the memory bottleneck encountered when processing large arrays.
### Closer Look

We begin by establishing the setup of the variables involved and their corresponding shapes relative to the input data:

- `H` is a scalar representing the height scaling of the image.
- `W` is a scalar representing the width scaling of the image.
- `q` is a PyTorch tensor of shape `(B,SZ1,SZ2,H×W)`.
- `k` is a PyTorch tensor of shape `(B,SZ1,SZ2,H×W)`.
- `v` is a PyTorch tensor of shape `(B,SZ1,SZ3,H×W)`.
- `scale` is a scalar value.
- `B` denotes the number of images being processed.
- `C` is defined as `SZ1×SZ3`.
- `pe` refers to a convolution function, where the input and output arrays share identical shapes.

For simplicity — and consistent with the common practice in most inference pipelines — we assume that only one image is processed at a time. We proceed under this assumption.

We now proceed to examine the issue in greater detail. The peak memory usage arises at the operation `(q.transpose(−2,−1) @ k)`, which allocates memory according to the formula -

```latex
1×SZ1×(H×W)×(H×W)×4 bytes
```

which simplifies to -
```latex
4×SZ1×(H×W)2 bytes.
```

As the computation involves array operations, the allocated memory must reside in contiguous memory blocks.

It is important to note that this operation internally leverages cuDNN’s batched matrix multiplication capabilities, as described in the [cuDNN Graph Library - Matmul Descriptor](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-graph-library.html#cudnn-backend-operation-matmul-descriptor). These kernels are highly optimized and provide excellent computational performance. Nevertheless, when operating on very large arrays, memory latency associated with accessing such extensive data becomes the dominant bottleneck.

Moreover, larger YOLO models, such as `YOLOv11-L` and `YOLOv11-X`, necessitate the allocation of even larger intermediate arrays, further amplifying peak memory consumption. This effect is compounded by the additional scaling and softmax operations, each of which also requires the allocation of arrays with similar memory footprints.
### Proposal

In the original implementation, all reduction operations — namely matrix multiplications and the softmax function — are performed along the last two axes of the tensors. Accordingly, a straightforward loop-based reformulation involves iterating over the first two axes while maintaining vectorized operations across the final two axes. The primary objective of this approach is to minimize peak memory consumption and enhance reduction efficiency. Transitioning matrix multiplications to a loop-based structure primarily requires appropriate indexing, whereas adapting the softmax operation, owing to its associative property, necessitates no substantial modifications.

Approach #1 :
```python
R = v.shape[2]
M,N,K,P = q.shape
out_dtype = torch.result_type(q, k)
var2 = torch.empty((M, N, R, P), dtype=out_dtype, device=device)

# Implementation using PyTorch operations but optimized for GPU
for i in range(M):
	for j in range(N):
	    # Compute in one go for all t positions            
	    # Compute q @ k.T for all positions in one go
	    attention_weights = torch.matmul(q[i, j].T, k[i, j])  # P x P
	    attention_weights = attention_weights * self.scale
	    
	    # Apply softmax along appropriate dimension
	    attention_probs = torch.nn.functional.softmax(attention_weights, dim=1)
	    
	    # Compute output for all R dimensions
	    var2[i, j] = torch.matmul(v[i,j], attention_probs.T)  # R x P

x = var2.view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
```

We can introduce optimizations within the nested loops to achieve more efficient memory usage. Specifically, by employing direct indexing and writing the final outputs directly into the tensor `x`, we can reduce intermediate memory overhead. This strategy is illustrated in the subsequent approach.

Approach #2 :
```python
R = v.shape[2]
M,N,K,P = q.shape

x = self.pe(v.reshape(B, C, H, W)).reshape(B,N,R,P)            

# Implementation using PyTorch operations but optimized for GPU
for i in range(M):
	for j in range(N):

		# Compute in one go for all t positions            
		# Compute q @ k.T for all positions in one go
		attention_weights = torch.matmul(q[i, j].T, k[i, j])  # P x P
		attention_weights = attention_weights * self.scale
		
		# Apply softmax along appropriate dimension
		attention_probs = torch.nn.functional.softmax(attention_weights, dim=1)
		
		# Compute output for all R dimensions
		x[i,j] += torch.matmul(v[i,j], attention_probs.T)  # R x P
		
	x = x.reshape(B, C, H, W)
```

In scenarios where memory constraints are particularly severe, we can take the optimization a step further by iterating over the large final axis of `q`. This leads to the development of our final approach.

Approach #3 :
```python
R = v.shape[2]
M,N,K,P = q.shape
out_dtype = torch.result_type(q, k)    
var2 = torch.empty((M, N, R, P), dtype=out_dtype, device=device) 
for i in range(M):
	for j in range(N):
		for t in range(P):
			s1 = ((q[i,j,:,t] @ k[i,j,:,:]) * self.scale).softmax(dim=0)
			s2 = v[i,j] @ s1
			var2[i,j,:,t] = s2

x = var2.view(B, C, H, W)  + self.pe(v.reshape(B, C, H, W))
```

### Benchmarking

As part of the benchmarking process, we aim to evaluate both the runtime and memory improvements. Specifically, we benchmark the original and proposed solutions with respect to memory consumption and execution time. For this evaluation, the input argument `imgsz` is set to 2560 × 2560. The `predict` method in YOLO processes an image through multiple passes within `ultralytics/nn/modules/block.py`, with the number of passes varying based on the model size as compiled below -