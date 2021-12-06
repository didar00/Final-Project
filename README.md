# Final-Project
Parallel Training on GPU with CUDA

#### Using Accelerate
Multi GPUs with Pytorch launcher
```
python -m torch.distributed.launch --nproc_per_node 2 --use_env ./train_model_gpu.py
```

Single GPU
```
python ./train_model_gpu.py  # from a server with a GPU
```
