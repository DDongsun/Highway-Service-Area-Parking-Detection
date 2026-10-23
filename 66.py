import torch

# 检查PyTorch是否可以使用CUDA
if torch.cuda.is_available():
    print("PyTorch supports GPU!")
    # 打印出可用的GPU数量
    print("Number of GPUs available:", torch.cuda.device_count())
    # 打印当前使用的GPU
    print("Current GPU:", torch.cuda.current_device())
    # 打印GPU的名称
    print("GPU name:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("PyTorch does not support GPU.")