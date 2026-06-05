import torch
import cv2
print("OpenCV 版本:", cv2.__version__)

#环境检查
# 检查 PyTorch 版本
print("PyTorch version:", torch.__version__)

# 检查 PyTorch 支持的 CUDA 版本
print("CUDA version supported by PyTorch:", torch.version.cuda)

# 检查 cuDNN 版本
print("cuDNN version used by PyTorch:", torch.backends.cudnn.version())

# 检查 CUDA 是否可用
print("CUDA is available:", torch.cuda.is_available())

# 获取当前可用 GPU 的数量
print("Number of GPUs available:", torch.cuda.device_count())

# 获取当前使用的 GPU 名称
if torch.cuda.is_available():
    print("Current GPU:", torch.cuda.get_device_name(0))