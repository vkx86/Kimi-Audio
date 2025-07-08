import torch

print(f'is_available {torch.cuda.is_available()}')
print(f'device_count {torch.cuda.device_count()}')
print(f'device_count {torch.cuda.current_device()}')
