import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("torch.version.cuda:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Device count:", torch.cuda.device_count())

    x = torch.ones((3,3), device='cuda')
    print("CUDA tensor OK, sum:", x.sum().item())
