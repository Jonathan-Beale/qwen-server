import torch

def test_cuda():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device count:", torch.cuda.device_count())
        print("Current device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
        x = torch.rand(5, 5).cuda()
        print("Tensor on CUDA:", x)
    else:
        print("CUDA is NOT available. Check your driver and Docker setup.")

if __name__ == "__main__":
    test_cuda()
