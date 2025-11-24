import torch
import sys

if len(sys.argv) > 1:
    matrix_size = int(sys.argv[1])
    print(f"Performing matrix multiplication of two {matrix_size}x{matrix_size} matrices...")
else:
    print("No arguments provided. Quiting ...")
    sys.exit()

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")


# Initialize CUDA streams
s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()
s3 = torch.cuda.Stream()
s4 = torch.cuda.Stream()

# Allocate matrices in the GPU memory
A1 = torch.randn(matrix_size, matrix_size, device=device)
B1 = torch.randn(matrix_size, matrix_size, device=device)
A2 = torch.randn(matrix_size, matrix_size, device=device)
B2 = torch.randn(matrix_size, matrix_size, device=device)

A3 = torch.randn(matrix_size, matrix_size, device=device)
B3 = torch.randn(matrix_size, matrix_size, device=device)
A4 = torch.randn(matrix_size, matrix_size, device=device)
B4 = torch.randn(matrix_size, matrix_size, device=device)

# Perform matrix multiplication
print(f"Performing matrix multiplication of two {matrix_size}x{matrix_size} matrices...")
with torch.cuda.stream(s1):
    C1 = torch.matmul(A1, B1)

with torch.cuda.stream(s2):
    C2 = torch.matmul(A2, B2)

with torch.cuda.stream(s3):
    C3 = torch.matmul(A3, B3)

with torch.cuda.stream(s4):
    C4 = torch.matmul(A4, B4)
torch.cuda.synchronize()
