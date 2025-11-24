import torch
import time
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

A1 = torch.randn(matrix_size, matrix_size, device=device)
B1 = torch.randn(matrix_size, matrix_size, device=device)

print(f"Allocated memory: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")

# Perform matrix multiplication
print(f"Performing matrix multiplication of two {matrix_size}x{matrix_size} matrices...")
start_time = time.perf_counter()

C1 = torch.matmul(A1, B1)

torch.cuda.synchronize()
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time:.6f} seconds")
