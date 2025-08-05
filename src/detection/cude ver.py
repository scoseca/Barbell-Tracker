import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA disponibile: {torch.cuda.is_available()}")
print(f"CUDA versione: {torch.version.cuda}")
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Memoria totale: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Test velocità GPU
x = torch.randn(1000, 1000).cuda()
y = torch.randn(1000, 1000).cuda()
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(50):
    z = torch.matmul(x, y)
end.record()
torch.cuda.synchronize()
print(f"Tempo operazione: {start.elapsed_time(end):.2f} ms")