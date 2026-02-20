import time
import numpy as np

# 1. Setup (Simulate a weight matrix for a 32x32x3 input)
input_size = 32 * 32 * 3
hidden_nodes = 512  # Simulating a medium-sized layer
weights = np.random.rand(input_size, hidden_nodes).astype(np.float32)
flat_input = input_data.flatten().astype(np.float32)

# 2. Measurement Loop
ps_latencies = []
for i in range(100):
    start = time.perf_counter()
    # Matrix Multiplication is the core of CNN math
    _ = np.dot(flat_input, weights) 
    end = time.perf_counter()
    ps_latencies.append((end - start) * 1000)

avg_ps = np.mean(ps_latencies)
print(f"CPU (PS) Average Latency: {avg_ps:.4f} ms")
print(f"CPU (PS) Throughput: {1000/avg_ps:.2f} FPS")
