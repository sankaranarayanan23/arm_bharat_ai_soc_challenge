import time
import numpy as np
import cv2
from driver import FINNExampleOverlay, io_shape_dict

# 1. Setup
accel = FINNExampleOverlay("resizer.bit", "zynq-iodma", io_shape_dict)
hw_shape = accel.ishape_normal()

# 2. Prepare Data (Using a dummy 'deer' frame)
dummy_img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
input_data = np.transpose(dummy_img, (2, 0, 1)).reshape(hw_shape).astype(np.uint8)

# 3. Measurement Loop
pl_latencies = []
for i in range(100):
    start = time.perf_counter()
    _ = accel.execute(input_data)
    end = time.perf_counter()
    pl_latencies.append((end - start) * 1000)

avg_pl = np.mean(pl_latencies)
print(f"FPGA (PL) Average Latency: {avg_pl:.4f} ms")
print(f"FPGA (PL) Throughput: {1000/avg_pl:.2f} FPS")
