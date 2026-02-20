import cv2
import numpy as np
import time

# 1. Setup Camera
cap = cv2.VideoCapture(1) # Change to 0 if 1 doesn't work

# 2. Setup Classes (Same as your FPGA code)
classes = ["Airplane", "Auto", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

prev_time = 0

print("Starting Benchmark... Press 'q' to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret: break

        # --- START INFERENCE SIMULATION ---
        t_start = time.perf_counter()

        # Pre-process
        resized = cv2.resize(frame, (32, 32))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Simulate a prediction result (in a real app, this is where the model runs)
        # We use the mean of the image to pick a "class" for the demo
        avg_val = np.mean(rgb)
        result_idx = int(avg_val % 10) 
        result_text = classes[result_idx]

        t_end = time.perf_counter()
        # --- END INFERENCE SIMULATION ---

        # 3. Calculations
        latency_ms = (t_end - t_start) * 1000
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        # 4. Display Labels on Screen
        # Black background box for high visibility
        cv2.rectangle(frame, (0, 0), (400, 130), (0, 0, 0), -1)

        # Green text for metrics
        cv2.putText(frame, f"Object: {result_text}", (15, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f"Latency: {latency_ms:.2f} ms", (15, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (15, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Laptop CPU Benchmark", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
