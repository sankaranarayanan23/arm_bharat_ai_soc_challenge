import cv2
import numpy as np
import matplotlib.pyplot as plt
from driver import FINNExampleOverlay, io_shape_dict
from IPython.display import clear_output, display, Image
# working code 
# 1. Initialize Overlay
accel = FINNExampleOverlay("resizer.bit", "zynq-iodma", io_shape_dict)
classes = ["Airplane", "Auto", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]
hw_shape = accel.ishape_normal()

# 2. Setup Video Capture (0 is usually the default USB camera)
cap = cv2.VideoCapture(0)

print("Starting Live Video. Press 'Interrupt' in Jupyter to stop.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 3. Pre-process Frame
        # Resize to CIFAR-10 dimensions
        resized = cv2.resize(frame, (32, 32))
        rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # 4. Handle NCHW vs NHWC
        if hw_shape[1] == 3: 
            input_data = np.transpose(rgb_img, (2, 0, 1))
        else:
            input_data = rgb_img

        # 5. Final Reshape and Execution
        input_data = input_data.reshape(hw_shape).astype(np.uint8)
        output = accel.execute(input_data)
        
        # 6. Extract Result
        result_idx = int(output[0][0])
        result_text = classes[result_idx]

        # 7. Display Results on the original frame
        cv2.putText(frame, f"Prediction: {result_text}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Stream to Jupyter Notebook
        _, encoded_img = cv2.imencode('.jpg', frame)
        clear_output(wait=True)
        display(Image(data=encoded_img))

except KeyboardInterrupt:
    print("Stopped by User")
finally:
    cap.release()
