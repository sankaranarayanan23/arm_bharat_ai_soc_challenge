# ARM Bharat AI SoC Challenge — CNN Accelerator

This is my submission for the ARM Bharat AI SoC Challenge. The idea was simple: take a CNN model, quantize it down to 1-bit weights and activations, compile it to run on an FPGA, and see how fast it can classify images compared to a regular CPU.

The model is CNV W1A1 (from Brevitas), trained on CIFAR-10, and deployed on a PYNQ-Z2 board using the FINN compiler. The end result is a live webcam demo where the FPGA classifies objects in real time.

---

## What's in this repo

```
├── brevitas_cnn_accelerator.ipynb   # The main notebook — runs the full FINN build pipeline
├── deployment.py                    # Live webcam inference, meant to run on the PYNQ board
├── benchmark_cpu.py                 # Quick CPU baseline test (runs on your laptop)
├── benchmark_pl.py                  # Measures FPGA accelerator latency on the PYNQ board
├── benchmark_ps.py                  # Measures ARM CPU latency on the PYNQ board (for comparison)
└── deploy-on-pynq-cnv/              # Everything you need to run on the PYNQ board
    ├── resizer.bit / resizer.hwh    # The compiled bitstream
    ├── input.npy                    # A sample CIFAR-10 image (class 3 = Cat)
    ├── driver.py                    # FINN driver with I/O shape config
    ├── driver_base.py               # Base overlay class (from Xilinx)
    ├── validate.py                  # Runs accuracy test on CIFAR-10 or MNIST
    ├── finn/                        # FINN runtime utils
    └── qonnx/                       # QONNX runtime (DataType definitions etc.)
```

---

## How it works

The Jupyter notebook (`brevitas_cnn_accelerator.ipynb`) handles the whole compilation flow — you run it once on your host machine inside the FINN Docker container. It:

1. Downloads the pre-trained CNV W1A1 model from Brevitas
2. Exports it to ONNX and converts it to FINN's internal format
3. Adds preprocessing (uint8 input normalization) and a Top-1 node at the end
4. Streamlines the graph — folds constants, absorbs biases, cleans up
5. Maps everything to hardware layers (matrix-vector units, sliding window generators, etc.)
6. Applies PE/SIMD folding to fit the PYNQ-Z2's resources
7. Runs Vivado synthesis and generates `resizer.bit` + `resizer.hwh`
8. Packages everything into `deploy-on-pynq-cnv.zip`

Once synthesis is done (takes ~30–60 minutes), you copy the zip to the PYNQ board and you're good to go.

---

## Running it

### On the PYNQ board

Copy over the deployment folder and test with the sample input:

```bash
python driver.py --exec_mode execute --inputfile input.npy --outputfile output.npy
```

For full accuracy validation on CIFAR-10:

```bash
python validate.py --dataset cifar10 --batchsize 100
```

For the live webcam demo:

```bash
python deployment.py
```

Press `Ctrl+C` to stop it.

### Benchmarks

Run these to compare performance:

```bash
# FPGA accelerator (on PYNQ board)
python benchmark_pl.py

# ARM CPU on the same board
python benchmark_ps.py

# Laptop CPU (simulated, no model needed)
python benchmark_cpu.py
```

---

## Classes

The model predicts one of 10 CIFAR-10 classes:
`Airplane, Auto, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck`

---

## Notes

- The `driver_base.py` and `validate.py` files are originally from Xilinx (BSD 3-Clause license), slightly adapted for this project.
- The folding config (PE/SIMD values per layer) in the notebook is tuned for the PYNQ-Z2. If you're targeting a different board, you'll need to adjust those.
- `benchmark_cpu.py` uses a simulated prediction (no actual model inference) — it's just to measure the camera/preprocessing pipeline latency on a laptop for comparison.

