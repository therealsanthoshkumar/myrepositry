#!/usr/bin/env python3
"""
CPU inference on Kria KV260 (2022.2 DPU Image)
Mirror of DPU benchmark for fair comparison
NumPy 2.x compatible | No OpenCV | CPUExecutionProvider only
"""
import onnxruntime as ort
import numpy as np
import time
import psutil
import os
from PIL import Image

# ---------------- CONFIG ----------------
ONNX_PATH   = "waste_classifier.onnx"
IMAGE_PATH  = "val"
RUNS        = 30
IMG_SIZE    = 224
# ----------------------------------------

def preprocess(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)   # add batch dim
    return img

print("Loading ONNX model...")
sess_options = ort.SessionOptions()
sess_options.inter_op_num_threads = 4
sess_options.intra_op_num_threads = 4
sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

session = ort.InferenceSession(
    ONNX_PATH,
    sess_options=sess_options,
    providers=["CPUExecutionProvider"]
)

input_name   = session.get_inputs()[0].name
input_shape  = tuple(session.get_inputs()[0].shape)
output_shape = tuple(session.get_outputs()[0].shape)

print("Input shape :", input_shape)
print("Output shape:", output_shape)

process = psutil.Process(os.getpid())

image_tensor = preprocess(IMAGE_PATH)

# Warmup pass (excluded from benchmark, mirrors DPU runner init behaviour)
session.run(None, {input_name: image_tensor})

latencies, cpu_usage, mem_usage = [], [], []

print("Starting CPU inference...\n")

for _ in range(RUNS):
    cpu_before = process.cpu_percent(interval=None)
    mem_before = process.memory_info().rss / (1024 * 1024)

    start = time.time()
    output = session.run(None, {input_name: image_tensor})
    end   = time.time()

    cpu_after = process.cpu_percent(interval=None)
    mem_after = process.memory_info().rss / (1024 * 1024)

    latencies.append((end - start) * 1000)
    cpu_usage.append(cpu_after)
    mem_usage.append(mem_after)

print("========== CPU INFERENCE RESULTS (KV260) ==========")
print(f"Average latency   : {np.mean(latencies):.2f} ms")
print(f"Average CPU usage : {np.mean(cpu_usage):.2f} %")
print(f"Average memory    : {np.mean(mem_usage):.2f} MB")
print("==================================================")