Object Detection Benchmarking on Raspberry Pi
This project benchmarks object detection models on a Raspberry Pi using different frameworks and formats, including PyTorch, ONNX, and NCNN. The goal is to find the most efficient model for object detection on a Raspberry Pi.

Project Information
This project was created for the 5th Semester B. Tech Minor Project at the Department of Computer Science and Engineering,

Files
benchmark_rpi_benchmark.py: Benchmarks object detection using the YOLO model on a Raspberry Pi with PyTorch.
benchmark_rpi_ncnn.py: Benchmarks object detection using the YOLO model on a Raspberry Pi with NCNN.
benchmark_rpi_onnx.py: Benchmarks object detection using the YOLO model on a Raspberry Pi with ONNX.
benchmarked-most-efficient.py: Benchmarks the most efficient object detection model.
best_ncnn_model/: Contains the NCNN model files and metadata.
metadata.yaml: Metadata for the NCNN model.
model_ncnn.py: Script for testing NCNN model inference.
model.ncnn.param: NCNN model parameters.
best.onnx: The ONNX model file.
best.pt: The PyTorch model file.
best.torchscript: The TorchScript model file.
README.md: This README file.
Requirements
Python 3.7+
OpenCV
NumPy
PyTorch
Ultralytics YOLO
NCNN
Picamera2 (for Raspberry Pi camera)
Installation
Clone the repository:
git clone <repository-url>
cd <repository-directory>
Install the required packages:
pip install -r requirements.txt
Usage
Run the benchmarking scripts:
For PyTorch:
python benchmark_rpi_benchmark.py
For NCNN:
python benchmark_rpi_ncnn.py
For ONNX:
python benchmark_rpi_onnx.py
Run the script to benchmark the most efficient model:
python benchmarked-most-efficient.py
