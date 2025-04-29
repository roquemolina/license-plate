**1\. Introduction**
====================

This document outlines the development of an ALPR **(Automatic License Plate Recognition)** system using YOLO for vehicle/plate detection and PaddleOCR for text recognition.

To fully understand the scope and nuances of the project, I'll outline some of the key drivers behind its conception:

1.  **Core Purpose**: The primary goal is to develop an ALPR system.
2.  **Offline Functionality & Portability**: The device must operate in environments without wired internet or alternating current (AC) power. It should be deployable outdoors for street-level installation.
3.  **On-Device Processing**: The system will analyze data locally to detect license plates. Upon identifying a match, it will transmit only essential information---such as the detected license plate number, the corresponding image frame, and the timestamp. This approach minimizes operational costs, as streaming full video footage is prohibitively expensive compared to transmitting concise, relevant data.

**Overview of the General Process**
-----------------------------------

To achieve our primary goal of automatic license plate recognition (ALPR), the workflow involves two key stages:

### **Stage 1: Training the AI Model**

-   **Objective**: Prepare a model to detect and interpret license plates accurately.
-   **Approach**:
    -   **Option 1**: Train a custom model from scratch using proprietary data and tailored configurations.
    -   **Option 2**: Fine-tune a pre-trained model (transfer learning) to adapt it to our specific use case. This balances efficiency and accuracy by leveraging existing architectures.

### **Stage 2: Detection & Recognition**

-   **Workflow**:
    1.  **Detect Vehicles**: Identify vehicles (cars, trucks, etc.) in the frame using a pre-trained **YOLO model** (trained on the COCO dataset).
    2.  **Localize License Plates**: Extract regions of interest (ROIs) containing license plates from detected vehicles.
    3.  **Read License Plates**: Use **PaddleOCR**, an optical character recognition (OCR) tool optimized for low-resolution or distorted text, to decode the alphanumeric characters on the plates.

2\. Installation
================

**Install Python**
------------------

-   **Purpose**: Python is the programming language required to run the application.
-   **Recommendations**:
    -   The project has been tested on **Intel**, **AMD**, and **ARM** CPUs architectures, NVIDIA and integrated GPUs without issues.
    -   **Avoid Python 3.13**: This version is too recent and may introduce compatibility problems. Use **Python 3.11** instead, as it is stable and fully supported by the project's dependencies.
    -   Follow OS-specific installation guides for your system (Windows, macOS, or Linux).

**Set Up a Virtual Environment**
--------------------------------

-   **Purpose**: Virtual environments isolate project dependencies to avoid conflicts with your system-wide Python installation.
-   **Tools**:
    -   **venv** (built-in Python module): Lightweight and sufficient for most use cases.
    -   **Conda** (via Anaconda/Miniconda): Ideal for managing complex dependencies and cross-platform compatibility.
-   **Instructions**:
    -   The project was developed using **venv**, but Conda is equally viable. Refer to the official documentation for [venv](https://docs.python.org/3/library/venv.html) or [Conda](https://docs.conda.io/) based on your preference.
    -   **Critical Reminder**: Always install dependencies (via **`pip`**) **within the activated virtual environment**.

**Install Code Dependencies**
-----------------------------

This step installs the core libraries required for the deployed pipeline. The installation process **varies slightly depending on whether you're using a GPU or CPU** for data processing.

### **Install Node.js**

Node.js is used to run a lightweight backend server that handles incoming license plate data via HTTP requests and save it in a CSV file. Later we can handle data orreceive frames.

-   **OS-Specific Instructions**:
    -   **Windows/macOS**: Download the LTS version from [Node.js Official Site](https://nodejs.org/).
    -   **Linux**: Use your package manager (e.g., **`sudo apt install nodejs npm`** for Ubuntu/Debian).

Install YOLO and paddleOCR Dependencies
---------------------------------------

### **Creating virtual environments**

```
python -m venv /path/to/new/virtual/environment #If using linux use python3

```

### Running venv

| Platform | Shell | Command to activate virtual environment |
| --- | --- | --- |
| POSIX | bash/zsh | `$ source *<venv>*/bin/activate` |
|  | fish | `$ source *<venv>*/bin/activate.fish` |
|  | csh/tcsh | `$ source *<venv>*/bin/activate.csh` |
|  | pwsh | `$ *<venv>*/bin/Activate.ps1` |
| Windows | cmd.exe | `C:\\> *<venv>*\\Scripts\\activate.bat` |
|  | PowerShell | `PS C:\\> *<venv>*\\Scripts\\Activate.ps1` |

When you activate a Python virtual environment (created with **`venv`** or **`conda`**), your command-line interface (CLI) will display the environment's name in parentheses at the beginning of the prompt.

```
(your-venv-name) user@device ~/$

```

### Install depedencies

**opencv <https://pypi.org/project/opencv-python/#frequently-asked-questions**> In Python, `import cv2` is the command used to import the OpenCV library. Once installed, `cv2` gives you access to all the functions and classes that OpenCV offers for image processing, computer vision, and machine learning tasks. In our case, we use it to take a picture of every frame.

```
 pip install opencv-python-headless

```

**request (**simple, yet elegant, HTTP library) <https://pypi.org/project/requests/>

```
pip install requests

```

**pytorch** <https://pytorch.org/get-started/locally/>

PyTorch is a foundational library for our project. Both **Ultralytics** (YOLO models) and **PaddleOCR** rely on PyTorch to process **tensors** -- multidimensional numerical arrays that serve as the primary data structure for machine learning computations. Follow link to install. **Installation Considerations**

**System Requirements**:

-   **Operating System**: Compatible with Windows, Linux, and macOS.
-   **Language**: Python (3.8+ recommended).
-   **Hardware**: CPU or NVIDIA GPU (CUDA-enabled).

**GPU vs. CPU**:

-   **CPU Version**: Use if no NVIDIA GPU is available. Slower but universally compatible.
-   **GPU Version**: Requires **CUDA 11.8**, which ensures compatibility with both Ultralytics and PaddleOCR.

**ultralytics** <https://github.com/ultralytics/ultralytics?tab=readme-ov-file>

**Ultralytics** enables running and training YOLO models. In this deployed project, we use it to load:

1.  **Pre-trained YOLOv8s** (from COCO dataset) for vehicle detection (**`yolo/models/yolo11s.pt`**).
2.  **Custom License Plate Detector** (**`yolo/models/license_plate_small_v1.pt`**), pre-trained by us (4k images, 100 epochs).

**Key Notes**:

-   Both models are used directly---no additional training required for deployment.
-   **Potential Improvements**:
    -   Train a larger license plate model (more epochs + expanded dataset).
    -   Experiment with larger YOLO variants (e.g., YOLOv8m/l) for improved accuracy at a slight speed cost.

```
pip install ultralytics

```

paddleOCR <https://paddlepaddle.github.io/PaddleOCR/latest/en/ppocr/quick_start.html>

The installation process for PaddleOCR depends on whether you intend to use **GPU acceleration** or **CPU-only processing**. Follow these steps:

**Install PaddlePaddle (Base Framework)**

```
pip install paddlepaddle-gpu # For GPU Users (NVIDIA with CUDA 11.x)
or
pip install paddlepaddle # For CPU users

```

**Install PaddleOCR**

```
pip install "paddleocr>=2.0.1" # Recommend to use version 2.0.1+

```

After completing the installation, follow these steps to launch the system:

### **Configuration Setup**

**Server Address**:

-   Update the API endpoint in your Python script (**`API_URL`** variable) to match your server's IP/domain.
-   **For Remote Testing**: Ensure firewall rules allow inbound/outbound traffic on the designated port (e.g., port 3000).

**Input Source**:

-   Configure **`INPUT_SOURCE`** in the Python script to use either:
    -   Local video file (e.g., **`./input.mp4`**)
    -   Camera stream (use **`0`** for default webcam)

### **Launch the Node.js Server**

```
npm install   # Install dependencies if not already done
node app.js

```

### **Run the ALPR Pipeline**

```
# Activate virtual environment first (if using)
source your-venv/bin/activate  # Linux/macOS
your-venv\\Scripts\\activate     # Windows

python3.11 license_reader_v5.py  # Use your actual script filename

```

3\. Get Data to TRAIN:
----------------------

One critical facet of the project involves training custom models. Below is the workflow and key requirements:

**Dataset Preparation**

-   **Option 1: Create our Own Dataset**
    -   **Tools**:

        -   **[CVAT.ai](http://CVAT.ai)** (Computer Vision Annotation Tool): Web-based platform for collaborative labeling.
        -   **LabelImg**: Lightweight desktop tool for bounding box annotation.
    -   **Process**:

        -   Annotate images with bounding boxes (license plates, vehicles).

        -   Generate annotation files (**`.txt`** per image) in YOLO format

            ```
            <class_id> <x_center> <y_center> <width> <height>

            ```

    -   **Note**: Manual annotation is time-intensive. Initial tests used only ~4k images.

-   **Option 2: Use Existing Datasets**
    -   **Sources**:
        -   **Open Images V7**: Large-scale dataset with diverse vehicle classes.
        -   **RoboFlow**: Preprocessed datasets with YOLO-compatible formats.
        -   **Kaggle**: Community-shared datasets (search for "license plates").

### **Training Workflow**

-   **Format Conversion**: Ensure dataset annotations match YOLO requirements.
-   **Configure YOLO**:
    -   Update **`data.yaml`** to specify class names and dataset paths.
    -   Adjust hyperparameters (epochs, batch size) in **`hyp.yaml`**.

1.  **Run Training**:

    ```
    yolo detect train data=config.yaml model=yolov8n.pt epochs=100 imgsz=640 device=0

    ```

**Key Recommendations**

-   **Dataset Size**: Aim for ≥10k annotated images for robust license plate detection.
-   **Model Variants**: Experiment with larger YOLO architectures (e.g., **`yolov8m.pt`**) for improved accuracy.
-   **Validation**: Split data into 80% train, 15% validation, 5% test sets.

**Interesting links:**

<https://storage.googleapis.com/openimages/web/visualizer/index.html?type=detection&set=train&c=%2Fm%2F0pcr&id=4e7c2085ed1a30f9>

To download those images we can use this library, but there is other ones <https://github.com/theAIGuysCode/OIDv4_ToolKit>.

<https://github.com/pjreddie/darknet/blob/master/data/coco.names> COCO NAMES to use library