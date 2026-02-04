# Automatic Extraction and Quantification of Long-term Changes in Map Features Using the Gaihouzu Digital Archive

This repository contains the source code for the graduation thesis "Automatic Extraction and Quantification of Long-term Changes in Map Features Using the Gaihouzu Digital Archive".

## Overview
This project aims to automatically extract and quantify map features (e.g., shrines, schools, fields) from "Gaihouzu" (Japanese military maps created during the Meiji to Showa periods) to analyze long-term land use changes.

The system utilizes deep learning models to detect map symbols and extract boundaries, comparing them with modern geospatial data.

### Key Features
* **Symbol Detection:** Uses **YOLOv8** to detect specific map symbols (shrines, schools, etc.).
* **Boundary Extraction:** Uses **DexiNed** for precise edge and boundary detection.
* **Georeferencing:** Alignment of historical maps with modern coordinates using QGIS/Python.

## Requirements
* **Python 3.10.12**
* CUDA-enabled GPU is recommended for training and inference.

## Installation

1.  Clone this repository:
    ```bash
    git clone [https://github.com/YuyaTake-h15/gaihouzu-thesis.git](https://github.com/YuyaTake-h15/gaihouzu-thesis.git)
    cd gaihouzu-thesis
    ```

2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

    *Note: If you haven't generated a requirements.txt yet, install the main dependencies manually:*
    ```bash
    pip install ultralytics opencv-python numpy pandas matplotlib geopandas
    ```

## Usage

1.  **Data Preparation (Gaihouzu Images):**
    Place your map image files in the `input/` directory.
    ```bash
    # Step 1: Resize the image if the input size is large (e.g., 15000x10000).
    python resize_img.py --source input/sample_map.jpg

    # Step 2: Generate 256x256 tiles from the map.
    python split_tiles.py --source input/sample_map.jpg
    ```
    *Note: For training, place your split image files in `data/images/train` and `data/images/val` with a 7:3 ratio.*

2.  **Ground Truth Preparation (Reference Data):**
    Download the school facility data from [MLIT National Land Numerical Information](https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-P29-2023.html) and place it in the directory.
    ```bash
    # Step 1: Get the x and y coordinates of the map corners.
    python get_point.py --source input/sample_map.jpg

    # Step 2: Perform georeferencing in QGIS and save the GCP file.
    # (Manual operation in QGIS required)

    # Step 3: Create shrine and temple datasets from GSI Vector Tiles (地理院ベクター).
    python gsi_vector_poi.py

    # Step 4: Create agriculture symbol datasets from GSI Vector Tiles.
    python agriculture_fixed.py
    ```

3.  **Inference (Detection):**
    Run the detection script to identify map symbols.
    ```bash
    # Step 1: Detect symbols and verify accuracy.
    python predict_symbols.py --source data/images/sample_map.jpg
    python confirm_combine_file.py --source output/all_detections_full_coords_clean.csv

    # Step 2: Convert coordinates using GCP data.
    python convert_point.py

    # Step 3: Generate binary images via inpainting for DexiNed.
    # Note: Ensure arguments are correct for your script.
    python dex_edge.py --image data/images/sample_map.jpg --csv all_detections_full_coords_clean.csv
    ```

4.  **Analysis:**
    Run the analysis script to quantify the changes.
    ```bash
    # Step 1: Detect existence of schools, shrines, and temples.
    python compare_existence.py

    # Step 2: Perform segmentation only (no comparison).
    python gaihouzu_seg_only.py

    # Step 3: Compare polygon and facility results.
    python agri_seg.py
    ```

## Author
* **Yuya Takeda**
* Yamaguchi University
* Kumamoto University (Incoming)

## License
[MIT License](LICENSE) - feel free to use this code for academic purposes.