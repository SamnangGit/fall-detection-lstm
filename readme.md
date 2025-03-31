# Vision-Based Fall Detection using Pose Estimation and Bidirectional LSTMs

This project implements a system for detecting human falls from video sequences using computer vision and deep learning techniques. It leverages Mediapipe for pose estimation, extracts relevant skeletal features, applies feature engineering (normalization, velocity), and trains a Bidirectional LSTM (BiLSTM) model to classify actions into different fall types and non-fall activities.

## Features

*   **Pose Estimation:** Uses Google's Mediapipe Pose to extract 33 keypoints from video frames.
*   **Feature Engineering:**
    *   Normalizes keypoint coordinates relative to the hip center for position invariance.
    *   Calculates landmark velocity to capture motion dynamics.
    *   Combines normalized coordinates and velocity into a 132-dimensional feature vector per frame.
*   **Sequence Modeling:** Employs a Bidirectional LSTM (BiLSTM) network to learn temporal patterns from sequences of 30 frames.
*   **Class Imbalance Handling:** Uses weighted cross-entropy loss during training to give appropriate importance to minority fall classes.
*   **Modular Pipeline:** Includes separate notebooks/scripts for:
    1.  Feature Extraction (`feature_extraction.ipynb`)
    2.  Model Training (`train_model.ipynb`)
    3.  Manual Inference on a single video (`manually_test_model.ipynb`)
*   **Provided Resources:** Includes links to the raw dataset, pre-processed features, and the trained model weights.

## Project Structure
```
├── processed_features/ # Directory containing extracted .pkl features (used by v2)
│ ├── backward_fall/
│ ├── forward_fall/
│ ├── side_fall/
│ └── non_fall/
├── v1/ # Older version (Python scripts)
│ ├── manually_test_model.py
│ ├── SCH_Fall_Detection_with_Feature_Extracti... # (Likely combined script)
│ └── train_model.py
├── v2/ # Current version (Jupyter Notebooks - improved readability)
│ ├── feature_extraction.ipynb
│ ├── manually_test_model.ipynb
│ └── train_model.ipynb
├── requirements.txt # Python package dependencies
└── README.md # This file
```
## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/SamnangGit/fall-detection-lstm.git
    cd fall-detection-lstm/v2
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # Activate the environment (Linux/macOS)
    source venv/bin/activate
    # Activate the environment (Windows)
    .\venv\Scripts\activate
    ```

3.  **Install dependencies:**
    Verify `requirements.txt` file  in your project root directory. Then, install the required packages using pip:
    ```bash
    pip install -r requirements.txt
    ```

## Data and Model Access

The raw video dataset, the pre-processed `.pkl` features, and the final trained model (`.pth`) file are available on Google Drive:

*   **Google Drive Link:** [https://drive.google.com/drive/folders/1x7eaI-zeiLexlL_NyHjuFsU7NGrupeFb?usp=sharing](https://drive.google.com/drive/folders/1x7eaI-zeiLexlL_NyHjuFsU7NGrupeFb?usp=sharing)

**Instructions:**

1.  Download the necessary files/folders from the Google Drive link.
2.  Place the `processed_features` folder (containing class subdirectories with `.pkl` files) at the root of the project directory or update the `FEATURES_DIR` path in `train_model.ipynb`.
3.  Place the trained model file (e.g., `fall_detection_rnn_local_weighted_loss.pth`) inside a `trained_models_local` directory at the root, or update the `MODEL_SAVE_DIR` / `model_file_path` in `train_model.ipynb` and `manually_test_model.ipynb`.
4.  If you intend to run feature extraction yourself, download the raw video dataset and place the class folders inside `dataset/train_data` and `dataset/test_data` (or update paths in `feature_extraction.ipynb`).

## Usage

**⚠️ Important:** The Jupyter notebooks (`.ipynb` files) contain hardcoded absolute paths (e.g., `/Users/samnangpheng/...`). **You MUST update these paths** in the configuration cells of each notebook to match the location of the data, features, and models on your local machine.

1.  **Feature Extraction (Optional - if not using provided features):**
    *   Ensure raw video data is correctly placed in the `dataset` directory structure.
    *   Open and run the cells in `feature_extraction.ipynb`.
    *   **Modify paths** inside the notebook (Cell 2: `BASE_DRIVE_PATH`, `TRAIN_DATA_DIR`, `TEST_DATA_DIR`, `FEATURES_OUTPUT_DIR`) before running.
    *   This will populate the `processed_features` directory with `.pkl` files.

2.  **Model Training (Optional - if not using provided model):**
    *   Ensure the `processed_features` directory is populated (either from download or Step 1).
    *   Open and run the cells in `train_model.ipynb`.
    *   **Modify paths** inside the notebook (Cell 1: `BASE_PROJECT_PATH`, `FEATURES_DIR`, `MODEL_SAVE_DIR`) before running.
    *   This will train the model and save the best checkpoint (`.pth` file) to the `trained_models_local` directory, along with generating results plots.

3.  **Inference on a Single Video:**
    *   Ensure you have the trained model (`.pth` file) in the `trained_models_local` directory.
    *   Open `manually_test_model.ipynb`.
    *   **Modify paths** in the user input section (Cell 6: `video_to_predict_path`, `model_file_path`) to point to your desired input video and the trained model file.
    *   Run the cells in the notebook.
    *   The output will show the predicted class, confidence score, and class probabilities for the input video.

## Results Summary

The model trained using the settings in `train_model.ipynb` (v3 - Weighted Loss Attempt) achieved the following performance on the unseen test set:

*   **Test Accuracy:** 80.51%
*   **Test Loss:** 0.5476 (Weighted)

Refer to the detailed report in [Samnang_Pheng_Fall_detection_report.pdf](./Samnang_Pheng_Fall_detection_report.pdf).  
An example inference output is shown below:
```
--- Prediction Result ---
Detected Class: non_fall
Confidence: 0.9975
Class Probabilities:
backward_fall: 0.0001
forward_fall: 0.0011
side_fall: 0.0013
non_fall: 0.9975
```