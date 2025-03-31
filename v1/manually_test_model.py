# -*- coding: utf-8 -*-
import os
import sys
import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
import argparse
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

# ==============================================================================
# Configuration (MUST MATCH TRAINING CONFIGURATION of v2 model)
# ==============================================================================
# --- Model Architecture Parameters ---
# !! IMPORTANT: These MUST match the parameters used to train the saved model !!
INPUT_SIZE = 33 * 4  # 132 (33 landmarks * [norm_x, norm_y, vel_x, vel_y])
HIDDEN_SIZE = 192
NUM_LAYERS = 2
NUM_CLASSES = 4      # Number of classes model was trained on
RNN_TYPE = 'LSTM'    # Or 'GRU', depending on the saved model
DROPOUT_PROB = 0.4
BIDIRECTIONAL = True # Model was trained bidirectionally

# --- Preprocessing Parameters ---
# !! IMPORTANT: These MUST match the parameters used during feature extraction/training !!
FRAME_SKIP = 3 
SEQUENCE_LENGTH = 30 
ORIGINAL_LANDMARK_DIM = 3 
NUM_LANDMARKS = 33 

# --- Class Labels ---
CLASS_NAMES = ["backward_fall", "forward_fall", "side_fall", "non_fall"]
index_to_name = {i: name for i, name in enumerate(CLASS_NAMES)}

# --- Device Configuration ---
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

LEFT_HIP_IDX = 23
RIGHT_HIP_IDX = 24


# ==============================================================================
# Model Definition (Copied EXACTLY from the training script v2)
# ==============================================================================
class FallDetectionRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, rnn_type='LSTM', dropout_prob=0.2, bidirectional=True):
        super(FallDetectionRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        rnn_dropout = dropout_prob if num_layers > 1 else 0
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                               batch_first=True, dropout=rnn_dropout,
                               bidirectional=bidirectional)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers,
                              batch_first=True, dropout=rnn_dropout,
                              bidirectional=bidirectional)
        else:
            raise ValueError("Unsupported RNN type. Choose 'LSTM' or 'GRU'.")

        self.dropout = nn.Dropout(dropout_prob)
        # Input to FC layer is hidden_size * num_directions
        self.fc = nn.Linear(hidden_size * self.num_directions, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        # Initialize hidden state for bidirectional
        h0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(x.device)
        if self.rnn_type == 'LSTM':
            c0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(x.device)
            hidden = (h0, c0)
        else: # GRU
            hidden = h0

        # RNN output shape: (batch, seq_len, num_directions * hidden_size)
        # hidden state shape: (num_layers * num_directions, batch, hidden_size)
        out, _ = self.rnn(x, hidden)

        # Use the output of the last time step
        # This contains the concatenation of the last forward and backward hidden states
        last_step_out = out[:, -1, :]
        last_step_out = self.dropout(last_step_out)
        out = self.fc(last_step_out)
        return out


# ==============================================================================
# Feature Extraction Function (MODIFIED to include Normalization & Velocity)
# ==============================================================================

def _normalize_and_calculate_velocity(sequence_data_raw, sequence_length, input_size, num_landmarks, orig_landmark_dim):
    """
    Normalizes pose relative to hip center, calculates velocity, and combines them.
    Mirrors the logic from the training Dataset.

    Args:
        sequence_data_raw (np.ndarray): Raw landmark data, shape (extracted_frames, num_landmarks * orig_dim)
                                        or (extracted_frames, num_landmarks, orig_dim).
        sequence_length (int): Target sequence length (e.g., 30).
        input_size (int): Final input size for the model (e.g., 132).
        num_landmarks (int): Number of landmarks (e.g., 33).
        orig_landmark_dim (int): Dimensions per landmark in raw data (e.g., 3 for x, y, vis).

    Returns:
        np.ndarray: Processed sequence, shape (sequence_length, input_size), or None if error.
    """
    num_extracted_frames = sequence_data_raw.shape[0]
    if num_extracted_frames == 0:
        print("Warning: _normalize_and_calculate_velocity received empty raw sequence.")
        return np.zeros((sequence_length, input_size), dtype=np.float32) # Return zeros

    # Expected input shape for processing: (num_extracted_frames, num_landmarks, orig_dim)
    try:
        if sequence_data_raw.ndim == 2: # If it's flattened (frames, landmarks * dim)
             sequence_reshaped_raw = sequence_data_raw.reshape(
                 num_extracted_frames, num_landmarks, orig_landmark_dim
             )
        elif sequence_data_raw.ndim == 3: # If it's already (frames, landmarks, dim)
            sequence_reshaped_raw = sequence_data_raw
        else:
            raise ValueError(f"Unexpected raw data shape: {sequence_data_raw.shape}")
    except ValueError as e:
         print(f"Error reshaping raw sequence data. Shape: {sequence_data_raw.shape}. Error: {e}")
         return np.zeros((sequence_length, input_size), dtype=np.float32) # Return zeros

    # --- Pad or Truncate the reshaped raw sequence to SEQUENCE_LENGTH ---
    padded_sequence_raw = np.zeros((sequence_length, num_landmarks, orig_landmark_dim), dtype=np.float32)
    len_to_copy = min(num_extracted_frames, sequence_length)
    padded_sequence_raw[:len_to_copy] = sequence_reshaped_raw[:len_to_copy]
    # Note: Padding remaining frames with zeros is implicit here.

    # --- Calculate Normalized Coordinates and Velocity ---
    normalized_coords = np.zeros((sequence_length, num_landmarks, 2), dtype=np.float32) # Store norm_x, norm_y
    velocities = np.zeros((sequence_length, num_landmarks, 2), dtype=np.float32) # Store vel_x, vel_y
    last_norm_coords = None

    for t in range(sequence_length):
        frame_data = padded_sequence_raw[t] # (num_landmarks, orig_dim)

        # Calculate hip center for normalization
        left_hip = frame_data[LEFT_HIP_IDX, :2] # x, y
        right_hip = frame_data[RIGHT_HIP_IDX, :2] # x, y

        # Basic check (using visibility if available, otherwise just check coords)
        if np.all(left_hip == 0) or np.all(right_hip == 0): 
            center_x, center_y = 0.0, 0.0 
        else:
             center_x = (left_hip[0] + right_hip[0]) / 2.0
             center_y = (left_hip[1] + right_hip[1]) / 2.0

        # Normalize coordinates (subtract center)
        current_norm_coords = frame_data[:, :2] - np.array([center_x, center_y])
        normalized_coords[t] = current_norm_coords

        # Calculate velocity (difference from last frame's normalized coords)
        if last_norm_coords is not None:
            # Avoid calculating velocity if current or previous frame was zero padding/undetected
             if not np.all(current_norm_coords == -np.array([center_x, center_y])) and \
                not np.all(last_norm_coords == -np.array([last_center_x, last_center_y])): # Check against original center subtraction
                 velocities[t] = current_norm_coords - last_norm_coords
             # else: velocity remains zero
        # else: velocity remains zero for the first frame

        last_norm_coords = current_norm_coords
        last_center_x, last_center_y = center_x, center_y # Store center used for next frame's velocity check


    # Combine features: [norm_x, norm_y, vel_x, vel_y] -> shape: (seq_len, num_landmarks, 4)
    combined_features = np.concatenate((normalized_coords, velocities), axis=-1)

    # Reshape back to (seq_len, num_landmarks * 4) which is (seq_len, input_size)
    final_processed_sequence = combined_features.reshape(sequence_length, input_size)

    return final_processed_sequence


def extract_and_process_features_from_video(video_path, frame_skip, sequence_length, input_size, num_landmarks, orig_landmark_dim):
    """
    Extracts Mediapipe keypoints, handles sequence length, normalizes,
    calculates velocity, and formats for the V2 model.

    Returns:
        np.ndarray: A numpy array of shape (sequence_length, input_size)
                    containing processed features, or None if processing fails.
    """
    # Initialize Mediapipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                        smooth_landmarks=True, enable_segmentation=False,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        pose.close()
        return None

    frame_count = 0
    raw_keypoints_sequence = [] # Store raw [x, y, visibility] tuples/lists temporarily

    # --- Step 1: Extract Raw Keypoints from Video Frames ---
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break # End of video

        # Process only every Nth frame
        if frame_count % frame_skip == 0:
            try:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                results = pose.process(image_rgb)

                if results.pose_landmarks:
                    frame_keypoints = []
                    for landmark in results.pose_landmarks.landmark:
                        # Store x, y, visibility for each landmark
                        frame_keypoints.append([landmark.x, landmark.y, landmark.visibility])
                    # Append as shape (num_landmarks, orig_landmark_dim)
                    raw_keypoints_sequence.append(np.array(frame_keypoints, dtype=np.float32))
                else:
                    # Pad with zeros if no pose detected in a sampled frame
                    raw_keypoints_sequence.append(np.zeros((num_landmarks, orig_landmark_dim), dtype=np.float32))

            except Exception as e:
                print(f"Error processing frame {frame_count} in {video_path}: {e}")
                # Append zeros if error occurs during processing
                raw_keypoints_sequence.append(np.zeros((num_landmarks, orig_landmark_dim), dtype=np.float32))

        frame_count += 1

        # Optimization: Stop slightly after gathering enough frames for the sequence
        # This avoids processing the whole video if it's long
        if len(raw_keypoints_sequence) > sequence_length + 5: # Add a small buffer
             break


    cap.release()
    pose.close() # Release Mediapipe resources

    # Convert list of arrays to a single numpy array (extracted_frames, num_landmarks, orig_dim)
    if not raw_keypoints_sequence:
         print(f"Warning: No keypoints extracted from {video_path}")
         return None
    sequence_data_raw = np.stack(raw_keypoints_sequence, axis=0)

    # --- Step 2: Normalize, Calculate Velocity, Pad/Truncate ---
    processed_sequence = _normalize_and_calculate_velocity(
        sequence_data_raw,
        sequence_length,
        input_size,
        num_landmarks,
        orig_landmark_dim
    )

    return processed_sequence


# ==============================================================================
# Main Prediction Logic
# ==============================================================================
def predict_single_video(video_path, model_path):
    """
    Loads the v2 model, processes a video using the v2 feature pipeline,
    and returns the prediction.
    """
    # --- Validate inputs ---
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'")
        return None
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'")
        return None

    print(f"Using device: {DEVICE}")
    print(f"Loading model from: {model_path}")

    # --- Load Model ---
    try:
        # 1. Instantiate the model with the *exact same* parameters as v2 training
        model = FallDetectionRNN(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            num_classes=NUM_CLASSES,
            rnn_type=RNN_TYPE,
            dropout_prob=DROPOUT_PROB, # Pass dropout prob
            bidirectional=BIDIRECTIONAL # Pass bidirectional flag
        )

        # 2. Load the saved state dictionary
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))

        # 3. Move model to the appropriate device
        model.to(DEVICE)

        # 4. Set model to evaluation mode
        model.eval()
        print("Model loaded successfully.")

    except Exception as e:
        print(f"Error loading the model: {e}")

        return None

    # --- Preprocess Video (using the new function) ---
    print(f"\nProcessing video with v2 pipeline: {video_path}")
    start_time = time.time()
    processed_feature_data = extract_and_process_features_from_video(
        video_path,
        FRAME_SKIP,
        SEQUENCE_LENGTH,
        INPUT_SIZE,
        NUM_LANDMARKS,
        ORIGINAL_LANDMARK_DIM
    )
    end_time = time.time()

    if processed_feature_data is None:
        print("Failed to extract and process features from the video.")
        return None

    print(f"Feature extraction & processing took {end_time - start_time:.2f} seconds.")
    print(f"Processed feature shape: {processed_feature_data.shape}") # Should be (SEQUENCE_LENGTH, INPUT_SIZE) e.g., (30, 132)

    # --- Prepare Tensor ---
    # Add batch dimension (batch size of 1) and move to device
    sequence_tensor = torch.tensor(processed_feature_data, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # --- Perform Inference ---
    print("Running inference...")
    with torch.no_grad(): # Disable gradient calculations for inference
        outputs = model(sequence_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

        predicted_class_index = predicted_idx.item()
        prediction_confidence = confidence.item()

    # --- Map index to class name ---
    predicted_class_name = index_to_name.get(predicted_class_index, "Unknown")

    print("\n--- Prediction Result ---")
    print(f"  Detected Class: {predicted_class_name}")
    print(f"  Confidence: {prediction_confidence:.4f}")
    print("  Class Probabilities:")
    for i, name in index_to_name.items():
        print(f"    {name}: {probabilities[0, i].item():.4f}")


    return predicted_class_name, prediction_confidence


# ==============================================================================
# Script Execution Entry Point
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fall Detection Prediction from Video (V2 Model)")
    parser.add_argument(
        "--video", type=str, required=True, help="Path to the input video file."
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to the trained V2 PyTorch model file (.pth)."
    )
    args = parser.parse_args()

    predict_single_video(args.video, args.model)

    print("\n--- Script Finished ---")