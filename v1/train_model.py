# -*- coding: utf-8 -*-
import os
import glob
import pickle
import time
import numpy as np
import random 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report 
import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd 
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import Counter 

print(f"PyTorch Version: {torch.__version__}")

# ==============================================================================
# Cell 1: Configuration - Updated for Weighted Loss & Tuning
# ==============================================================================

# --- User Defined Paths ---
BASE_PROJECT_PATH = '/Users/samnangpheng/Desktop/Fall_detection' # <-- ADJUST THIS PATH IF NEEDED

FEATURES_DIR = os.path.join(BASE_PROJECT_PATH, 'dataset/processed_features')
MODEL_SAVE_DIR = os.path.join(BASE_PROJECT_PATH, 'trained_models_local')
# Give the new model a distinct name
MODEL_NAME = 'fall_detection_rnn_local_v3_weighted_loss.pth' # Indicate weighted loss attempt

CLASS_FOLDERS = ["backward_fall", "forward_fall", "side_fall", "non_fall"]

# --- Feature & Preprocessing Parameters ---
ORIGINAL_SEQUENCE_LENGTH = 30
ORIGINAL_LANDMARK_DIM = 3
INPUT_SIZE = 33 * 4 # 132 (33 landmarks * [norm_x, norm_y, vel_x, vel_y])

# --- Data Augmentation Parameters (for Training) ---
AUGMENT_PROB = 0.55 # Slightly increase augmentation chance
NOISE_LEVEL = 0.006 # Slightly increase noise

# --- Model & Training Hyperparameters ---
NUM_CLASSES = len(CLASS_FOLDERS)
HIDDEN_SIZE = 192      
NUM_LAYERS = 2         
RNN_TYPE = 'LSTM'      
BIDIRECTIONAL = True   

# --- Training Strategy Parameters ---
USE_WEIGHTED_LOSS = True 

DROPOUT_PROB = 0.35     
BATCH_SIZE = 32
LEARNING_RATE = 0.0004 
WEIGHT_DECAY = 1e-5    
NUM_EPOCHS = 150       
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15
PATIENCE_EARLY_STOPPING = 25 
PATIENCE_LR_SCHEDULER = 10   

# --- Device Configuration ---
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

# --- Create output directories ---
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# --- Class Mappings ---
label_map = {name: i for i, name in enumerate(CLASS_FOLDERS)}
index_to_name = {i: name for name, i in label_map.items()}

print("Configuration loaded (v3 - Weighted Loss Attempt):")
print(f"  Input Size: {INPUT_SIZE}")
print(f"  Hidden Size: {HIDDEN_SIZE}, Num Layers: {NUM_LAYERS}, Bidirectional: {BIDIRECTIONAL}")
print(f"  Use Weighted Loss: {USE_WEIGHTED_LOSS}") # <<< Print new setting
print(f"  Dropout: {DROPOUT_PROB}")
print(f"  Learning Rate: {LEARNING_RATE}, Weight Decay: {WEIGHT_DECAY}")
print(f"  Augmentation Prob: {AUGMENT_PROB}, Noise Level: {NOISE_LEVEL}")
print(f"  Epochs: {NUM_EPOCHS}, Early Stopping Patience: {PATIENCE_EARLY_STOPPING}")

# ==============================================================================
# Cell 2: Data Loading & Splitting (Reads existing .pkl files) - No Changes
# ==============================================================================

def load_feature_files(feature_dir, class_folders, label_map):
    all_feature_files = []
    all_labels = []
    print(f"\nLoading feature files from: {feature_dir}")
    if not os.path.isdir(feature_dir): raise FileNotFoundError(f"Feature directory not found: {feature_dir}")
    for class_name in class_folders:
        class_label = label_map.get(class_name)
        if class_label is None: print(f"Warning: Class '{class_name}' not found in label_map. Skipping."); continue
        class_path = os.path.join(feature_dir, class_name)
        if not os.path.isdir(class_path): print(f"Warning: Class directory not found: {class_path}. Skipping."); continue
        pkl_files = glob.glob(os.path.join(class_path, '*.pkl'))
        if not pkl_files: print(f"Warning: No .pkl files found in {class_path}"); continue
        print(f"  Found {len(pkl_files)} features for class: {class_name}")
        all_feature_files.extend(pkl_files)
        all_labels.extend([class_label] * len(pkl_files))
    if not all_feature_files: raise ValueError(f"No .pkl files found in any class subdirectories of {feature_dir}")
    print(f"\nTotal feature files found: {len(all_feature_files)}")
    return all_feature_files, np.array(all_labels)

try:
    all_files, all_labels = load_feature_files(FEATURES_DIR, CLASS_FOLDERS, label_map)
except Exception as e:
    print(f"Error loading feature files: {e}"); exit()

# --- Split Data: Train / Validation / Test ---
train_val_files, test_files, train_val_labels, test_labels = train_test_split(
    all_files, all_labels, test_size=TEST_SPLIT, random_state=42, stratify=all_labels)
val_split_adjusted = VALIDATION_SPLIT / (1.0 - TEST_SPLIT)
train_files, val_files, train_labels, val_labels = train_test_split(
    train_val_files, train_val_labels, test_size=val_split_adjusted, random_state=42, stratify=train_val_labels)

print("\nData Splitting Complete:")
print(f"  Training samples:   {len(train_files)}")
print(f"  Validation samples: {len(val_files)}")
print(f"  Testing samples:    {len(test_files)}")
if not train_files or not val_files or not test_files:
    print("Warning: One or more data splits are empty."); # exit()

# ==============================================================================
# Cell 3: Dataset with Feature Engineering & Augmentation - No Changes
# ==============================================================================

class PoseSequenceDataset(Dataset):
    """
    Dataset class to load .pkl files, perform feature engineering
    (normalization, velocity), and optional augmentation (noise).
    """
    def __init__(self, feature_paths, labels, sequence_length, input_size,
                 orig_landmark_dim=3, is_train=False, augment_prob=0.0, noise_level=0.0):
        self.feature_paths = feature_paths
        self.labels = labels
        self.sequence_length = sequence_length 
        self.input_size = input_size 
        self.orig_landmark_dim = orig_landmark_dim
        self.num_landmarks = 33 

        # Augmentation parameters (only used if is_train is True)
        self.is_train = is_train
        self.augment_prob = augment_prob
        self.noise_level = noise_level

        # Define hip landmark indices
        self.left_hip_idx = 23
        self.right_hip_idx = 24

    def __len__(self):
        return len(self.feature_paths)

    def _normalize_and_calculate_velocity(self, sequence_data_raw):
        """Normalizes pose, calculates velocity, and combines them."""
        try:
            if isinstance(sequence_data_raw, torch.Tensor):
                sequence_data_raw = sequence_data_raw.numpy()
            expected_flat_len = self.sequence_length * self.num_landmarks * self.orig_landmark_dim
            if sequence_data_raw.size != expected_flat_len:
                 print(f"Warning: Raw data size mismatch. Expected {expected_flat_len}, got {sequence_data_raw.size}. Path: {self.feature_paths[idx] if 'idx' in locals() else 'Unknown'}. Returning zeros.")
                 return np.zeros((self.sequence_length, self.input_size), dtype=np.float32)

            sequence_reshaped = sequence_data_raw.reshape(
                self.sequence_length, self.num_landmarks, self.orig_landmark_dim
            )
        except ValueError as e:
             print(f"Error reshaping sequence data. Expected flat length {expected_flat_len}, got shape {sequence_data_raw.shape}. Error: {e}. Path: {self.feature_paths[idx] if 'idx' in locals() else 'Unknown'}")
             return np.zeros((self.sequence_length, self.input_size), dtype=np.float32)
        except AttributeError as e: # Handle cases where it might not be a numpy array initially
             print(f"Attribute error during reshape (is data a numpy array?). Shape: {getattr(sequence_data_raw, 'shape', 'N/A')}. Error: {e}. Path: {self.feature_paths[idx] if 'idx' in locals() else 'Unknown'}")
             return np.zeros((self.sequence_length, self.input_size), dtype=np.float32)


        normalized_coords = np.zeros((self.sequence_length, self.num_landmarks, 2), dtype=np.float32) # Store norm_x, norm_y
        velocities = np.zeros((self.sequence_length, self.num_landmarks, 2), dtype=np.float32) # Store vel_x, vel_y
        last_norm_coords = None

        for t in range(self.sequence_length):
            frame_data = sequence_reshaped[t]
             # Basic check for valid hip indices
            if self.left_hip_idx >= frame_data.shape[0] or self.right_hip_idx >= frame_data.shape[0]:
                 print(f"Warning: Invalid hip indices {self.left_hip_idx}, {self.right_hip_idx} for frame shape {frame_data.shape}. Skipping frame {t} normalization. Path: {self.feature_paths[idx] if 'idx' in locals() else 'Unknown'}")
                 continue 

            left_hip = frame_data[self.left_hip_idx, :2]
            right_hip = frame_data[self.right_hip_idx, :2]
            center_x = (left_hip[0] + right_hip[0]) / 2.0
            center_y = (left_hip[1] + right_hip[1]) / 2.0
            current_norm_coords = frame_data[:, :2] - np.array([center_x, center_y])
            normalized_coords[t] = current_norm_coords
            if last_norm_coords is not None:
                velocities[t] = current_norm_coords - last_norm_coords
            last_norm_coords = current_norm_coords

        combined_features = np.concatenate((normalized_coords, velocities), axis=-1)
        final_sequence = combined_features.reshape(self.sequence_length, self.input_size)
        return final_sequence

    def _add_noise(self, sequence_data):
        """Adds Gaussian noise to the sequence data."""
        noise = np.random.normal(0, self.noise_level, sequence_data.shape)
        return sequence_data + noise.astype(np.float32)

    def __getitem__(self, idx):
        feature_path = self.feature_paths[idx]
        label = self.labels[idx]
        try:
            with open(feature_path, 'rb') as f:
                sequence_data_raw = pickle.load(f)
                if not isinstance(sequence_data_raw, (np.ndarray, torch.Tensor)):
                    print(f"Warning: Loaded data from {feature_path} is not a NumPy array or Tensor (type: {type(sequence_data_raw)}). Attempting conversion.")
                    try:
                        sequence_data_raw = np.array(sequence_data_raw, dtype=np.float32)
                    except Exception as conv_e:
                         print(f"Error converting data from {feature_path} to NumPy array: {conv_e}. Returning zeros.")
                         return torch.zeros((self.sequence_length, self.input_size), dtype=torch.float32), torch.tensor(0, dtype=torch.long)

            processed_sequence = self._normalize_and_calculate_velocity(sequence_data_raw)
            if self.is_train and random.random() < self.augment_prob:
                processed_sequence = self._add_noise(processed_sequence)
            sequence_tensor = torch.tensor(processed_sequence, dtype=torch.float32)
            label_tensor = torch.tensor(label, dtype=torch.long)
            return sequence_tensor, label_tensor
        except FileNotFoundError:
            print(f"Error: File not found {feature_path}.")
            # Return zero tensors matching expected output shape and type
            return torch.zeros((self.sequence_length, self.input_size), dtype=torch.float32), torch.tensor(0, dtype=torch.long)
        except pickle.UnpicklingError as e:
             print(f"Error unpickling file {feature_path}: {e}. File might be corrupted or empty.")
             return torch.zeros((self.sequence_length, self.input_size), dtype=torch.float32), torch.tensor(0, dtype=torch.long)
        except Exception as e:
            print(f"Error loading/processing file {feature_path}: {e}")
            # Return zero tensors matching expected output shape and type
            return torch.zeros((self.sequence_length, self.input_size), dtype=torch.float32), torch.tensor(0, dtype=torch.long)


# --- Create Datasets ---
train_dataset = PoseSequenceDataset(train_files, train_labels, ORIGINAL_SEQUENCE_LENGTH, INPUT_SIZE,
                                    orig_landmark_dim=ORIGINAL_LANDMARK_DIM, is_train=True,
                                    augment_prob=AUGMENT_PROB, noise_level=NOISE_LEVEL)
val_dataset = PoseSequenceDataset(val_files, val_labels, ORIGINAL_SEQUENCE_LENGTH, INPUT_SIZE,
                                  orig_landmark_dim=ORIGINAL_LANDMARK_DIM, is_train=False)
test_dataset = PoseSequenceDataset(test_files, test_labels, ORIGINAL_SEQUENCE_LENGTH, INPUT_SIZE,
                                   orig_landmark_dim=ORIGINAL_LANDMARK_DIM, is_train=False)

# --- Create DataLoaders ---
# Set num_workers=0 on macOS for MPS compatibility if issues arise
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True if DEVICE != torch.device('cpu') else False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True if DEVICE != torch.device('cpu') else False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True if DEVICE != torch.device('cpu') else False)


print("\nDatasets and DataLoaders created with feature engineering & augmentation.")
# --- Quick check of one batch ---
try:
    seq_batch, label_batch = next(iter(train_loader))
    print(f"Sample batch - Sequence shape: {seq_batch.shape}, Label shape: {label_batch.shape}")
    print(f"Sample sequence (first item in batch) - First few features:\n{seq_batch[0, 0, :10]}...")
    print(f"Sample labels (first 5 in batch): {label_batch[:5]}")
except StopIteration:
    print("Warning: Train loader is empty.")
except Exception as e:
    print(f"Error fetching batch from train_loader: {e}")


# ==============================================================================
# Cell 4: Model Building (Bidirectional RNN) - No Changes
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
        self.fc = nn.Linear(hidden_size * self.num_directions, num_classes)

    def forward(self, x):
        # Ensure input tensor is on the correct device
        x = x.to(next(self.parameters()).device) # Ensure input is on same device as model

        # Initialize hidden state
        h0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(x.device)
        if self.rnn_type == 'LSTM':
            c0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(x.device)
            hidden = (h0, c0)
        else: # GRU
            hidden = h0

        # Forward propagate RNN
        # 'out' contains outputs for every time step
        # '_' contains the final hidden state (h_n, c_n for LSTM)
        out, _ = self.rnn(x, hidden)

        # We typically use the output of the *last* time step for classification
        # out shape: (batch_size, seq_length, hidden_size * num_directions)
        # Select the output of the last time step (-1)
        last_step_out = out[:, -1, :]

        # Apply dropout and pass through the fully connected layer
        last_step_out = self.dropout(last_step_out)
        out = self.fc(last_step_out)
        return out

# --- Instantiate the model ---
# Note: DROPOUT_PROB from Cell 1 is passed here
model = FallDetectionRNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES,
                         rnn_type=RNN_TYPE, dropout_prob=DROPOUT_PROB,
                         bidirectional=BIDIRECTIONAL)
model.to(DEVICE)

print("\nModel Architecture (v3 - Weighted Loss Attempt):")
print(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params}")

# ==============================================================================
# Cell 5: Training Setup - Loss Function (Potentially Weighted), Optimizer, Scheduler
# ==============================================================================

# --- Calculate Class Weights (if enabled) ---
class_weights_tensor = None
if USE_WEIGHTED_LOSS:
    print("\nCalculating class weights for weighted loss...")
    # Count occurrences of each class *only in the training set*
    label_counts = Counter(train_labels)
    if len(label_counts) != NUM_CLASSES:
        print(f"Warning: Training set does not contain all {NUM_CLASSES} classes! Found {len(label_counts)}.")
        # Ensure all classes potentially have a count, even if 0
        for i in range(NUM_CLASSES):
            label_counts[i] = label_counts.get(i, 0) # Default to 0 if not present

    # Calculate weights - inverse frequency based formula
    total_samples = len(train_labels)
    weights = []
    # Ensure weights are calculated in the correct order (0, 1, 2, ...)
    for i in range(NUM_CLASSES):
        count = label_counts.get(i, 0) # Get count, default to 0 if somehow missed
        if count == 0:
            # Handle missing class: Assign a high weight or a default weight?
            # Using total_samples / NUM_CLASSES as a proxy for average count if one class had 1 sample.
            # This gives it significant weight but avoids division by zero.
            print(f"Warning: Class {index_to_name.get(i, 'Unknown')} (index {i}) has 0 samples in training set. Assigning calculated high weight.")
            weight = total_samples / NUM_CLASSES if NUM_CLASSES > 0 else 1.0 # Avoid division by zero if NUM_CLASSES is 0
        else:
             weight = total_samples / (NUM_CLASSES * count)
        weights.append(weight)

    class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    print(f"  Class weights calculated: {class_weights_tensor.cpu().numpy()}")
    # Map weights back to class names for clarity
    class_weight_dict = {index_to_name.get(i, f'Unknown_{i}'): w for i, w in enumerate(weights)} # Added default name
    print(f"  Weights per class: {class_weight_dict}")


# --- Define Loss Function ---
# Use weights if calculated, otherwise use standard CE Loss
if class_weights_tensor is not None and USE_WEIGHTED_LOSS: # Check flag again just in case
     criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
     print("  Using Weighted CrossEntropyLoss.")
else:
     criterion = nn.CrossEntropyLoss()
     print("  Using standard CrossEntropyLoss.")

# --- Define Optimizer and Scheduler ---
# Note: LEARNING_RATE and WEIGHT_DECAY from Cell 1 are used here
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
# Note: PATIENCE_LR_SCHEDULER from Cell 1 is used here
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=PATIENCE_LR_SCHEDULER, verbose=True)

print("\nLoss function and optimizer defined.")
print(f"  Criterion: {type(criterion).__name__}")
print(f"  Optimizer: AdamW (LR={LEARNING_RATE}, Weight Decay={WEIGHT_DECAY})")
print(f"  Scheduler: ReduceLROnPlateau (Patience={PATIENCE_LR_SCHEDULER})")

# ==============================================================================
# Cell 6: Training & Validation Loop - Includes Early Stopping & Accuracy Tracking
# ==============================================================================

def evaluate_model(model, dataloader, criterion, device):
    """Evaluates the model on a given dataloader."""
    model.eval()
    running_loss = 0.0; correct_preds = 0; total_samples = 0
    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences, labels = sequences.to(device), labels.to(device)
            try:
                outputs = model(sequences)
                loss = criterion(outputs, labels) # Criterion will use weights if defined
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_preds += (predicted == labels).sum().item()
            except Exception as eval_e:
                print(f"Error during evaluation batch: {eval_e}")
                print(f"  Sequence shape: {sequences.shape}, Labels: {labels}")
                continue # Skip batch
    if not dataloader: # Handle empty dataloader
        return 0.0, 0.0
    epoch_loss = running_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    epoch_acc = (100.0 * correct_preds / total_samples) if total_samples > 0 else 0.0
    return epoch_loss, epoch_acc

print("\nStarting Training (v3 - Weighted Loss Attempt)...")
best_val_accuracy = 0.0
epochs_no_improve = 0

train_accuracies = []
val_accuracies = []
train_losses = [] 
val_losses = []  

# Use the new model name defined in Cell 1
model_save_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)

# --- Training Loop ---
epochs_completed = 0 # Track actual epochs run
for epoch in range(NUM_EPOCHS):
    epoch_start_time = time.time()
    model.train()
    running_loss_train = 0.0; correct_train = 0; total_train = 0

    for i, (sequences, labels) in enumerate(train_loader):
        sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)

        # --- Defensive Check ---
        if sequences.nelement() == 0: # Check if tensor is empty
             print(f"Warning: Empty sequence tensor encountered in training batch {i}. Skipping.")
             continue
        if labels.nelement() == 0:
             print(f"Warning: Empty label tensor encountered in training batch {i}. Skipping.")
             continue

        try:
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels) # Criterion will use weights if defined

            # --- Check for NaN/Inf loss ---
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN or Inf loss detected at epoch {epoch+1}, batch {i}. Skipping update.")
                continue 
            loss.backward()

            optimizer.step()

            running_loss_train += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        except Exception as train_e:
            print(f"Error during training batch {i} in epoch {epoch+1}: {train_e}")
            print(f"  Sequence shape: {sequences.shape}, Labels: {labels}")
            continue # Continue to next batch

    # --- Calculate Epoch Metrics ---
    # Avoid division by zero if train_loader was empty or all batches failed
    train_loss = running_loss_train / len(train_loader) if len(train_loader) > 0 else 0.0
    train_accuracy = (100.0 * correct_train / total_train) if total_train > 0 else 0.0

    # --- Validation Phase ---
    val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, DEVICE)

    epoch_duration = time.time() - epoch_start_time
    epochs_completed += 1 # Increment actual epochs completed

    # <<< Store metrics for plotting >>>
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] ({epoch_duration:.2f}s) | "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}% | "
          f"LR: {optimizer.param_groups[0]['lr']:.6f}")

    # --- LR Scheduling & Early Stopping ---
    scheduler.step(val_accuracy) # Step scheduler based on validation accuracy
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        try:
            torch.save(model.state_dict(), model_save_path)
            print(f"  -> Val accuracy improved to {best_val_accuracy:.2f}%. Saving model to {model_save_path}")
        except Exception as save_e:
             print(f"  -> Val accuracy improved to {best_val_accuracy:.2f}%, BUT FAILED TO SAVE MODEL: {save_e}")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"  -> Val accuracy did not improve for {epochs_no_improve} epoch(s). Best: {best_val_accuracy:.2f}%")

    # Note: PATIENCE_EARLY_STOPPING from Cell 1 is used here
    if epochs_no_improve >= PATIENCE_EARLY_STOPPING:
        print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
        break

print(f"\nTraining Finished after {epochs_completed} epochs.")
if os.path.exists(model_save_path):
     print(f"Best model (val acc: {best_val_accuracy:.2f}%) saved: {model_save_path}")
else:
    print(f"Warning: No model saved at {model_save_path}. Training may have failed or best model wasn't saved.")


# ==============================================================================
# Cell 7: Final Testing & Evaluation (with Confusion Matrix) - No Changes Required Here
# ==============================================================================

print("\nStarting Final Testing on the Unseen Test Set...")

# Use the new model name defined in Cell 1
if os.path.exists(model_save_path):
    print(f"Loading best model from: {model_save_path}")
    # Instantiate a fresh model instance with the *same* parameters as the saved one
    final_model = FallDetectionRNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES,
                                   rnn_type=RNN_TYPE, dropout_prob=DROPOUT_PROB, # Use dropout from config
                                   bidirectional=BIDIRECTIONAL)
    try:
        map_location = torch.device('cpu') if DEVICE == torch.device('cpu') else DEVICE
        final_model.load_state_dict(torch.load(model_save_path, map_location=map_location))
        final_model.to(DEVICE)
        final_model.eval() # Set to evaluation mode

        # --- Evaluate on Test Set ---
        # Use the standard (non-weighted) criterion for final evaluation loss reporting if desired,
        # but accuracy is the primary metric here. evaluate_model uses the training criterion by default.
        # If you want separate test loss without weights:
        # test_criterion_unweighted = nn.CrossEntropyLoss()
        # test_loss, test_accuracy = evaluate_model(final_model, test_loader, test_criterion_unweighted, DEVICE)
        test_loss, test_accuracy = evaluate_model(final_model, test_loader, criterion, DEVICE) # Use same criterion as training


        print("\n--- Test Set Evaluation Results ---")
        print(f"  Test Loss: {test_loss:.4f} (using {'weighted' if USE_WEIGHTED_LOSS and class_weights_tensor is not None else 'standard'} criterion)")
        print(f"  Test Accuracy: {test_accuracy:.2f}%")
        print("---------------------------------")

        # --- Generate Classification Report and Confusion Matrix ---
        all_preds = []
        all_true = []
        with torch.no_grad():
            for sequences, labels in test_loader:
                # <<< Add check for empty batch in test loader >>>
                if sequences.nelement() == 0 or labels.nelement() == 0:
                    print("Warning: Skipping empty batch in test loader during final evaluation.")
                    continue
                sequences = sequences.to(DEVICE)
                # Ensure model processes correctly
                try:
                    outputs = final_model(sequences)
                    _, predicted = torch.max(outputs.data, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_true.extend(labels.cpu().numpy())
                except Exception as test_eval_e:
                    print(f"Error evaluating batch in test set: {test_eval_e}")
                    print(f"  Sequence shape: {sequences.shape}")
                    continue # Skip this batch

        # Check if any predictions were made
        if not all_preds or not all_true:
             print("Error: No predictions generated for the test set. Cannot create report or matrix.")
        else:
            print("\n--- Classification Report (Test Set) ---")
            # Ensure target_names uses the CLASS_FOLDERS from Cell 1
            print(classification_report(all_true, all_preds, target_names=CLASS_FOLDERS, digits=3, zero_division=0)) # Added zero_division=0

            print("\n--- Confusion Matrix (Test Set) ---")
            cm = confusion_matrix(all_true, all_preds, labels=range(NUM_CLASSES)) # Ensure labels are correctly ordered
            # Ensure index/columns use the index_to_name mapping from Cell 1
            # Handle potential missing classes in predictions/true labels for CM indexing
            cm_index_names = [index_to_name.get(i, f"Class_{i}") for i in range(NUM_CLASSES)]
            cm_df = pd.DataFrame(cm, index=cm_index_names, columns=cm_index_names)


            plt.figure(figsize=(9, 7))
            sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix (Test Set - v3 Weighted Loss Attempt)') # Updated title
            plt.ylabel('Actual Class')
            plt.xlabel('Predicted Class')
            # Save the plot using the new model name base
            cm_save_path = os.path.join(MODEL_SAVE_DIR, f'{os.path.splitext(MODEL_NAME)[0]}_confusion_matrix.png')
            try:
                plt.savefig(cm_save_path)
                print(f"Confusion matrix saved to: {cm_save_path}")
            except Exception as plot_e:
                print(f"Error saving confusion matrix plot: {plot_e}")
            plt.show()


    except FileNotFoundError:
         print(f"Error: Model file not found at {model_save_path}")
    except Exception as e:
        print(f"Error loading model or evaluating on test set: {e}")
else:
    print(f"Could not find the saved model file: {model_save_path}. Skipping final testing.")

# ==============================================================================
# Cell 8: Plot Training & Validation Accuracy Graph
# ==============================================================================

print("\nPlotting Training & Validation Accuracy...")

# Check if training occurred and metrics were recorded
if train_accuracies and val_accuracies:
    epochs_range = range(1, epochs_completed + 1) # Use actual completed epochs

    plt.figure(figsize=(12, 6))

    # Plot Accuracy
    plt.subplot(1, 2, 1) # Create subplot for accuracy
    plt.plot(epochs_range, train_accuracies, label='Training Accuracy', marker='o', linestyle='-')
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy', marker='x', linestyle='--')
    plt.title('Training & Validation Accuracy (v3)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 105) # Set Y-axis limits for accuracy (0-100%)

    # Plot Loss (Optional but good practice)
    if train_losses and val_losses:
        plt.subplot(1, 2, 2) # Create subplot for loss
        plt.plot(epochs_range, train_losses, label='Training Loss', marker='o', linestyle='-')
        plt.plot(epochs_range, val_losses, label='Validation Loss', marker='x', linestyle='--')
        plt.title('Training & Validation Loss (v3)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        # Adjust Y-axis for loss if needed, e.g., plt.ylim(bottom=0)

    plt.tight_layout() # Adjust subplot params for a tight layout

    # Save the combined plot
    acc_loss_plot_save_path = os.path.join(MODEL_SAVE_DIR, f'{os.path.splitext(MODEL_NAME)[0]}_accuracy_loss_plot.png')
    try:
        plt.savefig(acc_loss_plot_save_path)
        print(f"Accuracy and Loss plot saved to: {acc_loss_plot_save_path}")
    except Exception as plot_e:
        print(f"Error saving accuracy/loss plot: {plot_e}")

    plt.show()
else:
    print("No accuracy data recorded (training might have failed early or lists are empty). Skipping plot generation.")


print("\n--- End of Script (v3 - Weighted Loss Attempt) ---")