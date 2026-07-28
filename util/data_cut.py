import os
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def process_z_axis_only(input_csv, output_dir, segment_length=512, step_size=256):

    if not os.path.exists(input_csv):
        logging.error(f"File not found: {input_csv}")
        return

    os.makedirs(output_dir, exist_ok=True)

    try:
        # 1. Targeted Loading (Only AccZ_ms2)
        target_col = 'GyroZ_degs'
        logging.info(f"Reading {target_col} from {input_csv}...")
        
        # We only load the one column we need to save memory
        df = pd.read_csv(input_csv, usecols=[target_col])
        data = df[target_col].dropna().values
        
        if len(data) < segment_length:
            logging.warning("Insufficient data length for segmentation.")
            return

        # 2. Z-Score Normalization
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        if std_val < 1e-6: # Check if the sensor was flatlining
            logging.error("Signal variance is too low. Check if sensor was working.")
            return
            
        data_norm = (data - mean_val) / std_val

        # 3. Sliding Window Segmentation
        # This creates a list of start indices for our windows
        indices = range(0, len(data_norm) - segment_length + 1, step_size)
        
        

        for i, start_idx in enumerate(indices):
            segment = data_norm[start_idx : start_idx + segment_length]
            
            # Save as 1D array of shape (512,)
            output_file = os.path.join(output_dir, f"z_sample_{i:05d}.npy")
            np.save(output_file, segment.astype(np.float32))

        logging.info(f"Saved {len(indices)} Z-axis samples to {output_dir}")

    except Exception as e:
        logging.error(f"Error processing Z-axis: {e}")

if __name__ == "__main__":
    # Configuration
    process_z_axis_only(
        input_csv=r"C:\Users\Papa\Desktop\MoE-Gyro\New_Day\dataset\0912-Sobh-4hr_cleaned.csv",
        output_dir= r"C:\Users\Papa\Desktop\MoE-Gyro\New_Day\training_data\0912-Sobh-4hr",
        segment_length= 256,
        step_size= 256
    )