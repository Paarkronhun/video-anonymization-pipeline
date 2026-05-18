# main.py (REPLACING THE ENTIRE FILE)

import cv2
import numpy as np
import logging
# Ensure this import can find our core logic module
from src.anonymizer import anonymize_frame

# --- Setup basic logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_single_image(input_path: str, output_path: str, anon_mode: str):
    """
    Function to process and anonymize a single static image file.
    This replaces the video reading functionality for testing purposes.
    """
    logging.info(f"Attempting to load static image from: {input_path}")
    
    # 1. Load the single image
    frame = cv2.imread(input_path)
    if frame is None:
        logging.error(f"FATAL ERROR: Could not load image from {input_path}. Check path and file integrity.")
        return
    
    logging.info("Image source successfully loaded.")

    # 2. Process and anonymize the image (The core logic is called here)
    anonymized_frame = anonymize_frame(frame, anon_mode) 

    # 3. Save the output image
    cv2.imwrite(output_path, anonymized_frame)
    
    logging.info("Image processing complete.")
    print(f"\n✅ Success! Anonymized image saved to: {output_path}")


if __name__ == "__main__":
    # --- USER CONFIGURATION SECTION (MUST BE EDITED BY USER) ---
    
    # 1. Define the path to your test image input
    INPUT_SOURCE = "data/frame_test.jpg" 
    
    # 2. Define the output file path for the resulting image
    OUTPUT_FILE = "output_anon/test_anon_output.jpg" 
    
    # 3. Define the anonymization mode ('faces_only' or 'full_bbox')
    ANON_MODE = 'faces_only' 
    # ----------------------------------------------------------
    
    # Execute the single image test run
    process_single_image(INPUT_SOURCE, OUTPUT_FILE, ANON_MODE)