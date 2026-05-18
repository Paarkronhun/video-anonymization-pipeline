# main.py

import cv2
import numpy as np
import logging
# We use our newly developed core logic module
from src.anonymizer import anonymize_frame

# --- Setup basic logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_video(input_path: str) -> cv2.VideoCapture | None:
    """Loads video from a file or an HTTPS stream URL."""
    logging.info(f"Attempting to load video from: {input_path}")
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logging.error(f"Error: Could not open video source at {input_path}")
        return None
    
    logging.info("Video source successfully opened.")
    return cap

def run_pipeline(input_source: str, output_path: str, anon_mode: str):
    """
    Main function orchestrating the anonymization process.
    This function controls the video reading, processing, and writing.
    """
    # 1. Load the source video
    cap = load_video(input_source)
    if cap is None:
        return
        
    # 2. Setup output video writer
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    print("="*50)
    print(f"STARTING ANONYMIZATION: Mode={anon_mode}")
    print("="*50)

    while True:
        ret, frame = cap.read()
        if not ret:
            break # End of video stream

        # *** CORE LOGIC HOOK ***
        # This is the function that calls the YOLO detector and masks the frame.
        anonymized_frame = anonymize_frame(frame, anon_mode) 
        # ************************

        # Write the processed frame to the output
        out.write(anonymized_frame)
        frame_count += 1

    # 3. Cleanup
    cap.release()
    out.release()
    logging.info(f"Processing complete. {frame_count} frames processed.")
    
    # 4. Secure Deletion Placeholder
    logging.info("Data cleanup sequence triggered.")

if __name__ == "__main__":
    # --- USER CONFIGURATION SECTION ---
    # 1. Define the path to the video you want to anonymize
    INPUT_SOURCE = "data/test.mp4" 
    
    # 2. Define the output file path
    OUTPUT_FILE = "output_anon/anonymized_video.mp4" 
    
    # 3. Define the anonymization mode ('faces_only' or 'full_bbox')
    ANON_MODE = 'faces_only' 
    # ----------------------------------
    
    # Run the full pipeline
    run_pipeline(INPUT_SOURCE, OUTPUT_FILE, ANON_MODE)