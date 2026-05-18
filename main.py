# main.py (REPLACING THE ENTIRE FILE)

import cv2
import numpy as np
import logging
import argparse # NEW: Library to handle command-line arguments
from src.anonymizer import anonymize_frame

# --- Setup basic logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_video(input_path: str) -> cv2.VideoCapture | None:
    """
    Attempts to open a video source (file or stream).
    Returns the video capture object or None if opening fails.
    """
    logging.info(f"Attempting to load video from: {input_path}")
    
    cap = cv2.VideoCapture(input_path)
    
    # Check the failure conditions
    if not cap.isOpened():
        logging.error(f"FATAL ERROR: Could not open video source at {input_path}.")
        logging.error("Check file paths, URL format, and required dependencies.")
        return None
    
    logging.info("Video source successfully opened.")
    return cap

def run_pipeline(input_source: str, output_path: str, anon_mode: str):
    """
    The main orchestration function. Handles I/O and guarantees resource cleanup.
    (Logic remains the same, it's the entry point that changes)
    """
    # 1. Setup and Load the video
    cap = load_video(input_source)
    if cap is None:
        return
        
    # 2. Setup output video writer
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) 
    
    # Use 'mp4v' codec for maximum compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    print("="*50)
    print(f"STARTING ANONYMIZATION: Mode={anon_mode}")
    print("="*50)

    # Use a try...finally block to guarantee resource cleanup
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break # The loop breaks upon reaching the end of the video stream

            # Core call to our advanced logic
            anonymized_frame = anonymize_frame(frame, anon_mode) 

            # Write the processed frame
            out.write(anonymized_frame)
            frame_count += 1

    except Exception as e:
        # If any unknown error happens during the loop, we catch it and log it.
        logging.critical(f"\n[PIPELINE FAILURE] Process interrupted due to an error: {e}")
    
    finally:
        # The 'finally' block runs NO MATTER WHAT—success, break, or error.
        logging.info("---- STARTING CLEANUP SEQUENCE ----")
        cap.release() # Release the video reader
        out.release() # Release the output writer
        logging.info("Video resources fully released. Pipeline finished.")
        print(f"\n✅ Success! {frame_count} frames processed and saved to {output_path}")


if __name__ == "__main__":
    # Initialization of the argument parser
    parser = argparse.ArgumentParser(
        description="A robust pipeline for video anonymization using deep learning object detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Define required arguments
    parser.add_argument(
        '--input', 
        type=str, 
        required=True, 
        help="Path to the video file or a live streaming URL (e.g., 'data/input.mp4')."
    )
    parser.add_argument(
        '--output', 
        type=str, 
        required=True, 
        help="Output file path for the anonymized video (e.g., 'output/anon.mp4')."
    )
    parser.add_argument(
        '--mode', 
        type=str, 
        default='faces_only', 
        choices=['faces_only', 'full_bbox'],
        help="The mode of anonymization: 'faces_only' (default) or 'full_bbox'."
    )
    
    # Parse the arguments provided by the user in the terminal
    args = parser.parse_args()
    
    # Run the pipeline using the values passed from the terminal
    run_pipeline(args.input, args.output, args.mode)