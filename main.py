import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import logging
import argparse

from src.anonymizer import anonymize_frame

# ==========================================================
# LOGGING
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ==========================================================
# VIDEO LOADER
# ==========================================================

def load_video(input_path: str):

    logging.info(f"Opening video source: {input_path}")

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        logging.error(f"Cannot open video source: {input_path}")
        return None

    return cap

# ==========================================================
# MAIN PIPELINE
# ==========================================================

def run_pipeline(input_source: str, output_path: str, anon_mode: str):
    
    cap = load_video(input_source)

    if cap is None:
        return

    # ------------------------------------------------------
    # Video properties
    # ------------------------------------------------------

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30
        
    
    logging.info(f"Resolution: {frame_width}x{frame_height}")
    logging.info(f"FPS: {fps}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logging.info(f"Total frames: {total_frames}")
    # ------------------------------------------------------
    # Video writer
    # ------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (frame_width, frame_height)
    )

    if not out.isOpened():
        logging.error("Failed to initialize video writer.")
        cap.release()
        return

    frame_count = 0

    print("=" * 60)
    print(f"STARTING ANONYMIZATION | MODE = {anon_mode}")
    print("=" * 60)

    # ------------------------------------------------------
    # Processing loop
    # ------------------------------------------------------

    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            anonymized_frame = anonymize_frame(
                frame=frame,
                mode=anon_mode
            )

            out.write(anonymized_frame)

            frame_count += 1

            if frame_count % 30 == 0:
                logging.info(f"Processed {frame_count}/{total_frames} frames | Progress {round(frame_count/total_frames*1000)/10} %")

    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")

    except Exception as e:
        logging.exception(f"Pipeline crashed: {e}")

    finally:

        logging.info("Releasing resources...")

        cap.release()
        out.release()

        cv2.destroyAllWindows()

        logging.info("Pipeline finished.")

        print("\n✅ DONE")
        print(f"Frames processed: {frame_count}")
        print(f"Saved to: {output_path}")

# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="YOLO Video Anonymization Pipeline"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input video path"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output video path"
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["face", "body"],
        help="""
        face -> censor only face
        body -> censor entire body
        """
    )
    args = parser.parse_args()

    run_pipeline(
        input_source=args.input,
        output_path=args.output,
        anon_mode=args.mode
    )