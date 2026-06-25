import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import logging
import argparse

from src.anonymizer import anonymize_frame
from src.homography import Homography
from src.road_user_tracker import RoadUserTracker
from src.pet_analyzer import PETAnalyzer, DEFAULT_PET_THRESHOLD_S
from src.report_generator import build_safety_report

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

def run_pipeline(
    input_source: str,
    output_path: str,
    anon_mode: str,
    calibration_path: str | None = None,
    report_path: str | None = None,
    pet_threshold_s: float = DEFAULT_PET_THRESHOLD_S,
    enable_safety_analysis: bool = True,
):

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

    # ------------------------------------------------------
    # Safety analysis setup (road-user tracking + PET)
    # ------------------------------------------------------
    # Runs in the SAME frame loop as anonymization (one decode pass) so we
    # don't pay for reading the video twice. Detection runs on the ORIGINAL
    # frame, before any blurring is applied -- anonymization only affects
    # what's written to output_path, never what the safety analysis sees.

    road_tracker = None
    homography = None

    if enable_safety_analysis:
        if not calibration_path or not os.path.isfile(calibration_path):
            logging.warning(
                "Safety analysis requested but no valid --calibration file "
                "was provided. Skipping PET/speed analysis and report "
                "generation -- only anonymization will run. "
                "See src/calibration.example.json for the expected format."
            )
            enable_safety_analysis = False
        else:
            try:
                homography = Homography.from_json(calibration_path)
                road_tracker = RoadUserTracker()
                if not road_tracker.initialized:
                    logging.warning(
                        "Road-user tracker failed to initialize -- "
                        "disabling safety analysis for this run."
                    )
                    enable_safety_analysis = False
            except Exception as e:
                logging.exception(f"Failed to set up safety analysis: {e}")
                enable_safety_analysis = False

    frame_count = 0

    print("=" * 60)
    print(f"STARTING ANONYMIZATION | MODE = {anon_mode}")
    if enable_safety_analysis:
        print("SAFETY ANALYSIS: ENABLED (PET near-miss detection + report)")
    else:
        print("SAFETY ANALYSIS: DISABLED")
    print("=" * 60)

    # ------------------------------------------------------
    # Processing loop
    # ------------------------------------------------------

    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            timestamp_s = frame_count / fps

            # ---- Safety analysis (reads the ORIGINAL, unblurred frame) ----
            if enable_safety_analysis:
                try:
                    road_tracker.update(frame, frame_count, timestamp_s)
                except Exception as e:
                    logging.exception(f"Road-user tracking failed on frame {frame_count}: {e}")

            # ---- Anonymization (writes the BLURRED frame to disk) ----
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

    # ------------------------------------------------------
    # PET analysis + HTML report (after the video loop finishes)
    # ------------------------------------------------------

    if enable_safety_analysis and road_tracker is not None:
        try:
            logging.info("Finalizing road-user tracks...")
            tracks = road_tracker.finalize()
            logging.info(f"Tracked {len(tracks)} road users total.")

            analyzer = PETAnalyzer(
                homography=homography,
                pet_threshold_s=pet_threshold_s,
            )
            conflicts, trajectories = analyzer.analyze(tracks)

            logging.info(
                f"PET analysis complete: {len(conflicts)} near-miss "
                f"conflicts flagged (threshold={pet_threshold_s}s)."
            )

            final_report_path = report_path or _default_report_path(output_path)

            build_safety_report(
                output_path=final_report_path,
                video_name=os.path.basename(input_source),
                video_duration_s=frame_count / fps,
                trajectories=trajectories,
                conflicts=conflicts,
                pet_threshold_s=pet_threshold_s,
                homography_reprojection_error_m=getattr(
                    homography, "_reprojection_error_m", None
                ),
            )

            print(f"📄 Safety report saved to: {final_report_path}")

        except Exception as e:
            logging.exception(f"Safety report generation failed: {e}")


def _default_report_path(output_video_path: str) -> str:
    base, _ = os.path.splitext(output_video_path)
    return f"{base}_safety_report.html"


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="YOLO Video Anonymization + Intersection Safety Pipeline"
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

    parser.add_argument(
        "--calibration",
        type=str,
        default=None,
        help=(
            "Path to a JSON file with 'image_points' and 'world_points' "
            "for ground-plane calibration (required for PET/speed safety "
            "analysis). See src/calibration.example.json for the format. "
            "If omitted, only anonymization runs -- no safety report."
        )
    )

    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help=(
            "Path to write the HTML safety report. Defaults to "
            "'<output>_safety_report.html' next to the output video."
        )
    )

    parser.add_argument(
        "--pet-threshold",
        type=float,
        default=DEFAULT_PET_THRESHOLD_S,
        help=(
            f"PET (seconds) below which a vehicle/pedestrian-cyclist path "
            f"crossing is flagged as a near-miss conflict. Default: "
            f"{DEFAULT_PET_THRESHOLD_S}s."
        )
    )

    parser.add_argument(
        "--no-safety-analysis",
        action="store_true",
        help="Disable PET/speed safety analysis entirely (anonymization only)."
    )

    args = parser.parse_args()

    run_pipeline(
        input_source=args.input,
        output_path=args.output,
        anon_mode=args.mode,
        calibration_path=args.calibration,
        report_path=args.report,
        pet_threshold_s=args.pet_threshold,
        enable_safety_analysis=not args.no_safety_analysis,
    )
