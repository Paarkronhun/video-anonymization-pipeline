# 🎥 Video Anonymization Pipeline: Advanced Tracking and Privacy Enforcement

## 📋 Project Goal
To develop a robust, state-of-the-art framework for processing video streams to detect, track, and securely obscure Personally Identifiable Information (PII), ensuring maximum data privacy and compliance with global regulations (e.g., GDPR).

## ✨ Key Features & Enhancements
This pipeline is a sophisticated solution designed to process high-resolution video feeds, whether from local files or simulated live streams.

*   **Deep Object Detection & Tracking:** Utilizes **YOLOv8/YOLOv11** for initial detection combined with **Kalman Filtering** and **Deep Sort** tracking algorithms. This ensures persistent identification of subjects across multiple frames, even during brief occlusions or camera movements.
*   **Targeted Anonymization Modes:**
    *   `face`: Blurs or pixelates a defined region around the detected human face.
    *   `body`: Obscures the entire bounding box of the detected subject (full body).
*   **Multi-Source Input:** Accepts local video files or can be architected for live/recorded video streams from HTTPS URLs.
*   **Data Hygiene and Compliance:** Implements a structured workflow that confirms the secure handling and eventual deletion/archiving of the original sensitive video data, adhering to "Privacy by Design" principles.
*   **Modular Design:** The core logic separates detection, tracking, and masking, allowing easy integration of advanced ML models (e.g., switching from YOLO to a specialized facial recognition model).

## 🏗️ System Architecture and Workflow

The pipeline operates through a secure, three-stage system, leveraging object tracking for superior accuracy.

### 1. Input Stage (Source Acquisition)
The system handles data acquisition.
*   **File Input:** Loads local video files (`.mp4`, etc.).
*   **Stream Input (Planned):** Optimized to handle streaming protocols, reading frames efficiently for real-time processing.

### 2. Core Processing Stage (Tracking & Anonymization)
This is the heart of the system, executed frame-by-frame.

*   **Detection (YOLO):** An optimized model identifies potential subjects (persons) and generates initial bounding boxes.
*   **Tracking (Kalman Filter):** The tracking mechanism maintains a persistent ID for each detected object. When a subject is briefly lost or the detection confidence drops, the system uses the Kalman Filter to predict the object's next location and reacquire the bounding box, ensuring smoother, more consistent anonymization.
*   **Transformation (Masking):** Based on the user-defined mode (`face` or `body`), the system applies masking techniques (e.g., solid black overlay, pixelation) to the tracked bounding box on the current frame.
*   **Reassembly:** The newly masked frames are efficiently written to a new output video stream.

### 3. Output Stage (Clean-up and Output)
*   The fully anonymized video is saved to the specified output path.
*   **Compliance Protocol:** The pipeline logs the completion and confirmation of the original source material's secure deletion/archiving, ensuring a complete audit trail.

## 🚀 Installation and Usage

### Prerequisites
*   Python 3.8+
*   `git` (Version Control)
*   A CUDA-enabled GPU (Highly recommended for real-time performance)

### Installation Steps
1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Paarkronhun/video-anonymization-pipeline
    cd video-anonymization-pipeline
    ```
2.  **Set Up Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Linux/Mac
    # venv\Scripts\activate    # On Windows
    ```
3.  **Install Dependencies:**
    *(Note: These dependencies include `torch`, `ultralytics`, and `opencv-python`)*
    ```bash
    pip install -r requirements.txt
    ```

### Usage Guide (Running the Pipeline)

The primary entry point is `main.py`. You must define your input source, output path, and the desired anonymization strategy.

**A. From a Local Video File**
Use this method when you have downloaded the video source.

```bash
python main.py \
    --input data/local_video_source.mp4 \
    --output data/local_anon.mp4 \
    --mode body
```

**B. From a Live/Online Stream (Future Implementation)**
When adapted for streaming, the input path will point to the stream URL, and the processing loop will run continuously until interrupted.

```bash
# Example syntax for a live feed (implementation pending advanced stream reader)
python main.py \
    --input "rtsp://ip_address:port" \
    --output data/live_anon.mp4 \
    --mode face
```

***

### 🎯 Technical Notes & Design Decisions

*   **Object Tracking Robustness:** By using a combination of YOLO detection and Kalman filtering, the system moves beyond simple frame-by-frame detection. It maintains a *track* of the individual, which is critical for masking consistency when subjects move quickly or briefly exit the frame.
*   **Scalability:** The detection layer is encapsulated within the `YoloDetector` class, making it trivial to swap out the backbone model (e.g., replacing YOLO with a specialized facial recognition API without changing the core anonymization logic).
*   **Resource Management:** The `main.py` structure includes explicit `try...finally` blocks to ensure that both the video capture object (`cap`) and the video writer object (`out`) are always correctly released, preventing resource leaks.
*   **Error Handling:** The inclusion of detailed logging and exception handling ensures that the pipeline gracefully fails and reports the exact point of failure.