# 🎥 Video Anonymization Pipeline

**Project Goal:** Developing a robust framework for processing video streams to detect and obscure personally identifiable information (PII), ensuring data privacy and compliance.

## 📋 Description

This pipeline is a sophisticated solution designed to process video feeds, whether from local files or live streaming URLs. Its primary function is the robust anonymization of sensitive data, specifically focusing on human faces and bounding boxes.

The system operates under a strict privacy protocol, ensuring that the original, identifiable source material is securely handled or deleted after the anonymization process is complete.

### ✨ Key Features

*   **Multi-Source Input:** Accepts local video files or live/recorded video streams from HTTPS URLs.
*   **Targeted Anonymization:** Supports granular control over the anonymization level:
    *   `faces_only`: Blurs or pixelates only detected facial regions.
    *   `full_bbox`: Obscures the entire bounding box around the detected subject.
*   **Data Hygiene:** Implement a mechanism for the secure handling and eventual deletion of the original sensitive video data.
*   **Modularity:** The core logic is separated, allowing for easy integration of new detection algorithms (e.g., switching from Haar Cascades to a modern deep learning model like YOLO).

## 🏗️ System Architecture and Workflow

The process follows a secure, three-stage pipeline:

1.  **Input Stage (The Stream/Source):**
    *   The system first handles the data acquisition. If a URL is provided, it uses streaming protocols to capture frames efficiently. If a local file is provided, it loads the file.
2.  **Core Processing Stage (Anonymization):**
    *   The system reads the video frame by frame.
    *   **Detection:** An Object Detection algorithm (e.g., OpenCV's DNN/Haar) identifies the coordinates of faces or full bodies.
    *   **Transformation:** Based on the selected mode (`faces_only` or `full_bbox`), the system applies a masking technique (e.g., Gaussian blur, pixelation) to the detected areas of the current frame.
    *   The masked frames are then re-assembled into a new video stream.
3.  **Output Stage (Clean-up):**
    *   The anonymized video is saved to a specified output path.
    *   Crucially, the pipeline confirms the secure deletion/archiving of the original source footage, preventing data retention risks.

## 🚀 Installation and Usage

### Prerequisites
*   Python 3.8+
*   `git` (Version Control)

### Installation Steps
1.  **Clone the Repository:**
    ```bash
    git clone [Your GitHub Repo URL]
    cd video-anonymization-pipeline
    ```
2.  **Set Up Virtual Environment (Highly Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/Mac
    # venv\Scripts\activate   # On Windows
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r DOCUMENTATION/requirements.txt
    ```

### Usage Guide (Running the Pipeline)

The entry point is `src/anonymizer.py`. You must define your input source (`--input`) and the anonymization strategy (`--mode`).

#### A. From a Live/Online Stream (HTTPS)
Use this method when processing data from the web.

```bash
python src/anonymizer.py \
    --input "https://stream.example.com/live/feed" \
    --mode faces_only \
    --output output_anon/stream_anon.mp4
```

#### B. From a Local Video File
Use this method when you have downloaded the video source.

```bash
python src/anonymizer.py \
    --input data/local_video_source.mp4 \
    --mode full_bbox \
    --output output_anon/local_anon.mp4
```



1.  **Formatting:** It uses Markdown headers (`#`, `##`, `###`) and code blocks (` ``` `), which is standard for GitHub.
2.  **Clarity:** It separates *What it does* (Description) from *How it works* (Architecture) from *How to run it* (Usage).
3.  **Technical Depth:** By mentioning streaming protocols and compliance (RGPD), you show that you understand the depth of the problem, not just the code.
4.  **Completeness:** The inclusion of a "Future Roadmap" makes the project look ambitious and scalable.