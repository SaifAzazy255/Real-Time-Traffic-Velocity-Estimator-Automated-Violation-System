# 🚦 Real-Time Traffic Velocity Estimator & Automated Violation System

A high-performance Computer Vision pipeline built with **YOLOv8** and **OpenCV** to monitor traffic flow, estimate vehicle speeds, and document traffic violations automatically.

## 🌟 Key Features
*   **Multi-Object Tracking:** Leverages YOLOv8 `persist=True` tracking to maintain unique vehicle IDs across frames.
*   **Dynamic Velocity Estimation:** Calculates real-time speed by measuring the time elapsed between two virtual boundary lines relative to the video's FPS.
*   **Automated Violation Documentation:**
    *   **Intelligent Triggering:** Detects vehicles exceeding a predefined speed threshold (e.g., 50 km/h).
    *   **Wide-Angle Snapshots:** Automatically captures and saves high-quality images of violating vehicles with context-aware padding.
*   **Clean Data Management:** Automated workspace cleanup that resets the violation logs on every fresh execution.

## 🛠️ Tech Stack
*   **Model:** YOLOv8 (Ultralytics) for real-time object detection and tracking.
*   **Processing:** OpenCV (Python) for frame manipulation and visualization.
*   **Environment:** Python 3.x.

## 📊 Technical Workflow
1.  **Detection:** Filters for `car` classes to optimize processing speed.
2.  **Tracking:** Assigns a unique ID to each vehicle as it hits the **Entry Line**.
3.  **Calculation:**
    *   $Velocity = \frac{Distance (meters)}{Time (seconds)} \times 3.6$
    *   Distance is calibrated based on real-world road markers.
4.  **Logging:** If $Velocity > Threshold$, a snapshot is saved to the `/violations` directory.

## 🚀 Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YourUsername/Traffic-Velocity-Estimator.git](https://github.com/YourUsername/Traffic-Velocity-Estimator.git)
