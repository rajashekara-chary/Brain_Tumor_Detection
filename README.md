# 🧠 Brain Tumor Detection Using YOLOv8

## 📌 Project Overview

Brain Tumor Detection is a deep learning-based computer vision project that uses the **YOLOv8 (You Only Look Once)** object detection algorithm to identify potential brain tumor regions from uploaded brain scan images.

The model is trained using a brain tumor dataset and integrated with a **Streamlit web application** to provide an interactive interface for uploading images and viewing detection results.

## 🎯 Objectives

- Detect potential brain tumor regions from brain scan images.
- Apply YOLOv8 for object detection.
- Train a deep learning model using a custom dataset.
- Display detected regions using bounding boxes.
- Build an interactive web application using Streamlit.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| YOLOv8 | Object detection |
| Ultralytics | YOLO model framework |
| OpenCV | Image processing |
| NumPy | Numerical operations |
| Pillow | Image handling |
| Streamlit | Web application |

## ⚙️ Project Workflow

1. Dataset collection and preparation.
2. Loading the YOLOv8n model.
3. Training the model on the brain tumor dataset.
4. Saving the trained model weights (`best.pt`).
5. Loading the trained model into Streamlit.
6. Uploading brain scan images.
7. Performing tumor region detection.
8. Displaying bounding boxes and confidence scores.

## 📂 Project Structure

```text
Brain_Tumor_Detection/
│
├── app.py
├── brain_tumor_detection.ipynb
├── best.pt
├── requirements.txt
└── README.md
```

## 🚀 Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/rajashekara-chary/Brain_Tumor_Detection.git
```

### 2. Navigate to the Project Folder

```bash
cd Brain_Tumor_Detection
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Model Weights

Place your trained `best.pt` file in the project directory.

The model weights are required to perform predictions.

### 5. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser, usually at:

```text
http://localhost:8501
```

## 💻 Application Features

- Upload brain scan images in JPG, JPEG, PNG, and BMP formats.
- Adjustable confidence threshold.
- YOLOv8-based object detection.
- Bounding box visualization.
- Detection class and confidence score display.
- Detection results in tabular format.
- Download annotated detection images.

## 🧠 Model Information

- **Model:** YOLOv8n
- **Framework:** Ultralytics
- **Task:** Object Detection
- **Image Size:** 640 × 640
- **Training Epochs:** 50

## 📊 Output

The application displays:

- Original uploaded brain scan image.
- Predicted image with bounding boxes.
- Detected class labels.
- Confidence scores.
- Detection coordinates.

## 🔮 Future Improvements

- Improve detection accuracy through hyperparameter tuning.
- Experiment with different YOLO architectures.
- Add detection history.
- Deploy the application using Streamlit Community Cloud or another hosting platform.
- Evaluate the model using precision, recall, mAP, and F1-score.

## ⚠️ Disclaimer

This project is developed for educational and research purposes only. Its predictions should not be considered a medical diagnosis. Always consult qualified medical professionals for medical interpretation.

## 👨‍💻 Author

**Rajashekara Chary Valaboju**

GitHub: [rajashekara-chary](https://github.com/rajashekara-chary)

---

⭐ If you find this project useful, consider giving the repository a star!
