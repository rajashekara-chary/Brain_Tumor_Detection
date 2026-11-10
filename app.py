
import streamlit as st
import os
import numpy as np

from ultralytics import YOLO
from PIL import Image


# --------------------------------------------------
# 1. Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Brain Tumor Detection System")

st.write(
    "Upload a brain scan image to detect potential tumor regions "
    "using a trained YOLOv8 model."
)


# --------------------------------------------------
# 2. Model Settings
# --------------------------------------------------

st.sidebar.header("Model Settings")

model_path = st.sidebar.text_input(
    "Model weights path",
    value="best.pt"
)

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.25,
    step=0.05
)


# --------------------------------------------------
# 3. Load YOLO Model
# --------------------------------------------------

@st.cache_resource
def load_model(weights_path):
    return YOLO(weights_path)


if not os.path.exists(model_path):

    st.error(
        f"Model weights not found: {model_path}"
    )

    st.info(
        "Place your trained best.pt file in the same folder "
        "as app.py or enter its correct path."
    )

    st.stop()


try:

    model = load_model(model_path)

    st.sidebar.success("Model loaded successfully!")

except Exception as e:

    st.error(f"Error loading YOLO model: {e}")

    st.stop()


# --------------------------------------------------
# 4. Upload Brain Scan Image
# --------------------------------------------------

st.subheader("Upload Brain Scan Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "bmp"]
)


# --------------------------------------------------
# 5. Image Processing and Prediction
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    image_array = np.array(image)

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Original Image")

        st.image(
            image,
            caption="Uploaded Brain Scan",
            use_container_width=True
        )

    with col2:

        st.subheader("Image Information")

        st.write("Image Name:", uploaded_file.name)

        st.write("Image Size:", image.size)

        st.write("Image Format:", image.format or "Converted to RGB")


    # --------------------------------------------------
    # 6. Detect Brain Tumor
    # --------------------------------------------------

    if st.button(
        "🔍 Detect Brain Tumor",
        type="primary",
        use_container_width=True
    ):

        with st.spinner("Analyzing brain scan..."):

            try:

                results = model.predict(
                    source=image_array,
                    conf=confidence,
                    imgsz=640,
                    verbose=False
                )

                result = results[0]


                # --------------------------------------------------
                # 7. Display Annotated Image
                # --------------------------------------------------

                annotated_image = result.plot()

                # Convert BGR to RGB

                annotated_image = annotated_image[:, :, ::-1]

                st.subheader("Detection Output")

                st.image(
                    annotated_image,
                    caption="YOLOv8 Detection Result",
                    use_container_width=True
                )


                # --------------------------------------------------
                # 8. Detection Results
                # --------------------------------------------------

                boxes = result.boxes

                if boxes is None or len(boxes) == 0:

                    st.warning(
                        "No objects detected above the selected "
                        "confidence threshold. This does not prove "
                        "that the scan is tumor-free."
                    )

                else:

                    st.success(
                        f"Total detections: {len(boxes)}"
                    )

                    detection_data = []

                    for box in boxes:

                        class_id = int(box.cls[0].item())

                        class_name = result.names.get(
                            class_id,
                            str(class_id)
                        )

                        score = float(box.conf[0].item())

                        x1, y1, x2, y2 = [
                            round(float(value), 2)
                            for value in box.xyxy[0].tolist()
                        ]

                        detection_data.append({

                            "Detected Class": class_name,

                            "Confidence": f"{score * 100:.2f}%",

                            "X1": x1,

                            "Y1": y1,

                            "X2": x2,

                            "Y2": y2

                        })


                    # Display detection table

                    st.subheader("Detection Details")

                    st.dataframe(
                        detection_data,
                        use_container_width=True,
                        hide_index=True
                    )


                    # --------------------------------------------------
                    # 9. Download Annotated Image
                    # --------------------------------------------------

                    annotated_pil = Image.fromarray(
                        annotated_image.astype("uint8")
                    )

                    from io import BytesIO

                    image_buffer = BytesIO()

                    annotated_pil.save(
                        image_buffer,
                        format="PNG"
                    )

                    st.download_button(
                        label="⬇️ Download Detection Image",
                        data=image_buffer.getvalue(),
                        file_name="brain_tumor_detection.png",
                        mime="image/png"
                    )

                    st.caption(
                        "Confidence represents the model's detection "
                        "score, not a medical probability."
                    )


            except Exception as e:

                st.error(
                    f"Error during prediction: {e}"
                )


# --------------------------------------------------
# 10. Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Brain Tumor Detection using YOLOv8 | "
    "Educational and research demonstration only. "
    "This application is not a medical diagnostic tool. "
    "Consult a qualified medical professional for scan interpretation."
)
