# Real-Time Face Detection using Deep Learning (Custom Model)

This project implements a real-time face detection system built entirely from scratch. The model is trained on custom data and deployed using **Streamlit** and **OpenCV**. It detects a specific face (my own) with decent accuracy — even without using a GPU.

---

## 📌 Project Overview

- 📸 Collected and annotated images of my own face using **LabelMe**
- 🔄 Performed data augmentation to create a robust dataset (~5400 images)
- 🧠 Designed a **custom model** with:
  - A **classification head**: detects if a face is present
  - A **regression head**: predicts bounding box coordinates
- ⚙️ Trained the model with two separate **custom loss functions**
- 💾 Implemented **checkpointing** to resume training after shutdowns
- 🖥️ Deployed the model using **Streamlit** for real-time detection with webcam

---

## 🛠️ Tech Stack

- Python 3.10  
- TensorFlow / Keras  
- OpenCV  
- Streamlit  
- LabelMe (for manual annotation)  
- Matplotlib, NumPy

---

