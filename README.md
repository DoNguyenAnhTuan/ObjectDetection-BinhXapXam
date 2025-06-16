```markdown
# 🎯 Object Detection - "Bình Xập Xàm"

This project applies **YOLOv7** for object detection in chaotic, cluttered environments — nicknamed "Bình Xập Xàm", which refers to random, messy scenes such as shelves, boxes, or mixed electronic parts.

---

## 📌 Project Goals

- Train YOLOv7 on a custom dataset with unstructured layouts.
- Detect multiple overlapping objects efficiently.
- Evaluate real-world performance in chaotic visual environments.

---

## 🧠 Model

- 🔍 Model: [YOLOv7](https://github.com/WongKinYiu/yolov7)
- 🧰 Framework: PyTorch
- 📷 Input: RGB images (640×640)
- 📦 Output: Bounding boxes with class labels and confidence scores

---

## 📁 Folder Structure

```

.
├── data/                # Dataset (images + labels in YOLO format)
├── weights/             # Trained YOLOv7 weights
├── detect.py            # Inference script
├── train.py             # Training script
├── utils/               # Helper functions
├── runs/                # Output results from training/inference
└── README.md

````

---

## 🚀 Getting Started

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
````

2. **Train the model**

   ```bash
   python train.py --img 640 --batch 16 --epochs 50 --data data/custom.yaml --weights yolov7.pt
   ```

3. **Run inference**

   ```bash
   python detect.py --weights runs/train/exp/weights/best.pt --source path/to/image_or_video
   ```

---

## 📊 Results

* mAP\@0.5: *To be updated*
* Inference Speed: \~12ms/frame (on RTX 3060)

> The model performs well on visually messy scenes and has been tested on real-world bins, shelves, and random clutter.

---

## 📬 Contact

**Do Nguyen Anh Tuan**
📍 MSc Student in IT @ LHU | FabLab @ EIU
🔗 [Portfolio](https://donguyenanhtuan.github.io/AnhTuan-Portfolio/)
🐙 [GitHub](https://github.com/DoNguyenAnhTuan)

---

