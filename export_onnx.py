"""
yolov8n.pt -> yolov8n.onnx

Modeli PyTorch formatindan ONNX formatina cevirir.
Bir kez calistirilir; sonuc dosyasi diske yazilir.
"""

from ultralytics import YOLO

import config

model = YOLO("yolov8n.pt")

path = model.export(
    format="onnx",
    imgsz=640,      # sabit girdi boyutu: optimizasyon icin grafik sekli bilinmeli
    opset=13,       # ONNX islem seti surumu; 13 genis destekli ve YOLOv8 ile uyumlu
    simplify=True,  # grafikteki gereksiz dugumleri temizle
    dynamic=False,  # degisken boyut yok -> daha agresif optimizasyon mumkun
)

print(f"\nONNX modeli yazildi: {path}")
