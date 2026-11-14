"""
yolov8n.pt -> yolov8n_int8_openvino_model/  (INT8, OpenVINO)

Nicemlemeyi Ultralytics'in kendisi yapiyor: YOLO mimarisini bildigi icin
hassas katmanlari (tespit kafasi) nicemleme disinda birakiyor ve
kalibrasyonu dogru sekilde yurutuyor.

Calistirma tarafinda ONNX Runtime yerine OpenVINO var -- Intel'in kendi
cikarim motoru, islemcideki INT8 komutlarini gercekten kullaniyor.

data: kalibrasyon veri seti. coco128 kucuk bir COCO alt kumesi ve ilk
calistirmada kendiliginden iniyor.
"""

from ultralytics import YOLO

model = YOLO("yolov8n.pt")

path = model.export(
    format="openvino",
    int8=True,
    data="coco128.yaml",
    imgsz=640,
)

print(f"\nOpenVINO INT8 modeli yazildi: {path}")
