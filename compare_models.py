"""
FP32 ONNX ile INT8 ONNX modellerinin ciktilarini karsilastirir.

Mantik evaluation.py ile ayni: bir referans al, ona gore olc.
Burada referans FP32 modelin ciktisi. Mutlak dogruluk degil --
"nicemleme neyi degistirdi" sorusunu cevapliyor.

Takip (track) yerine tespit (predict) kullaniyoruz: takip algoritmasi
kareler arasi durum tutuyor, bu da farki modelden mi takipciden mi
geldigini belirsizlestirirdi.
"""

import cv2
import numpy as np
from ultralytics import YOLO

FP32_MODEL = "yolov8n.pt"
INT8_MODEL = "yolov8n_int8_openvino_model"
VIDEO = "test_pedestrians.mp4"

SAMPLE_EVERY = 10     # her 10. kareyi al: iki modeli 1919 kare boyunca
                      # calistirmak ~16 dk surerdi, ornekleme yeterli
IOU_THRESHOLD = 0.5   # iki kutu "ayni nesne" sayilmasi icin gereken ortusme


def iou(a, b):
    """Iki kutunun ortusme orani (Intersection over Union)."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    kesisim = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if kesisim == 0:
        return 0.0

    alan_a = (a[2] - a[0]) * (a[3] - a[1])
    alan_b = (b[2] - b[0]) * (b[3] - b[1])
    return kesisim / (alan_a + alan_b - kesisim)


def tespitler(model, frame):
    """Bir karedeki insan tespitleri: kutular ve guven skorlari."""
    r = model.predict(frame, classes=[0], verbose=False)[0]
    kutular = r.boxes.xyxy.cpu().numpy()
    skorlar = r.boxes.conf.cpu().numpy()
    return kutular, skorlar


def main():
    fp32 = YOLO(FP32_MODEL)
    int8 = YOLO(INT8_MODEL)

    cap = cv2.VideoCapture(VIDEO)

    kare_no = 0
    olculen = 0

    fp32_toplam = 0      # FP32'nin bulduğu toplam tespit
    int8_toplam = 0      # INT8'in bulduğu toplam tespit
    eslesen = 0          # iki modelde de ayni yerde cikan tespit
    iou_toplam = 0.0     # eslesen kutularin ortusme toplami
    fp32_skor_toplam = 0.0
    int8_skor_toplam = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        kare_no += 1
        if kare_no % SAMPLE_EVERY != 0:
            continue

        a_kutu, a_skor = tespitler(fp32, frame)
        b_kutu, b_skor = tespitler(int8, frame)

        fp32_toplam += len(a_kutu)
        int8_toplam += len(b_kutu)
        fp32_skor_toplam += float(a_skor.sum())
        int8_skor_toplam += float(b_skor.sum())

        # Her FP32 kutusu icin en cok ortusen INT8 kutusunu bul.
        # Bir INT8 kutusu yalnizca bir kez eslesebilir.
        kullanilan = set()
        for ak in a_kutu:
            en_iyi_iou = 0.0
            en_iyi_idx = -1
            for i, bk in enumerate(b_kutu):
                if i in kullanilan:
                    continue
                o = iou(ak, bk)
                if o > en_iyi_iou:
                    en_iyi_iou = o
                    en_iyi_idx = i

            if en_iyi_iou >= IOU_THRESHOLD:
                eslesen += 1
                iou_toplam += en_iyi_iou
                kullanilan.add(en_iyi_idx)

        olculen += 1

    cap.release()

    print()
    print("--- FP32 vs INT8 karsilastirmasi ---")
    print(f"Olculen kare      : {olculen} (her {SAMPLE_EVERY}. kare)")
    print()
    print(f"FP32 tespit       : {fp32_toplam}")
    print(f"INT8 tespit       : {int8_toplam}")
    print(f"Tespit farki      : {int8_toplam - fp32_toplam:+d} "
          f"({(int8_toplam - fp32_toplam) / fp32_toplam * 100:+.1f}%)")
    print()
    print(f"Eslesen tespit    : {eslesen} / {fp32_toplam} "
          f"(%{eslesen / fp32_toplam * 100:.1f})")
    print(f"Kacirilani        : {fp32_toplam - eslesen}")
    print(f"Ortalama IoU      : {iou_toplam / eslesen:.3f}" if eslesen else "")
    print()
    print(f"FP32 ort. guven   : {fp32_skor_toplam / fp32_toplam:.3f}")
    print(f"INT8 ort. guven   : {int8_skor_toplam / int8_toplam:.3f}")


if __name__ == "__main__":
    main()
