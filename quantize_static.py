"""
yolov8n.onnx -> yolov8n_static_int8.onnx

Statik INT8 nicemleme. Dinamikten farki: aktivasyonlarin olcekleri
calisma aninda degil, onceden -- gercek veriyle olculerek -- sabitlenir.
Bu sayede grafik QLinearConv dugumlerine donusuyor; ONNX Runtime'in bu
dugum icin optimize edilmis bir INT8 uygulamasi var.

Olcekleri belirlemek icin "kalibrasyon verisi" gerekiyor: modele birkac
yuz temsili goruntu gosterip her katmandan hangi degerlerin gectigini
olcuyoruz. Veri olarak kendi videomuzu kullaniyoruz.

Bir kez calistirilir.
"""

from pathlib import Path
from onnxruntime.quantization import CalibrationMethod

import cv2
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

SRC = "yolov8n.onnx"
PREP = "yolov8n_prep.onnx"          # on islemeden gecmis ara dosya
DST = "yolov8n_static_int8.onnx"
VIDEO = "test_pedestrians.mp4"

IMGSZ = 640
CALIB_FRAMES = 10      # yuzdelik kalibrasyon her tensorun histogramini
                       # biriktiriyor: cok kare = bellek patlamasi
CALIB_EVERY = 150      # yine videoya yayilmis ornekler


def letterbox(frame):
    """Kareyi, en-boy oranini bozmadan 640x640'a oturt.

    YOLO'nun kendi on islemesinin ayni: kisa kenar griyle doldurulur.
    Kalibrasyonun gecerli olmasi icin modelin gercekte gordugu veriyle
    ayni bicimde olmasi sart -- farkli olursa olcekler yanlis cikar.
    """
    h, w = frame.shape[:2]
    oran = min(IMGSZ / h, IMGSZ / w)
    yeni_w, yeni_h = int(round(w * oran)), int(round(h * oran))

    kucuk = cv2.resize(frame, (yeni_w, yeni_h), interpolation=cv2.INTER_LINEAR)

    tuval = np.full((IMGSZ, IMGSZ, 3), 114, dtype=np.uint8)
    ust = (IMGSZ - yeni_h) // 2
    sol = (IMGSZ - yeni_w) // 2
    tuval[ust:ust + yeni_h, sol:sol + yeni_w] = kucuk

    rgb = cv2.cvtColor(tuval, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0      # 0-255 -> 0.0-1.0
    x = x.transpose(2, 0, 1)                # HWC -> CHW
    return np.expand_dims(x, 0)             # parti boyutu ekle


def kalibrasyon_kareleri():
    cap = cv2.VideoCapture(VIDEO)
    kareler = []
    sayac = 0
    while len(kareler) < CALIB_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        sayac += 1
        if sayac % CALIB_EVERY == 0:
            kareler.append(letterbox(frame))
    cap.release()
    return kareler


class VideoKalibrasyonOkuyucu(CalibrationDataReader):
    """Nicemleme aracinin bekledigi arayuz: get_next() ile sirayla veri verir."""

    def __init__(self, girdi_adi, kareler):
        self.girdi_adi = girdi_adi
        self.sira = iter(kareler)

    def get_next(self):
        veri = next(self.sira, None)
        if veri is None:
            return None
        return {self.girdi_adi: veri}


def main():
    print("1/3  On isleme (sekil cozumleme)...")
    quant_pre_process(SRC, PREP)

    print("2/3  Kalibrasyon kareleri hazirlaniyor...")
    girdi_adi = ort.InferenceSession(
        PREP, providers=["CPUExecutionProvider"]
    ).get_inputs()[0].name
    kareler = kalibrasyon_kareleri()
    print(f"     {len(kareler)} kare, girdi adi: {girdi_adi}")

    print("3/3  Statik nicemleme (birkac dakika surebilir)...")
    quantize_static(
        model_input=PREP,
        model_output=DST,
        calibration_data_reader=VideoKalibrasyonOkuyucu(girdi_adi, kareler),
        quant_format=QuantFormat.QDQ,      # QLinearConv'a kaynasabilen bicim
        per_channel=True,                  # her filtre kendi olcegini alir
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        calibrate_method=CalibrationMethod.Percentile,
        extra_options={"CalibPercentile": 99.999},
    )

    src_mb = Path(SRC).stat().st_size / 1024 / 1024
    dst_mb = Path(DST).stat().st_size / 1024 / 1024
    print()
    print(f"{SRC} : {src_mb:.1f} MB")
    print(f"{DST} : {dst_mb:.1f} MB")


if __name__ == "__main__":
    main()
