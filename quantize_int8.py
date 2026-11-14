"""
yolov8n.onnx -> yolov8n_int8.onnx

Agirliklari 32 bit ondalikli (FP32) sayilardan 8 bit tam sayiya (INT8)
indirir. Model dosyasi ~4 kat kuculur, islemci tam sayi aritmetigini
daha hizli yapar. Bedeli: her agirlik yuvarlanir, dogruluk duseilebilir.

Bir kez calistirilir.
"""

from pathlib import Path

from onnxruntime.quantization import quantize_dynamic, QuantType

SRC = Path("yolov8n.onnx")
DST = Path("yolov8n_int8.onnx")

quantize_dynamic(
    model_input=SRC,
    model_output=DST,
    weight_type=QuantType.QUInt8,  # agirliklar icin 8 bit isaretsiz tam sayi
)

src_mb = SRC.stat().st_size / 1024 / 1024
dst_mb = DST.stat().st_size / 1024 / 1024

print(f"\n{SRC}  : {src_mb:.1f} MB")
print(f"{DST} : {dst_mb:.1f} MB")
print(f"Kucultme oranı: {src_mb / dst_mb:.1f}x")
