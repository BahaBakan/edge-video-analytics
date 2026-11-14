import time

import cv2
import config
from detector import Detector
from events import EventDetector
from storage import Storage


# Create objects for Detector, EventDetector, and Storage
detector = Detector()
event_detector = EventDetector()
storage = Storage()

# Open video capture
cap = cv2.VideoCapture(config.VIDEO_SOURCE)

# --- Benchmark ---
# İlk karelerde model kendini ısıtıyor (bellek ayırma, ilk çağrı maliyeti).
# O kareler gerçek hızı temsil etmiyor, bu yüzden ölçüme katmıyoruz.
WARMUP_FRAMES = 5
frame_count = 0          # okunan toplam kare
measured_frames = 0      # ölçüme dahil edilen kare
total_inference = 0.0    # ölçülen karelerin toplam çıkarım süresi (saniye)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Perform object detection and tracking
    # Sadece bu çağrının süresini ölçüyoruz: video okuma, çizim ve ekrana
    # basma ölçümün dışında kalsın, çünkü nicemleme yalnızca modeli etkiliyor.
    start = time.perf_counter()
    results = detector.track(frame)
    inference_time = time.perf_counter() - start

    if frame_count > WARMUP_FRAMES:
        total_inference += inference_time
        measured_frames += 1

    # Check for new events
    yeni_girenler = event_detector.check_event(results)

    # Save new events to the database
    for nesne_id in yeni_girenler:
        storage.kaydet(nesne_id, "Bölgeye giriş")
        print(f"ID {nesne_id} bölgeye girdi ve kaydedildi.")



    annotated_frame = results.plot()
    cv2.rectangle(annotated_frame, (config.ROI[0], config.ROI[1]),
              (config.ROI[2], config.ROI[3]), (0, 0, 255), 2)
    #ROI rectangle
    
    cv2.putText(annotated_frame, f"Bolgeye giren: {event_detector.toplam_sayi()}",
            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    #exit on 'q' key press
    cv2.imshow("video", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


    # Release resources
cap.release()
cv2.destroyAllWindows()
storage.kapat()

# --- Benchmark sonucu ---
if measured_frames > 0:
    avg_ms = (total_inference / measured_frames) * 1000
    fps = measured_frames / total_inference
    print()
    print("--- Benchmark ---")
    print(f"Model           : {config.MODEL_PATH}")
    print(f"Olculen kare    : {measured_frames} (ilk {WARMUP_FRAMES} kare atlandi)")
    print(f"Toplam cikarim  : {total_inference:.2f} s")
    print(f"Kare basina     : {avg_ms:.1f} ms")
    print(f"Cikarim FPS     : {fps:.1f}")
else:
    print("Olculecek kadar kare islenmedi.")
