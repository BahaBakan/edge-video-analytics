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

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection and tracking
    results = detector.track(frame)

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