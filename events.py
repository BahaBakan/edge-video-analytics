

import config 

class EventDetector:
    def __init__(self):
        self.giren_idler = set()

    def check_event(self, result):

        yeni_girenler = []
        if result.boxes.id is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                merkez_x = int((x1 + x2) / 2)
                merkez_y = int((y1 + y2) / 2)       

                if config.ROI[0] < merkez_x < config.ROI[2] and config.ROI[1] < merkez_y < config.ROI[3]:
                        nesne_id = int(box.id[0])
                        if nesne_id not in self.giren_idler:
                            self.giren_idler.add(nesne_id)
                            yeni_girenler.append(nesne_id)

        return yeni_girenler


    def toplam_sayi(self):
         return len(self.giren_idler)