"""
README icin demo GIF'i uretir.

Videonun bir bolumunu isleyip, main.py'nin ekrana cizdiginin aynisini
(kutular, ROI, sayac) GIF olarak diske yazar. Ekran kaydedici gerekmez,
sonuc her calistirmada ayni.

GIF boyutu onemli: GitHub'da README icindeki buyuk bir GIF sayfayi
yavaslatir. Bu yuzden kareleri atlayarak ve kucultup yaziyoruz.
"""

import cv2
from PIL import Image

import config
from detector import Detector
from events import EventDetector

START_FRAME = 200      # ilk kareler bos; hareketin basladigi yerden al
GIF_FRAMES = 50        # GIF'e girecek kare sayisi
FRAME_STEP = 3         # her 3. kareyi al -> dosya boyutu ucte bir
WIDTH = 480            # GitHub'da README icinde bu genislik yeterli
FPS = 10
OUT = "demo.gif"


def main():
    detector = Detector()
    event_detector = EventDetector()

    cap = cv2.VideoCapture(config.VIDEO_SOURCE)
    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

    kareler = []
    sayac = 0

    while len(kareler) < GIF_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        # Her kareyi modele veriyoruz -- atlasaydik takip ID'leri kopardi.
        # Atlama sadece GIF'e hangi karelerin yazilacagini belirliyor.
        results = detector.track(frame)
        event_detector.check_event(results)

        sayac += 1
        if sayac % FRAME_STEP != 0:
            continue

        annotated = results.plot()

        # Once kucult, sonra ROI'yi ve sayaci ciz. Tersini yaparsak
        # kucultme cizgiyi de yaziyi da okunamayacak hale getiriyor.
        h, w = annotated.shape[:2]
        olcek = WIDTH / w
        kucuk = cv2.resize(annotated, (WIDTH, int(h * olcek)),
                           interpolation=cv2.INTER_AREA)

        cv2.rectangle(kucuk,
                      (int(config.ROI[0] * olcek), int(config.ROI[1] * olcek)),
                      (int(config.ROI[2] * olcek), int(config.ROI[3] * olcek)),
                      (0, 0, 255), 3)

        # Sayaci koyu bir seridin uzerine yaz: acik arka planda da okunsun.
        yazi = f"Bolgeye giren: {event_detector.toplam_sayi()}"
        (tw, th), _ = cv2.getTextSize(yazi, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(kucuk, (0, 0), (tw + 24, th + 20), (0, 0, 0), -1)
        cv2.putText(kucuk, yazi, (12, th + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # OpenCV BGR calisir, PIL RGB bekler
        # GIF 256 renk tutuyor; 128'e indirmek gozle fark edilmeden
        # dosyayi belirgin kuculuyor.
        pil = Image.fromarray(cv2.cvtColor(kucuk, cv2.COLOR_BGR2RGB))
        kareler.append(pil.convert("P", palette=Image.ADAPTIVE, colors=128))

    cap.release()

    if not kareler:
        print("Kare okunamadi. VIDEO_SOURCE ve START_FRAME degerlerini kontrol et.")
        return

    kareler[0].save(
        OUT,
        save_all=True,
        append_images=kareler[1:],
        duration=int(1000 / FPS),
        loop=0,
        optimize=True,
    )

    from pathlib import Path
    mb = Path(OUT).stat().st_size / 1024 / 1024
    print(f"\n{OUT} yazildi: {len(kareler)} kare, {mb:.1f} MB")


if __name__ == "__main__":
    main()
