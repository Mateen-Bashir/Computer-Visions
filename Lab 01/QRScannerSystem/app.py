import cv2
from datetime import datetime

detector = cv2.QRCodeDetector()
cap = cv2.VideoCapture(0)

scanned = set()

while True:
    ret, frame = cap.read()
    data,bbox,_ = detector.detectAndDecode(frame)

    if data:
        if data not in scanned:
            scanned.add(data)
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open("qr_log.txt","a") as f:
                f.write(data+" - "+now+"\n")

        cv2.putText(frame,data,(50,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,(0,255,0),2)

    cv2.imshow("QR Scanner",frame)

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()