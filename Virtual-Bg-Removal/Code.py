import cv2
import numpy as np
from rembg import remove

# Optional: load a custom background image and resize to frame size
custom_bg = cv2.imread("virtual_bg.jpg")
custom_bg = cv2.resize(custom_bg, (640, 480))  # Adjust based on your cam resolution

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    # Rembg wants RGB
    fg_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Segmentation (returns RGBA)
    fg_removed = remove(fg_img)
    fg_removed = cv2.cvtColor(fg_removed, cv2.COLOR_RGBA2BGRA)

    # Prepare masks
    alpha = fg_removed[:, :, 3] / 255.0  # Foreground mask in [0,1]
    alpha_3ch = np.dstack([alpha, alpha, alpha])
    # Blend foreground/background
    blended = fg_removed[:, :, :3] * alpha_3ch + custom_bg * (1 - alpha_3ch)
    blended = blended.astype(np.uint8)

    cv2.imshow("Virtual Background", blended)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
