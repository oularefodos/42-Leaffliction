import cv2 as cv
import matplotlib.pyplot as plt

def get_histogram(img_bgr):
    img_hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
    img_lab = cv.cvtColor(img_bgr, cv.COLOR_BGR2LAB)

    channels_data = [
        {"label": "red", "display_color": "red", "pixels": img_bgr[:, :, 2]},
        {"label": "green", "display_color": "green", "pixels": img_bgr[:, :, 1]},
        {"label": "blue", "display_color": "blue", "pixels": img_bgr[:, :, 0]},
        {"label": "lightness", "display_color": "darkgray", "pixels": img_lab[:, :, 0]},
        {"label": "green-magenta", "display_color": "pink", "pixels": img_lab[:, :, 1]},
        {"label": "blue-yellow", "display_color": "yellow", "pixels": img_lab[:, :, 2]},
        {"label": "hue", "display_color": "purple", "pixels": img_hsv[:, :, 0]},
        {"label": "saturation", "display_color": "cyan", "pixels": img_hsv[:, :, 1]},
        {"label": "value", "display_color": "orange", "pixels": img_hsv[:, :, 2]},
    ]

    plt.figure(figsize=(8, 6))
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.title("Color Channel Histogram", fontsize=16, fontweight='bold')
    plt.xlabel("Pixel Intensity", fontsize=14)
    plt.ylabel("Proportion of pixels (%)", fontsize=14)
    # plt.xlim([0, 256])
    plt.grid(True, alpha=1)

    for channel in channels_data:

        hist = cv.calcHist(
            images=[channel['pixels']],
            channels=[0],
            mask=None,
            histSize=[256],
            ranges=[0, 256],        
        )

        hist = (hist / hist.sum()) * 100
        plt.plot(hist, color=channel['display_color'], label=channel['label'])

    plt.legend(title="Color Channel", loc="upper right")
    
    return plt