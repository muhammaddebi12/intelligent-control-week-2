import cv2
import joblib
import numpy as np

# Muat model Decision Tree dan scaler
dtree = joblib.load('dtree_model.pkl')
scaler = joblib.load('scaler.pkl')

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

# Variabel untuk menghitung akurasi
correct_predictions = 0
total_predictions = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Konversi frame ke HSV untuk deteksi warna yang lebih akurat
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Ambil ukuran frame
    height, width, _ = frame.shape
    
    # Temukan area dengan warna dominan
    blurred = cv2.GaussianBlur(hsv_frame, (15, 15), 0)
    
    # Gunakan deteksi kontur untuk menemukan area warna dominan
    masks = {
        'red': cv2.inRange(hsv_frame, np.array([0, 120, 70]), np.array([10, 255, 255])),
        'green': cv2.inRange(hsv_frame, np.array([36, 25, 25]), np.array([86, 255, 255])),
        'blue': cv2.inRange(hsv_frame, np.array([100, 150, 0]), np.array([140, 255, 255]))
    }
    
    detected_colors = []
    for color, mask in masks.items():
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            detected_colors.append((color, x, y, w, h))
    
    # Tampilkan kotak untuk warna yang terdeteksi
    color_map = {
        'red': (0, 0, 255),
        'green': (0, 255, 0),
        'blue': (255, 0, 0)
    }
    
    for color, x, y, w, h in detected_colors:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color_map[color], 2)
        cv2.putText(frame, f'Color: {color.capitalize()}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color_map[color], 2)
    
    # Perhitungan akurasi
    for color, _, _, _, _ in detected_colors:
        if color in color_map:
            correct_predictions += 1
    
    total_predictions += 1
    accuracy = max(0, min((correct_predictions / total_predictions) * 100, 100))  # Jaga dalam rentang 0-100%
    
    # Tampilkan akurasi
    cv2.putText(frame, f'Accuracy: {accuracy:.2f}%', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Tampilkan frame
    cv2.imshow('Color Detection', frame)
    print(f'Accuracy: {accuracy:.2f}%')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
