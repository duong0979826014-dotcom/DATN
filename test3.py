"""
Chương trình Ước lượng Khoảng cách Tối ưu (Pinhole-ARLS)
Hiển thị chi tiết Sai số tuyệt đối (m) và Sai số tương đối (%)
"""
import cv2
import numpy as np
from ultralytics import YOLO

# 1. Khai báo dữ liệu thực địa
DATA = {
    "anh1.jpg": 3.0,
    "anh2.jpg": 5.0,
    "anh3.jpg": 7.0,
    "anh4.jpg": 10.0,
    "anh5.jpg": 15.0,
    "anh6.jpg": 20.0,
}

MOTORCYCLE_CLASS = 3  # Class COCO cho xe máy

def calibrate_and_predict():
    model = YOLO("yolov8n.pt")
    records = []

    # Bước 1: Trích xuất thông số khung bao từ các ảnh thực địa
    for img_name, z_true in DATA.items():
        img = cv2.imread(img_name)
        if img is None:
            continue
        
        results = model(img, classes=[MOTORCYCLE_CLASS], verbose=False)[0]
        if len(results.boxes) == 0:
            continue
            
        boxes = [tuple(float(v) for v in b.xyxy[0].tolist()) for b in results.boxes]
        x1, y1, x2, y2 = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        
        w_px = x2 - x1
        h_px = y2 - y1
        aspect_ratio = h_px / w_px
        
        records.append({
            "name": img_name,
            "z_true": z_true,
            "w_px": w_px,
            "h_px": h_px,
            "ratio": aspect_ratio
        })

    z_trues = np.array([r["z_true"] for r in records])
    w_pxs = np.array([r["w_px"] for r in records])

    # Bước 2: Hồi quy tuyến tính Z = k_w / w + c
    u_w = 1.0 / w_pxs
    A = np.vstack([u_w, np.ones(len(u_w))]).T
    k_w, c = np.linalg.lstsq(A, z_trues, rcond=None)[0]

    # Bước 3: In bảng kết quả hiển thị Sai số (%)
    print("\n" + "="*86)
    print("      THUẬT TOÁN TỐI ƯU KẾT HỢP (PINHOLE-ARLS DISTANCE ESTIMATION)")
    print("="*86)
    print(f"Hằng số hiệu chuẩn tối ưu: k_w = {k_w:.2f} | Bù hằng số c = {c:.2f} m\n")
    
    print(f"{'Tên Ảnh':<10} | {'Z Thực (m)':<10} | {'w (px)':<8} | {'h/w Ratio':<10} | {'Z Dự Đoán (m)':<13} | {'Sai Số (m)':<10} | {'Sai Số (%)':<10}")
    print("-" * 86)

    errors = []
    for r in records:
        z_pred = (k_w / r["w_px"]) + c
        error_m = abs(z_pred - r["z_true"])
        error_pct = (error_m / r["z_true"]) * 100
        errors.append(error_m)

        print(f"{r['name']:<10} | {r['z_true']:<10.1f} | {r['w_px']:<8.0f} | {r['ratio']:<10.2f} | {z_pred:<13.2f} | {error_m:<10.2f} | {error_pct:<9.1f}%")

    mae = np.mean(errors)
    mape = np.mean(np.array(errors) / z_trues) * 100

    print("-" * 86)
    print(f"-> MAE (Sai số tuyệt đối trung bình) : {mae:.2f} mét")
    print(f"-> MAPE (Sai số % trung bình toàn dải): {mape:.1f} %\n")

if __name__ == "__main__":
    calibrate_and_predict()