"""
Kiểm tra công thức hình học (Known-Height / Known-Width) bằng vài ảnh có khoảng cách thật.

Sửa DATA (tên ảnh -> khoảng cách thật từ camera tới xe, mét) và H_REAL, rồi chạy:
    python six_photos.py
Ảnh để cùng thư mục. Chương trình lưu kiemtra_<tên ảnh> để em xem khung YOLO có đúng không.
"""
import cv2
import numpy as np

DATA = {
    "anh1.jpg": 3.0,
    "anh2.jpg": 5.0,
    "anh3.jpg": 7.0,
    "anh4.jpg": 10.0,
    "anh5.jpg": 15.0,
    "anh6.jpg": 20.0,
}
H_REAL = 1.2        # Chiều cao thật của xe máy (m), đo bằng thước; dùng để in ra tiêu cự f
MOTORCYCLE = 3      # Lớp COCO cho xe máy


def get_boxes():
    from ultralytics import YOLO
    yolo = YOLO("yolov8n.pt")
    rows = []
    for name, z in DATA.items():
        img = cv2.imread(name)
        if img is None:
            print("Không đọc được ảnh:", name)
            continue
        res = yolo(img, classes=[MOTORCYCLE], verbose=False)[0]
        if len(res.boxes) == 0:
            print("YOLO không thấy xe máy trong", name)
            continue
        boxes = [tuple(float(v) for v in b.xyxy[0].tolist()) for b in res.boxes]
        x1, y1, x2, y2 = max(boxes, key=lambda t: (t[2] - t[0]) * (t[3] - t[1]))
        rows.append((name, z, y2 - y1, x2 - x1))  # (tên, Z thật, chiều cao px, chiều rộng px)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
        cv2.imwrite("kiemtra_" + name, img)
    return rows


def fit_k(z, h):
    """h = k / Z  (hồi quy qua gốc tọa độ), với k = f * kích thước thật."""
    u = 1.0 / z
    return float((h * u).sum() / (u * u).sum())


def analyze(rows, col, ten):
    z = np.array([r[1] for r in rows])
    h = np.array([r[col] for r in rows])
    n = len(z)
    if n < 3:
        print("Cần ít nhất 3 ảnh hợp lệ")
        return
    k = fit_k(z, h)
    print(f"\n=== Theo {ten} khung bao: Z = k / {ten} ===")
    print(f"k = {k:.1f} (k = f x kích thước thật)")
    if col == 2:
        print(f"Tiêu cự suy ra f = k / H_REAL = {k / H_REAL:.0f} px")
    print(f"{'ảnh':>10} {'Z thật':>7} {'px':>7} {'Z (fit hết)':>12} {'sai số %':>9} {'Z (bỏ ảnh này)':>15} {'sai số %':>9}")
    e_all, e_loo = [], []
    for i in range(n):
        keep = np.arange(n) != i
        zh = k / h[i]
        zh_loo = fit_k(z[keep], h[keep]) / h[i]
        e_all.append(abs(zh - z[i]))
        e_loo.append(abs(zh_loo - z[i]))
        print(f"{rows[i][0]:>10} {z[i]:7.1f} {h[i]:7.0f} {zh:12.2f} {e_all[-1] / z[i] * 100:9.1f} "
              f"{zh_loo:15.2f} {e_loo[-1] / z[i] * 100:9.1f}")
    e_all, e_loo = np.array(e_all), np.array(e_loo)
    print(f"Fit hết : MAE = {e_all.mean():.2f} m | MAPE = {(e_all / z).mean() * 100:.1f} %")
    print(f"Bỏ từng ảnh (đánh giá trung thực hơn): MAE = {e_loo.mean():.2f} m | MAPE = {(e_loo / z).mean() * 100:.1f} %")


if __name__ == "__main__":
    rows = get_boxes()
    analyze(rows, 2, "chiều cao")
    analyze(rows, 3, "chiều rộng")