import cv2
from ultralytics import YOLO

# 1. Khởi tạo mô hình
model = YOLO('yolov8n.pt')

# 2. Thông số thực tế
h_that = 1.2  # Chiều cao thực tế của xe máy (mét)
f_camera = None

# Danh sách ảnh và khoảng cách thật tương ứng
# (Thay số 5.0 ở ảnh 2 bằng khoảng cách thực tế bạn đo bằng thước)
danh_sach_anh = [
    {"file": "anh1.jpg", "d_that": 3.0},  # Ảnh 1: mốc 3.0m dùng làm chuẩn tính f
    {"file": "anh2.jpg", "d_that": 5.0},  # Sửa 4.0 thành khoảng cách thật của anh2
    {"file": "anh3.jpg", "d_that": 7.0},  # Sửa 5.0 thành khoảng cách thật của anh3
    {"file": "anh4.jpg", "d_that": 10.0},  # Sửa 6.0 thành khoảng cách thật của anh4
    {"file": "anh5.jpg", "d_that": 15.0},  # Sửa 7.0 thành khoảng cách thật của anh5
    {"file": "anh6.jpg", "d_that": 20.0},  # Sửa 8.0 thành khoảng cách thật của anh6
]

print(f"{'File':<18} | {'h_ảnh (px)':<10} | {'d_tính_toán':<12} | {'d_thực_tế':<10} | {'Sai số'}")
print("-" * 75)

for i, data in enumerate(danh_sach_anh):
    frame = cv2.imread(data["file"])
    if frame is None:
        print(f"Không đọc được ảnh: {data['file']} (Kiểm tra lại xem ảnh đã chép vào thư mục TEST chưa)")
        continue
        
    results = model(frame, verbose=False)[0]
    
    xe_may_box = None
    for box in results.boxes:
        if int(box.cls[0]) == 3:  # Class 3 trong COCO là motorcycle
            xe_may_box = box.xyxy[0].tolist()
            break
            
    if xe_may_box:
        y1, y2 = xe_may_box[1], xe_may_box[3]
        h_anh = y2 - y1
        
        # Bức ảnh đầu tiên (3m) dùng để tìm tiêu cự f của camera
        if i == 0:
            f_camera = (data["d_that"] * h_anh) / h_that
            d_tinh = data["d_that"]
            print(f"-> Đã tìm được tiêu cự f của camera: {f_camera:.2f} pixel")
        else:
            # Các ảnh sau dùng f vừa tìm được để tính khoảng cách
            d_tinh = (h_that * f_camera) / h_anh
            
        sai_so = abs(d_tinh - data["d_that"])
        print(f"{data['file']:<18} | {h_anh:<10.1f} | {d_tinh:<12.2f} | {data['d_that']:<10.2f} | {sai_so:.2f} m")
    else:
        print(f"{data['file']:<18} | Không tìm thấy xe máy")