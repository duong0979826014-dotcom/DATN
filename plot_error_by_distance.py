import os
import matplotlib.pyplot as plt
import pandas as pd

# 1. Dữ liệu chính xác từ đồ thị gốc của bạn (3m - 25m)
data = {
    "distance": [3, 5, 7, 10, 15, 20, 25],
    "abs_error": [0.11, 0.62, 0.30, 0.47, 0.96, 1.58, 3.18]
}

summary = pd.DataFrame(data)

# 2. Tự động tạo thư mục lưu kết quả nếu chưa có
os.makedirs("results/plots", exist_ok=True)

# 3. Vẽ biểu đồ đúng chuẩn giao diện đồ thị gốc
plt.figure(figsize=(8, 5))

plt.plot(
    summary["distance"],
    summary["abs_error"],
    marker="o",
    linewidth=2,
    color="#1f77b4"
)

# Thêm nhãn giá trị tại từng điểm mốc
for _, row in summary.iterrows():
    plt.text(
        row["distance"] + 0.3,
        row["abs_error"],
        f"{row['abs_error']:.2f}",
        fontsize=9.5,
        va="center"
    )

plt.title("MAE theo khoảng cách", fontsize=12)
plt.xlabel("Khoảng cách thực (m)", fontsize=10)
plt.ylabel("MAE (m)", fontsize=10)
plt.grid(True, linestyle="-", alpha=0.7)

# Căn chỉnh lề khoảng giá trị các trục
plt.xlim(1, 26.5)
plt.ylim(-0.1, 3.4)

output_path = "results/plots/error_by_distance.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nSaved: {output_path}")