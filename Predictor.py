import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from collections import Counter
import numpy as np

# Khởi tạo một danh sách để lưu các bộ số dự đoán
predicted_sets = []

# Số lần dự đoán
num_predictions = 10

# Số lần lặp cho thuật toán Monte Carlo
num_iterations = 1000

for i in range(num_predictions):
    # Load dữ liệu từ tệp Excel
    data = pd.read_excel(r"C:\Users\judyh\Desktop\TKB\LotteryNumberPredictor-main\LotteryNumberPredictor-main\previous_data.xlsx")

    # Lọc dữ liệu để chỉ bao gồm các hàng mà tất cả các số nằm trong phạm vi [1, 45]
    filtered_data = data[(data['1st_number'].between(1, 55)) &
                         (data['2nd_number'].between(1, 55)) &
                         (data['3rd_number'].between(1, 55)) &
                         (data['4th_number'].between(1, 55)) &
                         (data['5th_number'].between(1, 55)) &
                         (data['6th_number'].between(1, 55))]

    # Chia dữ liệu thành các đặc trưng (X) và mục tiêu (y)
    X = filtered_data[['1st_number', '2nd_number', '3rd_number', '4th_number', '5th_number', '6th_number']]
    y = filtered_data.iloc[:, 1:]

    # Huấn luyện một mô hình Random Forest Regression
    model = RandomForestRegressor(n_estimators=1000, random_state=None)
    model.fit(X, y)

    # Khởi tạo mảng để lưu trữ các dự đoán từ các lần lặp Monte Carlo
    all_predictions = []

    for _ in range(num_iterations):
        # Tạo một bộ số ngẫu nhiên mới cho dự đoán
        new_data = pd.DataFrame({
            "1st_number": [np.random.randint(1, 45) for _ in range(100)],
            "2nd_number": [np.random.randint(1, 45) for _ in range(100)],
            "3rd_number": [np.random.randint(1, 45) for _ in range(100)],
            "4th_number": [np.random.randint(1, 45) for _ in range(100)],
            "5th_number": [np.random.randint(1, 45) for _ in range(100)],
            "6th_number": [np.random.randint(1, 45) for _ in range(100)],
        })

        # Sử dụng mô hình đã được huấn luyện để dự đoán 6 số tiếp theo cho mỗi bộ đặc trưng
        predictions = model.predict(new_data)

        # Lưu trữ các dự đoán từ mỗi lần lặp Monte Carlo
        all_predictions.extend(predictions)

    # Tính trung bình của các dự đoán từ tất cả các lần lặp Monte Carlo
    avg_predictions = np.mean(all_predictions, axis=0)

    # Lấy bộ số có khả năng cao nhất dựa trên trung bình của các dự đoán
    most_likely_set = avg_predictions

    # Chuyển most_likely_set thành các số nguyên
    rounded_most_likely_set = [round(x) for x in most_likely_set]
    # Sắp xếp tăng dần
    rounded_most_likely_set.sort()

    # Thêm bộ số dự đoán vào danh sách
    predicted_sets.append(rounded_most_likely_set)

    # In ra bộ số có khả năng cao nhất
    print(f"{i+1:02d}. Bộ số có khả năng cao nhất là:", rounded_most_likely_set)

# Đếm số lần xuất hiện của từng bộ số
set_counts = Counter(tuple(sorted(s)) for s in predicted_sets)

# Tìm bộ số có số lần xuất hiện nhiều nhất
most_common_set, most_common_count = set_counts.most_common(1)[0]

# In ra bộ số và số lần xuất hiện của nó
print("Bộ số phổ biến nhất là:", most_common_set)
print("Nó xuất hiện", most_common_count, "lần.")


