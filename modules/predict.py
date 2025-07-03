import pandas as pd
import numpy as np
import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

model_point = []

for points in range(0, 55):
    df = data_new[data_new['location_id'] == points]
    df["Datetime"] = pd.to_datetime(df["Datetime"], format="%H:%M %d/%m/%Y")
    # df.sort_values("Datetime", inplace=True)
    print(df.columns.tolist())

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace(',', '.', regex=False)

    # Chuyển về kiểu float, trừ cột thời gian
    for col in df.columns:
        if col != 'Datetime':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    data = df[[
        "PM2.5",
        # "location_id"
        ]].values
    
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # ==== 3. Hàm tạo chuỗi thời gian ====
    def create_sequences(data, seq_len):
        xs, ys = [], []
        for i in range(seq_len, len(data)):
            x = data[i-seq_len:i]
            y = data[i]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    SEQ_LEN = 24  # dùng 5 bước trước để dự đoán bước tiếp theo
    X, y = create_sequences(data_scaled, SEQ_LEN)
    # ==== 4. Chia train/test ====
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)
    X_train = torch.nan_to_num(X_train)
    y_train = torch.nan_to_num(y_train)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    BATCH_SIZE = 128
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    epochs = 100
    
    for epoch in range(epochs):

        model = LSTMModel()
        # ==== 6. Train ====
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        model.train()
        for xb, yb in train_loader:
            
            output = model(xb)
            loss = loss_fn(output, yb)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # ==== 7. Dự đoán ====
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).numpy()
        y_test_inv = scaler.inverse_transform(y_test)
        y_pred_inv = scaler.inverse_transform(y_pred)
    model_point.append(model)
    # ==== 8. Hiển thị kết quả ====
    print("\n📈 Dự đoán PM2.5:")
    for real, pred in zip(y_test_inv[:5], y_pred_inv[:5]):
        print(f"Thực tế: {real[0]:.2f} - Dự đoán: {pred[0]:.2f}")
model_point