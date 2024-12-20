import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

# 데이터 불러오기 및 전처리
train_data = pd.read_csv("data/train.csv")
train_data.fillna(train_data.mean(), inplace=True)

X_train = train_data[['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP', 'CO']]
y_train = train_data['NOX']

# 모델 학습
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# 교차 검증 및 성능 평가
cross_val_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
mean_mse = -cross_val_scores.mean()

y_pred = model.predict(X_train)

mse = mean_squared_error(y_train, y_pred)
r2 = r2_score(y_train, y_pred)

print(f"Cross-Validation Mean Squared Error: {mean_mse:.2f}")
print(f"Mean Squared Error on Train Data: {mse:.2f}")
print(f"R^2 Score on Train Data: {r2:.2f}")

# 그래프 1: 실제값 대 예측값 (True vs Predicted)
plt.figure(figsize=(10, 6))
plt.scatter(y_train, y_pred, color='blue', alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], color='red', linewidth=2)
plt.title('True vs Predicted NOX (Train Data)')
plt.xlabel('True NOX')
plt.ylabel('Predicted NOX')
plt.show()

# 그래프 2: 잔차 그래프 (Residual Plot)
residuals = y_train - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5, color='green')
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residual Plot')
plt.xlabel('Predicted NOX')
plt.ylabel('Residuals')
plt.show()

# 그래프 3: Feature Importance
feature_importances = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title('Feature Importance')
plt.show()

# 그래프 4: 분포 비교 (Distribution of True vs Predicted NOX)
sns.kdeplot(y_train, color='blue', label='True NOX')
sns.kdeplot(y_pred, color='orange', label='Predicted NOX')
plt.legend()
plt.title('Distribution of True vs Predicted NOX')
plt.show()



