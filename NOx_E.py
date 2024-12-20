import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

# 데이터 불러오기 및 전처리
train_data = pd.read_csv("data/train.csv")  # CSV 파일에서 데이터 불러오기
train_data.fillna(train_data.mean(), inplace=True)  # 결측값을 각 열의 평균으로 대체

# 독립 변수(X)와 종속 변수(y) 설정
X_train = train_data[['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP', 'CO']]
y_train = train_data['NOX']

# 모델 학습
model = RandomForestRegressor(random_state=42)  # 랜덤 포레스트 회귀 모델 초기화 (재현성을 위한 random_state 설정)
model.fit(X_train, y_train)  # 학습 데이터를 사용하여 모델 학습

# 교차 검증 및 성능 평가
cross_val_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
# 교차 검증을 통해 모델의 안정성과 성능 평가 (MSE 기준)

mean_mse = -cross_val_scores.mean()  # 평균 MSE 계산 (음수 값을 양수로 변환)

y_pred = model.predict(X_train)  # 학습 데이터에 대해 예측값 생성

mse = mean_squared_error(y_train, y_pred)  # 학습 데이터의 MSE 계산
r2 = r2_score(y_train, y_pred)  # 학습 데이터의 결정 계수(R²) 계산

# 결과 출력
print(f"Cross-Validation Mean Squared Error: {mean_mse:.2f}")
print(f"Mean Squared Error on Train Data: {mse:.2f}")
print(f"R^2 Score on Train Data: {r2:.2f}")

# 결과 디렉터리 생성
result_dir = "result"  # 저장 디렉터리 이름
os.makedirs(result_dir, exist_ok=True)  # 디렉터리가 없으면 생성, 이미 존재하면 무시

# 그래프 1: 실제값 대 예측값 (True vs Predicted)
plt.figure(figsize=(10, 6))  # 그래프 크기 설정
plt.scatter(y_train, y_pred, color='blue', alpha=0.5)  # 실제값과 예측값의 산점도 생성
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], color='red', linewidth=2)
# 빨간색 선을 기준선으로 추가 (완벽한 예측 시 위치)
plt.title('True vs Predicted NOX (Train Data)')
plt.xlabel('True NOX')
plt.ylabel('Predicted NOX')
plt.savefig(os.path.join(result_dir, 'true_vs_predicted.png'))  # 그래프를 파일로 저장
plt.close()  # 그래프 닫기 (메모리 절약)

# 그래프 2: 잔차 그래프 (Residual Plot)
residuals = y_train - y_pred  # 잔차 계산 (실제값 - 예측값)
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5, color='green')  # 잔차 산점도
plt.axhline(y=0, color='red', linestyle='--')  # y=0 기준선 추가
plt.title('Residual Plot')
plt.xlabel('Predicted NOX')
plt.ylabel('Residuals')
plt.savefig(os.path.join(result_dir, 'residual_plot.png'))  # 그래프 저장
plt.close()

# 그래프 3: Feature Importance
feature_importances = pd.DataFrame({
    'Feature': X_train.columns,  # 독립 변수 이름
    'Importance': model.feature_importances_  # 랜덤 포레스트에서 계산된 특성 중요도
}).sort_values(by='Importance', ascending=False)  # 중요도 순으로 정렬

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances)  # 막대 그래프 생성
plt.title('Feature Importance')
plt.savefig(os.path.join(result_dir, 'feature_importance.png'))  # 그래프 저장
plt.close()

# 그래프 4: 분포 비교 (Distribution of True vs Predicted NOX)
plt.figure(figsize=(10, 6))
sns.kdeplot(y_train, color='blue', label='True NOX')  # 실제값 분포
sns.kdeplot(y_pred, color='orange', label='Predicted NOX')  # 예측값 분포
plt.legend()  # 범례 추가
plt.title('Distribution of True vs Predicted NOX')
plt.savefig(os.path.join(result_dir, 'distribution_plot.png'))  # 그래프 저장
plt.close()
