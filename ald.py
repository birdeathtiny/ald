import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.exceptions import NotFittedError
import joblib
import os  
from scipy.optimize import minimize
import random
import sys

# ==========================================
# 0. 환경 설정 및 장치 확인
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==========================================
# 1. 데이터 불러오기 및 전처리 (새 파일 및 컬럼명 적용)
# ==========================================
try:
    # 1-1. CSV 파일 읽기 (파일 경로 문제 해결)
    file_name = 'AI_ALD1.csv.csv'
    
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd() 
        print("Warning: '__file__' not defined. Using current working directory.")

    file_path = os.path.join(script_dir, file_name)
    print(f"Attempting to load file from: {file_path}") 
    
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #
    # 💡 [수정] : 인코딩을 'utf-8'에서 'cp949' (한글 Excel)로 변경
    df = pd.read_csv(file_path, encoding='cp949')
    #
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # 컬럼명 앞/뒤 공백 및 따옴표 제거 (KeyError 방지)
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace('"', '', regex=False) 

except FileNotFoundError:
    print(f"Error: '{file_name}' 파일을 찾을 수 없습니다.")
    print(f"스크립트와 동일한 폴더 ({script_dir})에 파일이 있는지 확인해주세요.")
    sys.exit()
except UnicodeDecodeError:
    # 혹시 'cp949'도 실패하면 'euc-kr'을 시도하라는 안내
    print(f"Error: 'cp949' 인코딩으로 파일을 읽지 못했습니다.")
    print(f"파일을 메모장으로 열어 '다른 이름으로 저장' -> 인코딩을 'UTF-8'로 선택한 뒤 다시 시도해보세요.")
    sys.exit()
except Exception as e:
    print(f"An error occurred while loading the file: {e}")
    sys.exit()

print(f"클리닝 후 컬럼 목록: {df.columns.to_list()}")

# 1-2. 데이터 클리닝 (수치형 변환) - 새 CSV 컬럼명 기준
numeric_cols = [
    'Precursor_Pulse Time (s)', 'Co-reactant_Pulse Time (s)', 'Cycles (n)',
    'Temperature (c)', 
    'Pressure (torr)', 'Purge Time (s)',
    'Purge Gas Flow Rate (cm3/min)', 
    'Precursor Flow Rate (cm3/min)',
    'Co-reactant Flow Rate (cm3/min)',
    'Thickness (nm)',
    'GPC (A/cycle)'
]

# 1-3. '-'를 NaN으로 변경 및 수치형 변환
print("\n데이터 클리닝 수행 중...")
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].replace('-', np.nan), errors='coerce')
    else:
        print(f"Warning: '{col}' 컬럼이 numeric_cols 목록에 있지만 CSV에 존재하지 않습니다.")

# 1-4. 사용할 컬럼만 선택
categorical_cols = ['Precursor']
process_cols = [col for col in numeric_cols if col not in ['Thickness (nm)', 'GPC (A/cycle)']]
output_cols = ['GPC (A/cycle)', 'Thickness (nm)']

all_used_cols = list(set(categorical_cols + process_cols + output_cols).intersection(df.columns))
df_clean = df[all_used_cols].copy()

# 1-5. 결측치 처리
df_clean = df_clean.dropna(subset=output_cols) 
df_clean[process_cols] = df_clean[process_cols].fillna(0)
df_clean['Precursor'] = df_clean['Precursor'].fillna('Unknown')

if df_clean.empty:
    print("오류: GPC 또는 Thickness 데이터가 있는 유효한 행이 없습니다. CSV 파일을 확인해주세요.")
    sys.exit()

print(f"클리닝 및 전처리 완료. 학습 데이터 {len(df_clean)}개 확보.")

# 1-6. 범주형(Precursor) 원-핫 인코딩
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
precursor_encoded = encoder.fit_transform(df_clean[['Precursor']])
precursor_columns = encoder.get_feature_names_out(['Precursor'])
precursor_df = pd.DataFrame(precursor_encoded, columns=precursor_columns, index=df_clean.index)

precursor_map = {chr(97 + i): name.replace('Precursor_', '') for i, name in enumerate(encoder.categories_[0])}
print(f"Precursor Mapping: {precursor_map}")

# 1-7. 최종 데이터프레임 결합
df_features = df_clean[process_cols].join(precursor_df)
df_outputs = df_clean[output_cols]

# 1-8. 입력(X), 출력(y1=GPC, y2=Thickness) 분리
X = df_features.values
y_gpc = df_outputs['GPC (A/cycle)'].values.reshape(-1, 1)
y_thick = df_outputs['Thickness (nm)'].values.reshape(-1, 1)

# 1-9. 스케일링
scaler_X = MinMaxScaler()
scaler_y1 = MinMaxScaler() # GPC
scaler_y2 = MinMaxScaler() # Thickness

X_scaled = scaler_X.fit_transform(X)
y_gpc_scaled = scaler_y1.fit_transform(y_gpc)
y_thick_scaled = scaler_y2.fit_transform(y_thick)

# 1-10. 텐서 변환 및 장치 이동
X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
y1_tensor = torch.tensor(y_gpc_scaled, dtype=torch.float32).to(device)
y2_tensor = torch.tensor(y_thick_scaled, dtype=torch.float32).to(device)

print(f"입력(X) 텐서 크기: {X_tensor.shape}")
print(f"GPC(Y1) 텐서 크기: {y1_tensor.shape}")
print(f"Thickness(Y2) 텐서 크기: {y2_tensor.shape}")

# ==========================================
# 2. 데이터셋 정의 및 분할
# ==========================================
dataset_gpc = TensorDataset(X_tensor, y1_tensor)
dataset_thick = TensorDataset(X_tensor, y2_tensor)

train_size = int(0.8 * len(dataset_gpc))
test_size = len(dataset_gpc) - train_size

if train_size == 0 or test_size == 0:
    print("오류: 데이터셋이 너무 작아 train/test로 분할할 수 없습니다. (최소 2개 이상의 데이터 필요)")
    sys.exit()

train_gpc, test_gpc = random_split(dataset_gpc, [train_size, test_size])
train_thick, test_thick = random_split(dataset_thick, [train_size, test_size])

BATCH_SIZE = max(1, train_size // 4) 
print(f"Batch Size: {BATCH_SIZE}")

train_loader_gpc = DataLoader(train_gpc, batch_size=BATCH_SIZE, shuffle=True)
test_loader_gpc = DataLoader(test_gpc, batch_size=BATCH_SIZE)
train_loader_thick = DataLoader(train_thick, batch_size=BATCH_SIZE, shuffle=True)
test_loader_thick = DataLoader(test_thick, batch_size=BATCH_SIZE)

# ==========================================
# 3. 모델 정의
# ==========================================
class GPCModel(nn.Module):
    def __init__(self, input_dim):
        super(GPCModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

class ThicknessModel(nn.Module):
    def __init__(self, input_dim):
        super(ThicknessModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

# ==========================================
# 4 & 5. 학습 및 평가 함수
# ==========================================
def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 50 == 0 or epoch == epochs - 1:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.6f}")

def evaluate_model(model, test_loader, scaler_y, model_name):
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            preds.append(outputs.cpu().numpy())
            actuals.append(y_batch.cpu().numpy())
            
    if not preds:
        print(f"평가 중 오류: {model_name} 예측값이 없습니다.")
        return
        
    preds = np.vstack(preds)
    actuals = np.vstack(actuals)
    
    try:
        preds_rescaled = scaler_y.inverse_transform(preds)
        actuals_rescaled = scaler_y.inverse_transform(actuals)
        mse = np.mean((preds_rescaled - actuals_rescaled)**2)
        print(f"{model_name} Test MSE: {mse:.4f}")
    except NotFittedError:
        print("오류: 스케일러가 fit되지 않아 역변환할 수 없습니다.")
    except Exception as e:
        print(f"평가 중 오류 발생: {e}")

# 하이퍼파라미터 설정
GPC_EPOCHS = 300
THICKNESS_EPOCHS = 300
LEARNING_RATE = 0.001

input_dim = X_tensor.shape[1]
criterion = nn.MSELoss()

# GPC 모델 학습
model_gpc = GPCModel(input_dim).to(device)
optimizer_gpc = torch.optim.Adam(model_gpc.parameters(), lr=LEARNING_RATE)
print("\n===== Training GPC Model =====")
train_model(model_gpc, train_loader_gpc, criterion, optimizer_gpc, GPC_EPOCHS)

# Thickness 모델 학습
model_thick = ThicknessModel(input_dim).to(device)
optimizer_thick = torch.optim.Adam(model_thick.parameters(), lr=LEARNING_RATE)
print("\n===== Training Film Thickness Model =====")
train_model(model_thick, train_loader_thick, criterion, optimizer_thick, THICKNESS_EPOCHS)

print("\n===== Evaluating Models =====")
evaluate_model(model_gpc, test_loader_gpc, scaler_y1, "GPC")
evaluate_model(model_thick, test_loader_thick, scaler_y2, "Film Thickness")

# ==========================================
# 6. 모델 및 스케일러 저장
# ==========================================
print("\n===== Saving Models and Scalers =====")
save_dir = script_dir 
torch.save(model_gpc.state_dict(), os.path.join(save_dir, 'ald_gpc_model.pth'))
torch.save(model_thick.state_dict(), os.path.join(save_dir, 'ald_thick_model.pth'))
joblib.dump(scaler_X, os.path.join(save_dir, 'scaler_X.pkl'))
joblib.dump(scaler_y1, os.path.join(save_dir, 'scaler_y1_gpc.pkl'))
joblib.dump(scaler_y2, os.path.join(save_dir, 'scaler_y2_thick.pkl'))
joblib.dump(encoder, os.path.join(save_dir, 'encoder_precursor.pkl'))
print(f"All artifacts saved to {save_dir}")

# ==========================================
# 7. 레시피 검색 최적화 함수 (전구체 고정)
# ==========================================
process_cols_names = df_features.columns.to_list()[:len(process_cols)]
process_cols_count = len(process_cols_names)

def objective_function(X_process_scaled, target_thickness, target_gpc, precursor_one_hot_vector):
    X_full_scaled = np.concatenate([X_process_scaled, precursor_one_hot_vector])
    X_tensor = torch.tensor(X_full_scaled.reshape(1, -1), dtype=torch.float32).to(device)
    
    model_thick.eval()
    model_gpc.eval()
    with torch.no_grad():
        pred_thick_scaled = model_thick(X_tensor).cpu().numpy()
        pred_gpc_scaled = model_gpc(X_tensor).cpu().numpy()

    pred_thick = scaler_y2.inverse_transform(pred_thick_scaled)[0, 0]
    pred_gpc = scaler_y1.inverse_transform(pred_gpc_scaled)[0, 0]
    
    thickness_error = (pred_thick - target_thickness)**2
    gpc_error = (pred_gpc - target_gpc)**2
    
    cost = thickness_error * 1.0 + gpc_error * 1.5 
    
    return cost

def find_best_recipe(target_thickness, target_gpc, selected_precursor_code, n_runs=10):
    
    selected_precursor_name = precursor_map.get(selected_precursor_code)
    if not selected_precursor_name:
        print(f"Error: Invalid precursor code '{selected_precursor_code}'.")
        return

    try:
        precursor_one_hot = encoder.transform([[selected_precursor_name]])[0]
    except NotFittedError:
        print("Error: Encoder is not fitted. Cannot transform precursor name.")
        return

    print(f"\n--- Searching for Optimal Recipe for {selected_precursor_name} (Target: T={target_thickness:.2f}nm, G={target_gpc:.2f}Å/cycle) ---")
    
    bounds_scaled = [(0, 1) for _ in range(process_cols_count)]
    
    best_cost = float('inf')
    best_recipe_scaled_process = None
    
    for run in range(n_runs):
        initial_guess_scaled_process = np.random.rand(process_cols_count)
        
        optimization_result = minimize(
            fun=lambda X_scaled_process: objective_function(X_scaled_process, target_thickness, target_gpc, precursor_one_hot),
            x0=initial_guess_scaled_process,
            bounds=bounds_scaled,
            method='L-BFGS-B', 
            options={'maxiter': 500}
        )
        
        current_cost = optimization_result.fun
        
        if optimization_result.success and current_cost < best_cost:
            best_cost = current_cost
            best_recipe_scaled_process = optimization_result.x
        
        if n_runs <= 20:
             print(f"Run {run+1}/{n_runs}: Cost={current_cost:.4f}, Success={optimization_result.success}")

    if best_recipe_scaled_process is None:
        print("Optimization failed to find a valid solution.")
        return

    # --- 최종 최적 레시피 해석 ---
    best_X_scaled_full = np.concatenate([best_recipe_scaled_process, precursor_one_hot])
    
    try:
        best_recipe_full = scaler_X.inverse_transform(best_X_scaled_full.reshape(1, -1))
    except NotFittedError:
        print("Error: Scaler_X is not fitted. Cannot inverse_transform recipe.")
        return

    pred_X_tensor = torch.tensor(best_X_scaled_full.reshape(1, -1), dtype=torch.float32).to(device)
    with torch.no_grad():
        try:
            pred_thick = scaler_y2.inverse_transform(model_thick(pred_X_tensor).cpu().numpy())[0, 0]
            pred_gpc = scaler_y1.inverse_transform(model_gpc(pred_X_tensor).cpu().numpy())[0, 0]
        except NotFittedError:
            print("Error: Y-Scalers are not fitted. Cannot inverse_transform predictions.")
            return

    recipe_df = pd.DataFrame(best_recipe_full[:, :process_cols_count], columns=process_cols_names)

    print("\n--- Final Optimized ALD Recipe (Process Conditions) ---")
    print(f"Selected Precursor: {selected_precursor_name}")
    print("-" * 30)
    print(f"Predicted Thickness: {pred_thick:.2f} nm (Target: {target_thickness:.2f} nm)")
    print(f"Predicted GPC: {pred_gpc:.2f} Å/cycle (Target: {target_gpc:.2f} Å/cycle)")
    print("=" * 30)
    print("Optimized Process Conditions:")
    for col in recipe_df.columns:
        print(f"  - {col}: {np.round(recipe_df[col].values[0], 3)}")

# ==========================================
# 8. 사용자 입력 및 레시피 검색 실행
# ==========================================
print("\n" + "="*50)
print("AI ALD Recipe Suggestion System Ready!")
print("="*50)

precursor_options = ", ".join([f"{code} ({name})" for code, name in precursor_map.items()])
print(f"Please select the precursor code: {precursor_options}")

try:
    user_precursor_code = input("Enter Precursor Code (e.g., a): ").lower().strip()
    if user_precursor_code not in precursor_map:
        raise ValueError("Invalid precursor code.")
    
    user_target_thickness = float(input("Enter Target Film Thickness (nm): "))
    user_target_gpc = float(input("Enter Target Growth Per Cycle (Å/cycle): "))
except ValueError as e:
    print(f"Invalid input: {e}")
    sys.exit()
except EOFError:
    print("\nInput cancelled. Exiting.")
    sys.exit()

# 최적 레시피 검색 실행
find_best_recipe(user_target_thickness, user_target_gpc, user_precursor_code, n_runs=100)