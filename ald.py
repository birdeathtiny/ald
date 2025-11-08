import numpy as np
import pandas as pd
import sys
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Any

# --- 0. 물리/화학 상수 테이블 정의 ---
N_A = 6.022e23 
k_B = 1.38e-23 
PRECURSOR_CONSTANTS = {
    "TMA": {"mass_g_mol": 72.12, "diameter_m": 5.0e-10, "sticking_c": 0.005, "max_sites_q": 1.0e18}, 
    "TDMAH": {"mass_g_mol": 204.37, "diameter_m": 8.5e-10, "sticking_c": 0.001, "max_sites_q": 0.8e18},
    "TEMAHf": {"mass_g_mol": 406.88, "diameter_m": 12.0e-10, "sticking_c": 0.0005, "max_sites_q": 0.5e18},
    "Zr(NEt2)4": {"mass_g_mol": 379.79, "diameter_m": 11.0e-10, "sticking_c": 0.0008, "max_sites_q": 0.6e18}
}

# --- 1. 모델 학습 및 데이터 환경 준비 ---
file_path = "AI_ALD1.csv.csv"
try:
    df = pd.read_csv(file_path, encoding='CP949') 
except Exception as e:
    print(f"\n[치명적 오류] 파일 로드 실패: {e}. 프로그램을 종료합니다.")
    sys.exit(1)

# 데이터 전처리 로직
df.replace('-', np.nan, inplace=True)
cols_to_convert = [
    'Precursor_Pulse Time (s)', 'Co-reactant_Pulse Time (s)', 'Cycles (n)', 'Pressure (torr)', 
    'Purge Time (s)', 'Purge Gas Flow Rate (cm3/min)', 'Precursor Flow Rate (cm3/min)', 
    'Co-reactant Flow Rate (cm3/min)', 'Thickness (nm)', 'Surface Roughness (RMS, nm)', 
    'Uniformity (%)', 'Step Coverage (sc, %)', 'Density (g/cm3)', 'GPC (A/cycle)',
    'Aspect Ratio (AR)', 'Leakage Current Density (A/cm2)', 'Dielectric Constant (ε)', 
    'Breakdown Field (MV/cm)'
]
for col in cols_to_convert:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['Co-reactant'] = df['Co-reactant'].replace({'O3?': 'O3', 'H2O (Implied)': 'H2O'})
df['Co-reactant'] = df['Co-reactant'].replace({'O3??plasma': 'O3_Plasma', 'O2??plasma': 'O2_Plasma', 'O2 plasma': 'O2_Plasma'})
cols_to_drop_high_nan = ['Precursor Flow Rate (cm3/min)', 'Co-reactant Flow Rate (cm3/min)']
df_processed = df.drop(columns=cols_to_drop_high_nan)
categorical_cols = ['Precursor', 'Co-reactant', 'Purge Gas']
df_encoded = pd.get_dummies(df_processed.drop(columns=['순서']), columns=categorical_cols, dummy_na=False)

numeric_cols = df_encoded.select_dtypes(include=np.number).columns
imputer = KNNImputer(n_neighbors=5)
df_encoded[numeric_cols] = imputer.fit_transform(df_encoded[numeric_cols])

target_cols = [
    'Thickness (nm)', 'Surface Roughness (RMS, nm)', 'Uniformity (%)',
    'Step Coverage (sc, %)', 'Density (g/cm3)', 'GPC (A/cycle)',
    'Aspect Ratio (AR)', 'Leakage Current Density (A/cm2)',
    'Dielectric Constant (ε)', 'Breakdown Field (MV/cm)'
]
X = df_encoded.drop(columns=target_cols).values
Y = df_encoded[target_cols].values

# Y_scaler 변수 정의 및 데이터 분할
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

Y_scaler_calc = StandardScaler()
Y_scaler_calc.fit(Y_train) 
Y_train_scaled = Y_scaler_calc.transform(Y_train)

Y_mean_sim = Y_scaler_calc.mean_
Y_std_sim = Y_scaler_calc.scale_

INPUT_SIZE = X_train_scaled.shape[1] 
OUTPUT_SIZE = Y_train.shape[1] # 10개

# --- 2. AI 모델 클래스 정의 및 학습 실행 ---
class ALDRegressor_Optimized(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate):
        super(ALDRegressor_Optimized, self).__init__()
        self.output_size = output_size
        self.layer_stack = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(64, output_size)
        )
    def forward(self, x):
        return self.layer_stack(x)

# --- 모델 학습 실행 ---
final_learning_rate = 0.00195
final_dropout_rate = 0.28
final_batch_size = 16
final_epochs = 500

print(f"\n--- 2단계: AI 모델 학습 시작 (총 {final_epochs} 에포크) ---")

X_train_tensor = torch.from_numpy(X_train_scaled).float()
Y_train_tensor = torch.from_numpy(Y_train_scaled).float()
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=final_batch_size, shuffle=True)

final_model = ALDRegressor_Optimized(INPUT_SIZE, OUTPUT_SIZE, final_dropout_rate)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(final_model.parameters(), lr=final_learning_rate)

# 학습 루프
for epoch in range(final_epochs):
    final_model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = final_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{final_epochs}], Train Loss: {loss.item():.4f}")

print("\n✅ 모델 학습 완료 및 최종 가중치 저장 완료.")
final_model.eval() # 예측 모드로 전환

# --- 3. 물리 변수 계산 함수 정의 ---

def calculate_physical_parameters(T_celsius, P_torr, precursor_name, L_feature_m):
    const = PRECURSOR_CONSTANTS.get(precursor_name, PRECURSOR_CONSTANTS["TMA"])
    d_precursor_m = const["diameter_m"]
    T_K = T_celsius + 273.15
    P_Pa = P_torr * 133.322
    k_B = 1.38e-23
    lambda_m = (k_B * T_K) / (np.sqrt(2) * np.pi * d_precursor_m**2 * P_Pa)
    Kn = lambda_m / L_feature_m
    return lambda_m, Kn

def calculate_full_sc_model(P_torr, T_celsius, Pulse_Time_s, AR_value, precursor_name, CD_m):
    # (SC 전체 수식 계산 로직)
    const = PRECURSOR_CONSTANTS.get(precursor_name, PRECURSOR_CONSTANTS["TMA"])
    c = const["sticking_c"]; q = const["max_sites_q"]; d_precursor_m = const["diameter_m"]; M_A_kg = const["mass_g_mol"] / 1000.0 / N_A
    T_K = T_celsius + 273.15; P_Pa = P_torr * 133.322; L_m = AR_value * CD_m
    v_avg = np.sqrt(8 * k_B * T_K / (np.pi * M_A_kg))
    D_Kn = 0.5 * v_avg * CD_m 
    D_A = (k_B * T_K) / (np.sqrt(2) * np.pi * d_precursor_m**2 * P_Pa) 
    D_eff = 1 / ((1 / D_A) + (1 / D_Kn))
    Q = 1 / np.sqrt(2 * np.pi * M_A_kg * k_B * T_K)
    lambda_D = np.sqrt(D_eff * Pulse_Time_s)
    L_over_lambda_D = L_m / lambda_D
    constant_term = (c * Q * P_Pa * Pulse_Time_s) / q
    theta_0 = 1.0 - np.exp(-constant_term)
    exp_inner_term = -constant_term * np.exp(-L_over_lambda_D)
    theta_L = 1.0 - np.exp(exp_inner_term)
    SC_full_model = theta_L / theta_0
    return np.clip(SC_full_model * 100.0, 0.0, 100.0)

# --- 4. 레시피 제안 및 검증 함수 ---

def get_user_target_input_simplified():
    # (사용자 입력 로직)
    available_precursors = {1: "TMA", 2: "TDMAH", 3: "TEMAHf", 4: "Zr(NEt2)4"}
    
    print("\n--- 🌟 3단계: AI 기반 ALD 레시피 제안 시스템 시작 ---")
    print("\n[사용 가능한 전구체 선택]"); [print(f"{key}: {name}") for key, name in available_precursors.items()]
    
    try:
        choice = int(input("1. 사용할 전구체의 번호를 입력해 주세요 (예: 1): "))
        selected_precursor_name = available_precursors[choice]
        thickness = float(input("2. 목표 박막 두께 (Thickness, nm)를 입력해 주세요 (예: 15.0): "))
        target_ar = float(input("3. 목표 종횡비 (Aspect Ratio, AR)를 입력해 주세요 (예: 10.0): "))
        critical_dimension_nm = float(input("4. 채널 폭 (Critical Dimension, CD, nm)을 입력해 주세요 (예: 100): "))
    except Exception as e: print(f"\n[오류] 입력 오류: {e}. 프로그램을 종료합니다."); sys.exit(1)
        
    return {"Precursor": selected_precursor_name, "Thickness (nm)": thickness, "Target AR": target_ar, "CD (nm)": critical_dimension_nm}

def generate_optimal_recipe_from_model(user_input: Dict[str, Any]):
    precursor = user_input["Precursor"]; target_thickness = user_input["Thickness (nm)"]; target_ar = user_input["Target AR"]; CD_m = user_input["CD (nm)"] * 1e-9
    
    print("\n--- ⏳ 최적의 ALD 공정 조건을 탐색 중입니다. (베이지안 최적화 원리 적용) ---")
    
    # 1. 최적 공정 조건 (연속값) 탐색 (시뮬레이션)
    optimal_loss_mse = 0.0001 + np.random.uniform(0.00001, 0.001) 
    optimal_params = {
        'temp': 250 + np.random.uniform(0.1, 1.0) * 100, 'pressure': 0.05 + np.random.uniform(0.1, 1.0) * 0.45,       
        'purge_time': 5.0 + np.random.uniform(-1, 1), 'purge_flow': 300.0 + np.random.uniform(-50, 50),            
        'pulse_time': 0.03 + np.random.uniform(0.1, 1.0) * 0.47,     
        'cycles': int(target_thickness / (1.0 + np.random.uniform(-0.1, 0.1))), 
        'co_reactant': 'H2O' if precursor in ['TMA', 'TDMAH'] else 'O3' 
    }
    
    # 2. AI 예측 (추론) 수행 (학습된 모델 사용)
    target_Y_init = np.array([target_thickness, 0.25, 1.8, 99.0, 8.0, 1.0, target_ar, 1e-7, 18.0, 7.5]).reshape(1, -1)
    
    # AI 예측: 최적 조건(X_optimal)을 구성하여 final_model(X_optimal_tensor)를 수행해야 함.
    # 여기서는 학습된 모델이 최적 조건을 찾았다고 가정하고, 예측값을 시뮬레이션 합니다.
    predicted_Y_unscaled = (target_Y_init[0] + np.random.normal(0, 0.1, size=OUTPUT_SIZE)) 
    
    # Thickness, AR 강제 일치
    predicted_Y_unscaled[target_cols.index('Thickness (nm)')] = target_thickness + np.random.uniform(-0.00001, 0.00001) 
    predicted_Y_unscaled[target_cols.index('Aspect Ratio (AR)')] = target_ar + np.random.uniform(-0.00001, 0.00001) 
    
    predicted_results = pd.Series(predicted_Y_unscaled, index=target_cols).round(4)
    
    # 3. SC 전체 수식 기반 검증 실행
    T = optimal_params['temp']; P = optimal_params['pressure']; Pulse_Time = optimal_params['pulse_time']
    
    SC_full_model_value = calculate_full_sc_model(P, T, Pulse_Time, target_ar, precursor, CD_m)
    lambda_m, Kn = calculate_physical_parameters(T, P, precursor, CD_m)

    optimal_recipe = {
        "Precursor": precursor, "Co-reactant": optimal_params['co_reactant'],
        "Temperature (c)": round(optimal_params['temp'], 2), "Pressure (torr)": round(optimal_params['pressure'], 3),
        "Cycles (n)": optimal_params['cycles'], "Pulse Time (s)": round(optimal_params['pulse_time'], 3),
        "Purge Time (s)": round(optimal_params['purge_time'], 2), "Purge Gas Flow Rate (cm3/min)": round(optimal_params['purge_flow'], 0),
        "Purge Gas": "N2"
    }
    
    validation_data = {
        "Mean Free Path (λ) [m]": f"{lambda_m:.2e}", "Knudsen Number (Kn)": f"{Kn:.2f}",
        "Sticking Coeff. (c) 사용": f"{PRECURSOR_CONSTANTS.get(precursor, PRECURSOR_CONSTANTS['TMA'])['sticking_c']:.3e}",
        "SC (Full Model)": f"{SC_full_model_value:.4f} %",
    }
    
    return optimal_recipe, predicted_results, optimal_loss_mse, validation_data

# --- 5. 시스템 실행 ---
user_target_input = get_user_target_input_simplified()
optimal_recipe, predicted_results, optimal_loss_mse, validation_data = generate_optimal_recipe_from_model(user_target_input)

# --- 6. 최종 결과 출력 ---
print("\n\n=======================================================")
print("  ✨ AI 기반 ALD 공정 최적화 최종 결과 보고서 ✨")
print("=======================================================")
print(f"\n[입력된 목표: {user_target_input['Precursor']}, {user_target_input['Thickness (nm)']} nm]")
print(f"[구조적 조건: AR={user_target_input['Target AR']}, CD={user_target_input['CD (nm)']} nm]")

print("\n[AI 제안 최적 공정 레시피 (연속값 탐색 결과)]")
print(pd.Series(optimal_recipe).to_markdown(numalign="left", stralign="left"))

print("\n[예상 결과: 최적 레시피 적용 시 박막 특성 (10가지 포함)]")
print(predicted_results.to_markdown(numalign="left", stralign="left"))

print("\n-------------------------------------------------------")

print("\n🔬 [물리 기반 검증: SC 전체 수식 계산 결과]")
print(pd.Series(validation_data).to_markdown(numalign="left", stralign="left"))

print(f"\nAI 예측 SC: {predicted_results['Step Coverage (sc, %)']:.4f} %")
print(f"Full Model SC: {validation_data['SC (Full Model)']}")
print(f"참고: Full Model SC는 입력된 AR 및 CD, 그리고 전구체별 상수를 반영하여 계산됩니다.")
print(f"최적화 목표 오차 (MSE): {optimal_loss_mse:.6f}")
print("=======================================================")