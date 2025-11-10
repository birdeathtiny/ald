import numpy as np
import pandas as pd
import sys
import os 
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Any, List, Tuple
from scipy.optimize import minimize

# --- 0. 물리/화학 상수 테이블 정의 ---
N_A = 6.022e23 
k_B = 1.38e-23 

# 💡 [수정 A] SC 저하를 의도적으로 유발 (물리 모델 전용)
PRECURSOR_CONSTANTS = {
    "TMA": {"mass_g_mol": 72.12, "diameter_m": 5.0e-10, "sticking_c": 0.005, "max_sites_q": 1.0e18}, 
    "TDMAH": {"mass_g_mol": 204.37, "diameter_m": 8.5e-10, "sticking_c": 0.001, "max_sites_q": 0.8e18}, 
    "TEMAHf": {"mass_g_mol": 406.88, "diameter_m": 12.0e-10, "sticking_c": 0.005, "max_sites_q": 0.5e18}, 
    "Zr(NEt2)4": {"mass_g_mol": 379.79, "diameter_m": 11.0e-10, "sticking_c": 0.008, "max_sites_q": 0.6e18}
}
# (참고: 위 값은 물리적 정확성보다 최적화 챌린지를 위해 수정된 값임)


# --- 1. 모델 학습 및 데이터 환경 준비 ---
file_path = "AI_ALD1.csv.csv"
try:
    df = pd.read_csv(file_path, encoding='CP949') 
except Exception as e:
    print(f"\n[치명적 오류] 파일 로드 실패: {e}. 프로그램을 종료합니다.")
    sys.exit(1)

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

# 💡 [논리적 수정 1] AI 예측 대상 (8개)
target_cols = [
    'Thickness (nm)', 'Surface Roughness (RMS, nm)', 'Uniformity (%)',
    'Density (g/cm3)', 'GPC (A/cycle)',
    'Leakage Current Density (A/cm2)', 'Dielectric Constant (ε)', 
    'Breakdown Field (MV/cm)'
]

# 💡 [오류 수정] AI가 무시할 컬럼 (물리 모델 전용 또는 사용자 입력값)
cols_to_ignore_for_ai = [
    'Step Coverage (sc, %)', # 물리 모델이 전담
    'Aspect Ratio (AR)'      # 사용자 입력값이므로 AI의 입/출력이 아님
]

try:
    # 💡 [오류 수정] AI 입력 = (전체) - (예측 대상 8개) - (무시 대상 2개)
    # 이렇게 하면 AR이 입력 피처에서 제외됩니다. (예: 35개 피처)
    ALL_INPUT_FEATURES_ORDERED = df_encoded.drop(
        columns=target_cols + cols_to_ignore_for_ai
    ).columns.tolist()
    
    ALL_OUTPUT_FEATURES_ORDERED = target_cols

except KeyError:
    print("\n[치명적 오류] target_cols 또는 cols_to_ignore_for_ai에 CSV에 없는 컬럼명이 포함되어 있습니다.")
    sys.exit(1)

# (데이터 누수 방지 로직 - 이전과 동일)
X = df_encoded[ALL_INPUT_FEATURES_ORDERED].values
Y = df_encoded[ALL_OUTPUT_FEATURES_ORDERED].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

X_imputer = KNNImputer(n_neighbors=5)
X_train = X_imputer.fit_transform(X_train)
X_test = X_imputer.transform(X_test)

Y_imputer = KNNImputer(n_neighbors=5)
Y_train = Y_imputer.fit_transform(Y_train)
Y_test = Y_imputer.transform(Y_test)

X_scaler = StandardScaler()
Y_scaler = StandardScaler()

# 💡 [오류 수정] X_scaler는 이제 AR, SC가 제외된 피처(예: 35개)로 fit 됩니다.
X_train_scaled = X_scaler.fit_transform(X_train) 
X_test_scaled = X_scaler.transform(X_test)
Y_train_scaled = Y_scaler.fit_transform(Y_train)
Y_test_scaled = Y_scaler.transform(Y_test)

INPUT_SIZE = X_train_scaled.shape[1] 
OUTPUT_SIZE = Y_train.shape[1] 

print(f"\n--- 1단계: 데이터 준비 완료 ---")
print(f"AI 모델 입력 피처 수: {INPUT_SIZE} (AR, SC 제외)")
print(f"AI 모델 출력 피처 수: {OUTPUT_SIZE} (AR, SC 제외)")


# --- 2. AI 모델 클래스 정의 및 학습 실행 ---
class ALDRegressor_Optimized(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate):
        super(ALDRegressor_Optimized, self).__init__()
        self.output_size = output_size
        self.layer_stack = nn.Sequential(
            nn.Linear(input_size, 128), 
            nn.ReLU(), 
            nn.Dropout(dropout_rate), 
            nn.Linear(128, 64), 
            nn.ReLU(), 
            nn.Dropout(dropout_rate), 
            nn.Linear(64, output_size) # 8개 출력
        )
    def forward(self, x):
        return self.layer_stack(x)

# (하이퍼파라미터 및 조기 종료 설정 - 이전과 동일)
final_learning_rate = 0.00195
final_dropout_rate = 0.28
final_batch_size = 16
final_epochs = 500
VALIDATION_SPLIT = 0.2 
PATIENCE = 30 
WEIGHT_DECAY = 1e-5 
BEST_MODEL_PATH = 'best_ald_model.pth' 

print(f"\n--- 2단계: AI 모델 학습 시작 (최대 {final_epochs} 에포크, 조기 종료 적용) ---")
print(f"AI가 예측할 물성: {ALL_OUTPUT_FEATURES_ORDERED}")

X_train_tensor = torch.from_numpy(X_train_scaled).float()
Y_train_tensor = torch.from_numpy(Y_train_scaled).float()
X_test_tensor = torch.from_numpy(X_test_scaled).float()
Y_test_tensor = torch.from_numpy(Y_test_scaled).float()

X_train_final, X_val, Y_train_final, Y_val = train_test_split(
    X_train_tensor, Y_train_tensor, test_size=VALIDATION_SPLIT, random_state=42
)

train_dataset = TensorDataset(X_train_final, Y_train_final)
val_dataset = TensorDataset(X_val, Y_val)

train_loader = DataLoader(train_dataset, batch_size=final_batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=final_batch_size, shuffle=False)

final_model = ALDRegressor_Optimized(INPUT_SIZE, OUTPUT_SIZE, final_dropout_rate)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    final_model.parameters(), 
    lr=final_learning_rate, 
    weight_decay=WEIGHT_DECAY 
)

# (조기 종료 로직 - 이전과 동일)
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(final_epochs):
    final_model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = final_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    
    final_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = final_model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{final_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(final_model.state_dict(), BEST_MODEL_PATH) 
        patience_counter = 0
    else:
        patience_counter += 1
        
    if patience_counter >= PATIENCE:
        print(f"\n[조기 종료] {PATIENCE} 에포크 동안 검증 손실이 개선되지 않아 Epoch {epoch+1}에서 학습을 중단합니다.")
        break

print(f"\n✅ 모델 학습 완료. 최고 성능 모델(Val Loss: {best_val_loss:.4f})을 불러옵니다.")
final_model.load_state_dict(torch.load(BEST_MODEL_PATH))
final_model.eval()

with torch.no_grad():
    Y_test_pred = final_model(X_test_tensor)
    test_loss = criterion(Y_test_pred, Y_test_tensor)
print(f"--- 🚀 최종 모델 테스트셋 MSE (8개 물성): {test_loss.item():.6f} ---")


# --- 3. 물리 변수 계산 함수 정의 (SC 전담) ---
# (이전과 동일)
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
    const = PRECURSOR_CONSTANTS.get(precursor_name, PRECURSOR_CONSTANTS["TMA"])
    
    c = const["sticking_c"]; q = const["max_sites_q"]; d_precursor_m = const["diameter_m"]; M_A_kg = const["mass_g_mol"] / 1000.0 / N_A
    T_K = T_celsius + 273.15; P_Pa = P_torr * 133.322; L_m = AR_value * CD_m
    
    v_avg = np.sqrt(8 * k_B * T_K / (np.pi * M_A_kg))
    lambda_m = (k_B * T_K) / (np.sqrt(2) * np.pi * d_precursor_m**2 * P_Pa)
    D_A = (1/3) * lambda_m * v_avg
    D_Kn = (1/3) * v_avg * CD_m 
    D_eff = 1 / ((1 / D_A) + (1 / D_Kn))
    
    Q = 1 / np.sqrt(2 * np.pi * M_A_kg * k_B * T_K)
    lambda_D = np.sqrt(D_eff * Pulse_Time_s)
    L_over_lambda_D = L_m / (lambda_D + 1e-12)
    
    constant_term = (c * Q * P_Pa * Pulse_Time_s) / q
    theta_0 = 1.0 - np.exp(-constant_term)
    
    exp_inner_term = -constant_term * np.exp(-L_over_lambda_D)
    theta_L = 1.0 - np.exp(exp_inner_term)
    
    SC_full_model = theta_L / (theta_0 + 1e-12)
    return np.clip(SC_full_model * 100.0, 0.0, 100.0)


# --- 4. AI 모델 입력을 위한 '번역기' 함수 ---
# 💡 [오류 수정] target_ar 인자 제거
def create_model_input(
    recipe_params: Dict[str, Any], 
    precursor_name: str, 
    co_reactant_name: str, 
    purge_gas_name: str
) -> pd.DataFrame:
    # 💡 [오류 수정] ALL_INPUT_FEATURES_ORDERED (예: 35개)
    input_df = pd.DataFrame(columns=ALL_INPUT_FEATURES_ORDERED)
    input_df.loc[0] = 0.0
    
    for key, value in recipe_params.items():
        if key in input_df.columns:
            input_df.at[0, key] = value
            
    # 💡 [오류 수정] AR 입력 로직 완전 삭제
            
    precursor_col = f"Precursor_{precursor_name}"
    if precursor_col in input_df.columns:
        input_df.at[0, precursor_col] = 1.0
        
    coreactant_col = f"Co-reactant_{co_reactant_name}"
    if coreactant_col in input_df.columns:
        input_df.at[0, coreactant_col] = 1.0

    purge_gas_col = f"Purge Gas_{purge_gas_name}"
    if purge_gas_col in input_df.columns:
        input_df.at[0, purge_gas_col] = 1.0
        
    return input_df


# --- 5. '레시피 -> AI 예측' 수행 함수 ---
# 💡 [오류 수정] target_ar 인자 제거
def predict_from_recipe(
    recipe_params: Dict[str, Any],
    precursor_name: str,
    co_reactant_name: str,
    purge_gas_name: str
) -> pd.Series:
    # 💡 [오류 수정] target_ar 인자 제거
    input_df = create_model_input(recipe_params, precursor_name, co_reactant_name, purge_gas_name)
    
    # 💡 [오류 수정] X_scaler는 35개 피처를 transform 합니다.
    X_scaled = X_scaler.transform(input_df.values)
    X_tensor = torch.from_numpy(X_scaled).float()
    
    final_model.eval()
    with torch.no_grad():
        Y_pred_scaled_tensor = final_model(X_tensor)
        
    Y_pred_unscaled = Y_scaler.inverse_transform(Y_pred_scaled_tensor.numpy())[0]
    
    predicted_results = pd.Series(Y_pred_unscaled, index=ALL_OUTPUT_FEATURES_ORDERED).round(4)
    return predicted_results


# --- 5.5. 최적화를 위한 '제약 조건 함수' (동적 목표치 적용) ---
# (이전과 동일 - SC는 물리 모델이 전담)
def constraint_step_coverage(
    x: np.ndarray, 
    user_input: Dict[str, Any], 
    co_reactant_name: str, 
    purge_gas_name: str,
    cost_weights: Dict[str, float] 
) -> float:
    
    target_ar = user_input["Target AR"] # 사용자가 입력한 AR
    if target_ar <= 5:
        TARGET_SC_MIN = 98.0 
    elif target_ar <= 15:
        TARGET_SC_MIN = 90.0 
    else:
        TARGET_SC_MIN = 85.0 
    
    T_celsius = x[0]
    P_torr = x[1]
    Pulse_Time_s = x[2]
    
    # SC 계산은 물리 모델이 전담 (AI와 무관)
    phys_sc = calculate_full_sc_model(
        P_torr=P_torr,
        T_celsius=T_celsius,
        Pulse_Time_s=Pulse_Time_s,
        AR_value=user_input["Target AR"], # 사용자가 입력한 AR
        precursor_name=user_input["Precursor"],
        CD_m=user_input["CD (nm)"] * 1e-9
    )
    
    return phys_sc - TARGET_SC_MIN


# --- 6. 최적화를 위한 '목적 함수 (Objective Function)' ---
COST_WEIGHTS = {
    "thickness": 100.0, 
    "gpc": 30.0,
    "roughness": 10.0
}

def objective_function(
    x: np.ndarray, 
    user_input: Dict[str, Any], 
    co_reactant_name: str, 
    purge_gas_name: str,
    cost_weights: Dict[str, float] 
) -> float:
    
    # 1. 최적화 변수(x)를 레시피(dict)로 매핑
    recipe_params = {
        "Temperature (c)": x[0],
        "Pressure (torr)": x[1],
        "Precursor_Pulse Time (s)": x[2],
        "Purge Time (s)": x[3],
        "Purge Gas Flow Rate (cm3/min)": x[4],
        "Cycles (n)": x[5],
        "Co-reactant_Pulse Time (s)": x[2] 
    }
    
    # 2. 레시피로 AI 예측 수행
    try:
        # 💡 [오류 수정] target_ar 인자 제거
        predicted_results = predict_from_recipe(
            recipe_params, 
            user_input["Precursor"], 
            co_reactant_name, 
            purge_gas_name
        )
    except Exception as e:
        return 1e9 
        
    # 3. 목표값 정의
    target_thickness = user_input["Thickness (nm)"]
    target_gpc_ideal = (target_thickness * 10) / (x[5] + 1e-6) 
    
    # 4. 오차 (Cost) 계산
    w_thickness = cost_weights.get("thickness", 1.0)
    w_gpc = cost_weights.get("gpc", 1.0)
    w_roughness = cost_weights.get("roughness", 1.0)

    pred_thickness = predicted_results.get('Thickness (nm)', target_thickness)
    pred_gpc = predicted_results.get('GPC (A/cycle)', 0)
    pred_roughness = predicted_results.get('Surface Roughness (RMS, nm)', 10)

    cost_thickness = ((pred_thickness - target_thickness) / target_thickness)**2
    cost_gpc = ((pred_gpc - target_gpc_ideal) / target_gpc_ideal)**2
    cost_roughness = (pred_roughness / 5.0)**2

    total_cost = (
        w_thickness * cost_thickness +
        w_gpc * cost_gpc +
        w_roughness * cost_roughness
    )
    
    return total_cost


# --- 7. 레시피 제안 및 검증 함수 (실제 최적화 수행) ---

def get_user_target_input_simplified():
    # (이전과 동일)
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
    
    print("\n--- ⏳ 최적의 ALD 공정 조건을 탐색 중입니다. (SLSQP, AR 동적 SC 제약) ---")
    
    co_reactant = 'H2O' if precursor in ['TMA', 'TDMAH'] else 'O3'
    purge_gas = "N2" 
    
    # (Bounds, Initial Guess - 이전과 동일)
    bounds = [
        (150, 400),     # Temperature (c)
        (0.01, 1.0),    # Pressure (torr)
        (0.05, 2.0),    # Precursor_Pulse Time (s)
        (1.0, 10.0),    # Purge Time (s)
        (50, 500),      # Purge Gas Flow Rate (cm3/min)
        (max(10, int(target_thickness / 2.0)), int(target_thickness / 0.3)) # Cycles (n)
    ]
    initial_guess = [
        300, 0.5, 0.1, 5.0, 300, int(target_thickness / 1.0)
    ]
    
    args = (user_input, co_reactant, purge_gas, COST_WEIGHTS)
    
    constraints = ({
        'type': 'ineq',
        'fun': constraint_step_coverage, # 물리 모델 SC 제약조건
        'args': args
    })
    
    result = minimize(
        objective_function, # AI 예측 기반 비용 함수
        initial_guess,
        args=args,
        method='SLSQP', 
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 100, 'eps': 1e-6}
    )
    
    if not result.success:
        print(f"\n[경고] 최적화가 수렴에 실패했습니다 (혹은 제약조건 위반): {result.message}")
        
    optimal_x = result.x
    
    optimal_recipe_params = {
        "Temperature (c)": optimal_x[0],
        "Pressure (torr)": optimal_x[1],
        "Precursor_Pulse Time (s)": optimal_x[2],
        "Purge Time (s)": optimal_x[3],
        "Purge Gas Flow Rate (cm3/min)": optimal_x[4],
        "Cycles (n)": optimal_x[5],
        "Co-reactant_Pulse Time (s)": optimal_x[2]
    }
    
    # 💡 [오류 수정] 최종 예측 시 target_ar 인자 제거
    predicted_results = predict_from_recipe(
        optimal_recipe_params,
        precursor,
        co_reactant,
        purge_gas
    )
    
    T = optimal_x[0]; P = optimal_x[1]; Pulse_Time = optimal_x[2]
    # SC 검증 (물리 모델이 유일한 SC 소스)
    SC_full_model_value = calculate_full_sc_model(P, T, Pulse_Time, target_ar, precursor, CD_m)
    lambda_m, Kn = calculate_physical_parameters(T, P, precursor, CD_m)

    optimal_recipe_report = {
        "Precursor": precursor, "Co-reactant": co_reactant,
        "Temperature (c)": round(T, 2), "Pressure (torr)": round(P, 3),
        "Cycles (n)": int(optimal_x[5]), 
        "Precursor Pulse Time (s)": round(Pulse_Time, 3),
        "Co-reactant Pulse Time (s)": round(Pulse_Time, 3),
        "Purge Time (s)": round(optimal_x[3], 2), 
        "Purge Gas Flow Rate (cm3/min)": round(optimal_x[4], 0),
        "Purge Gas": purge_gas
    }
    
    validation_data = {
        "Mean Free Path (λ) [m]": f"{lambda_m:.2e}", "Knudsen Number (Kn)": f"{Kn:.2f}",
        "Sticking Coeff. (c) 사용 ": f"{PRECURSOR_CONSTANTS.get(precursor, PRECURSOR_CONSTANTS['TMA'])['sticking_c']:.3e}",
        "SC (Full Model)": f"{SC_full_model_value:.4f} %",
    }
    
    optimization_stats = {
        "Optimization Success": result.success,
        "Optimization Message": result.message,
        "Function Evaluations (nfev)": result.nfev,
        "Iterations (nit)": result.nit,
        "Final Cost (fun)": f"{result.fun:.6f}"
    }
    
    return optimal_recipe_report, predicted_results, validation_data, optimization_stats

# --- 8. 시스템 실행 ---
user_target_input = get_user_target_input_simplified()
optimal_recipe, predicted_results, validation_data, optimization_stats = generate_optimal_recipe_from_model(user_target_input)

# --- 9. 최종 결과 출력 ---
print("\n\n=======================================================")
print("  ✨ AI 기반 ALD 공정 최적화 최종 결과 보고서 ✨")
print("=======================================================")
print(f"\n[입력된 목표: {user_target_input['Precursor']}, {user_target_input['Thickness (nm)']} nm]")
print(f"[구조적 조건: AR={user_target_input['Target AR']}, CD={user_target_input['CD (nm)']} nm]")

print("\n[AI 제안 최적 공정 레시피 (SLSQP 탐색 결과)]")
print(pd.Series(optimal_recipe).to_markdown(numalign="left", stralign="left"))

# 💡 [논리적 수정] AI 예측 결과 (8가지, SC/AR 제외)
print("\n[AI 예측 결과: 최적 레시피 적용 시 박막 특성 (8가지)]")
print(predicted_results.to_markdown(numalign="left", stralign="left"))

print("\n-------------------------------------------------------")

print("\n🔬 [물리 기반 검증: SC 전체 수식 계산 결과]")
print(pd.Series(validation_data).to_markdown(numalign="left", stralign="left"))
print(f"\n물리 모델 SC: {validation_data['SC (Full Model)']} (최적화 제약조건으로 사용됨, AR 기반 동적 목표)")
print(f"참고: AI는 SC/AR을 예측하지 않으며, 물리 모델이 유일한 SC 계산 주체입니다.")

print("\n-------------------------------------------------------")

print("\n📈 [최적화(SLSQP) 수렴 리포트]")
print(pd.Series(optimization_stats).to_markdown(numalign="left", stralign="left"))
print(f"최적화 목표 오차 (Cost): {optimization_stats['Final Cost (fun)']} (Thickness, GPC, Roughness 기준)")
print("=======================================================")

# 임시 모델 파일 삭제
if os.path.exists(BEST_MODEL_PATH):
    os.remove(BEST_MODEL_PATH)
    print(f"\n[정리] 임시 모델 파일 ({BEST_MODEL_PATH})이 삭제되었습니다.")