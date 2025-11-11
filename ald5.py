import streamlit as st
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
import joblib # 👈 자산 저장을 위해 필요

# --- 0. 물리/화학 상수 테이블 정의 ---
N_A = 6.022e23
k_B = 1.38e-23
PRECURSOR_CONSTANTS = {
    "TMA": {"mass_g_mol": 72.12, "diameter_m": 5.0e-10, "sticking_c": 0.005, "max_sites_q": 1.0e18},
    "TDMAH": {"mass_g_mol": 204.37, "diameter_m": 8.5e-10, "sticking_c": 0.001, "max_sites_q": 0.8e18},
    "TEMAHf": {"mass_g_mol": 406.88, "diameter_m": 12.0e-10, "sticking_c": 0.005, "max_sites_q": 0.5e18},
    "Zr(NEt2)4": {"mass_g_mol": 379.79, "diameter_m": 11.0e-10, "sticking_c": 0.008, "max_sites_q": 0.6e18}
}

# --- AI 모델 클래스 정의 ---
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
            nn.Linear(64, output_size)
        )
    def forward(self, x):
        return self.layer_stack(x)

# -----------------------------------------------------------------
# ✨ [핵심 로직] 학습 또는 로드를 수행하는 캐시 함수
# -----------------------------------------------------------------
@st.cache_resource # 👈 이 앱 세션에서 딱 한 번만 실행됨
def load_or_train_model():
    """
    앱이 시작될 때 모델과 자산을 로드합니다.
    파일이 없으면 그 자리에서 학습(Training)을 실행합니다.
    """
    MODEL_PATH = 'best_ald_model.pth'
    ASSETS_PATH = 'ald_assets.joblib'

    # --- A. 파일이 이미 있는 경우 (학습 건너뛰기) ---
    if os.path.exists(MODEL_PATH) and os.path.exists(ASSETS_PATH):
        st.info(f"'{MODEL_PATH}'와 '{ASSETS_PATH}'에서 기존 자산을 로드합니다... (학습 건너뜀)")
        
        # 1. 스케일러 등 자산 로드
        assets_data = joblib.load(ASSETS_PATH)
        
        # 2. AI 모델 로드
        try:
            model = ALDRegressor_Optimized(
                input_size=len(assets_data["ALL_INPUT_FEATURES_ORDERED"]),
                output_size=len(assets_data["ALL_OUTPUT_FEATURES_ORDERED"]),
                dropout_rate=0.28
            )
            model.load_state_dict(torch.load(MODEL_PATH))
            model.eval()
            
            assets_data["model"] = model
            st.success("✅ 자산 로드 완료.")
            return assets_data
        
        except Exception as e:
            st.warning(f"모델 로드 실패: {e}. 자산을 삭제하고 재학습을 시도합니다.")
            if os.path.exists(MODEL_PATH): os.remove(MODEL_PATH)
            if os.path.exists(ASSETS_PATH): os.remove(ASSETS_PATH)

    # --- B. 파일이 없는 경우 (최초 1회 학습 실행) ---
    st.warning(f"'{MODEL_PATH}' 파일이 없습니다. 지금부터 모델 학습을 시작합니다...")
    st.info("이 작업은 앱 최초 실행 시 한 번만 수행되며, 수 분 정도 걸릴 수 있습니다.")
    
    # [기존 1. 모델 학습 및 데이터 환경 준비]
    file_path = "AI_ALD1.csv.csv"
    try:
        df = pd.read_csv(file_path, encoding='CP949')
    except Exception as e:
        st.error(f"[치명적 오류] 파일 로드 실패: {e}. 프로그램을 종료합니다.")
        st.stop()

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

    target_cols = [
        'Thickness (nm)', 'Surface Roughness (RMS, nm)', 'Uniformity (%)',
        'Density (g/cm3)', 'GPC (A/cycle)',
        'Leakage Current Density (A/cm2)', 'Dielectric Constant (ε)',
        'Breakdown Field (MV/cm)'
    ]
    cols_to_ignore_for_ai = [
        'Step Coverage (sc, %)',
        'Aspect Ratio (AR)'
    ]

    try:
        ALL_INPUT_FEATURES_ORDERED = df_encoded.drop(
            columns=target_cols + cols_to_ignore_for_ai
        ).columns.tolist()
        ALL_OUTPUT_FEATURES_ORDERED = target_cols
    except KeyError:
        st.error("[치명적 오류] CSV에 없는 컬럼명이 포함되어 있습니다.")
        st.stop()

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
    X_train_scaled = X_scaler.fit_transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)
    Y_train_scaled = Y_scaler.fit_transform(Y_train)
    Y_test_scaled = Y_scaler.transform(Y_test)
    INPUT_SIZE = X_train_scaled.shape[1]
    OUTPUT_SIZE = Y_train.shape[1]

    # [기존 2. AI 모델 클래스 정의 및 학습 실행]
    final_learning_rate = 0.00195
    final_dropout_rate = 0.28
    final_batch_size = 16
    final_epochs = 500
    VALIDATION_SPLIT = 0.2
    PATIENCE = 30
    WEIGHT_DECAY = 1e-5

    st.info(f"--- 2단계: AI 모델 학습 시작 (최대 {final_epochs} 에포크) ---")

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

    best_val_loss = float('inf')
    patience_counter = 0

    # (학습 진행률 표시를 위한 st.progress)
    progress_bar = st.progress(0, "학습 진행 중...")
    
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
        
        # (프로그레스 바 업데이트)
        progress_bar.progress((epoch + 1) / final_epochs, f"Epoch [{epoch+1}/{final_epochs}], Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # (학습이 완료된 모델 상태를 저장)
            torch.save(final_model.state_dict(), MODEL_PATH) 
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= PATIENCE:
            st.info(f"[조기 종료] Epoch {epoch+1}에서 학습을 중단합니다.")
            break
    
    progress_bar.empty() # 프로그레스 바 제거
    st.success(f"✅ 모델 학습 완료. '{MODEL_PATH}'에 저장되었습니다.")
    
    # 학습된 최종 모델 로드 (혹은 이미 메모리에 있는 것 사용)
    final_model.load_state_dict(torch.load(MODEL_PATH))
    final_model.eval()

    # [자산 패키징]
    assets_data = {
        "model": final_model,
        "X_scaler": X_scaler,
        "Y_scaler": Y_scaler,
        "ALL_INPUT_FEATURES_ORDERED": ALL_INPUT_FEATURES_ORDERED,
        "ALL_OUTPUT_FEATURES_ORDERED": ALL_OUTPUT_FEATURES_ORDERED
    }

    # (스케일러와 피처 리스트를 파일로 저장)
    joblib.dump({k: v for k, v in assets_data.items() if k != 'model'}, ASSETS_PATH)
    st.success(f"✅ 전처리기(Scaler) 및 자산을 '{ASSETS_PATH}'에 저장했습니다.")

    return assets_data
# -----------------------------------------------------------------
# [앱 시작]
# -----------------------------------------------------------------
st.set_page_config(page_title="ALD 레시피 최적화", layout="wide")
st.title("✨ AI 기반 ALD 공정 최적화 시스템")

# 1. 자산 로드 (이 함수가 모든 것을 처리)
# (최초 실행 시 여기서 학습이 발생하고 멈춰 있게 됨)
try:
    ASSETS = load_or_train_model()
except Exception as e:
    st.error(f"자산 로드 또는 모델 학습 중 치명적 오류 발생: {e}")
    st.exception(e)
    st.stop() # 👈 오류 시 앱 중지

# 2. 로드된 자산을 전역 변수로 할당
final_model = ASSETS["model"]
X_scaler = ASSETS["X_scaler"]
Y_scaler = ASSETS["Y_scaler"]
ALL_INPUT_FEATURES_ORDERED = ASSETS["ALL_INPUT_FEATURES_ORDERED"]
ALL_OUTPUT_FEATURES_ORDERED = ASSETS["ALL_OUTPUT_FEATURES_ORDERED"]


# --- 3. 물리 변수 계산 함수 정의 (SC 전담) ---
# (이전과 동일 ... 생략)
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
# (이전과 동일 ... 생략)
def create_model_input(
    recipe_params: Dict[str, Any],
    precursor_name: str,
    co_reactant_name: str,
    purge_gas_name: str
) -> pd.DataFrame:
    input_df = pd.DataFrame(columns=ALL_INPUT_FEATURES_ORDERED)
    input_df.loc[0] = 0.0
    for key, value in recipe_params.items():
        if key in input_df.columns:
            input_df.at[0, key] = value
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
# (이전과 동일 ... 생략)
def predict_from_recipe(
    recipe_params: Dict[str, Any],
    precursor_name: str,
    co_reactant_name: str,
    purge_gas_name: str
) -> pd.Series:
    input_df = create_model_input(recipe_params, precursor_name, co_reactant_name, purge_gas_name)
    X_scaled = X_scaler.transform(input_df.values)
    X_tensor = torch.from_numpy(X_scaled).float()
    final_model.eval()
    with torch.no_grad():
        Y_pred_scaled_tensor = final_model(X_tensor)
    Y_pred_unscaled = Y_scaler.inverse_transform(Y_pred_scaled_tensor.numpy())[0]
    predicted_results = pd.Series(Y_pred_unscaled, index=ALL_OUTPUT_FEATURES_ORDERED).round(4)
    return predicted_results

# --- 5.5 & 6. 최적화 '제약 조건' 및 '목적 함수' ---
# (이전과 동일 ... 생략)
COST_WEIGHTS = {
    "thickness": 100.0,
    "gpc": 30.0,
    "roughness": 10.0
}
def constraint_step_coverage(
    x: np.ndarray,
    user_input: Dict[str, Any],
    co_reactant_name: str,
    purge_gas_name: str,
    cost_weights: Dict[str, float]
) -> float:
    target_ar = user_input["Target AR"]
    if target_ar <= 5: TARGET_SC_MIN = 98.0
    elif target_ar <= 15: TARGET_SC_MIN = 90.0
    else: TARGET_SC_MIN = 85.0
    T_celsius = x[0]; P_torr = x[1]; Pulse_Time_s = x[2]
    phys_sc = calculate_full_sc_model(
        P_torr=P_torr, T_celsius=T_celsius, Pulse_Time_s=Pulse_Time_s,
        AR_value=user_input["Target AR"], precursor_name=user_input["Precursor"],
        CD_m=user_input["CD (nm)"] * 1e-9
    )
    return phys_sc - TARGET_SC_MIN

def objective_function(
    x: np.ndarray,
    user_input: Dict[str, Any],
    co_reactant_name: str,
    purge_gas_name: str,
    cost_weights: Dict[str, float]
) -> float:
    recipe_params = {
        "Temperature (c)": x[0], "Pressure (torr)": x[1],
        "Precursor_Pulse Time (s)": x[2], "Purge Time (s)": x[3],
        "Purge Gas Flow Rate (cm3/min)": x[4], "Cycles (n)": x[5],
        "Co-reactant_Pulse Time (s)": x[2]
    }
    try:
        predicted_results = predict_from_recipe(
            recipe_params, user_input["Precursor"],
            co_reactant_name, purge_gas_name
        )
    except Exception as e:
        return 1e9
    target_thickness = user_input["Thickness (nm)"]
    target_gpc_ideal = (target_thickness * 10) / (x[5] + 1e-6)
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
# (이전과 동일 ... 생략)
def generate_optimal_recipe_from_model(user_input: Dict[str, Any]):
    precursor = user_input["Precursor"]; target_thickness = user_input["Thickness (nm)"]; target_ar = user_input["Target AR"]; CD_m = user_input["CD (nm)"] * 1e-9
    co_reactant = 'H2O' if precursor in ['TMA', 'TDMAH'] else 'O3'
    purge_gas = "N2"
    bounds = [
        (150, 400), (0.01, 1.0), (0.05, 2.0), (1.0, 10.0), (50, 500),
        (max(10, int(target_thickness / 2.0)), int(target_thickness / 0.3))
    ]
    initial_guess = [
        300, 0.5, 0.1, 5.0, 300, int(target_thickness / 1.0)
    ]
    args = (user_input, co_reactant, purge_gas, COST_WEIGHTS)
    constraints = ({'type': 'ineq', 'fun': constraint_step_coverage, 'args': args})
    result = minimize(
        objective_function, initial_guess, args=args, method='SLSQP',
        bounds=bounds, constraints=constraints,
        options={'maxiter': 100, 'eps': 1e-6}
    )
    
    if not result.success:
        st.warning(f"[경고] 최적화 수렴 실패: {result.message}")
        
    optimal_x = result.x
    optimal_recipe_params = {
        "Temperature (c)": optimal_x[0], "Pressure (torr)": optimal_x[1],
        "Precursor_Pulse Time (s)": optimal_x[2], "Purge Time (s)": optimal_x[3],
        "Purge Gas Flow Rate (cm3/min)": optimal_x[4], "Cycles (n)": optimal_x[5],
        "Co-reactant_Pulse Time (s)": optimal_x[2]
    }
    predicted_results = predict_from_recipe(
        optimal_recipe_params, precursor, co_reactant, purge_gas
    )
    T = optimal_x[0]; P = optimal_x[1]; Pulse_Time = optimal_x[2]
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
        "Optimization Success": result.success, "Optimization Message": result.message,
        "Function Evaluations (nfev)": result.nfev, "Iterations (nit)": result.nit,
        "Final Cost (fun)": f"{result.fun:.6f}"
    }
    return optimal_recipe_report, predicted_results, validation_data, optimization_stats


# --- 8. Streamlit UI 구성 ---
# (이 코드는 load_or_train_model 함수가 성공적으로 완료된 후에만 실행됨)
available_precursors = {1: "TMA", 2: "TDMAH", 3: "TEMAHf", 4: "Zr(NEt2)4"}

with st.sidebar:
    st.header("🎯 목표 조건 입력")
    selected_precursor_name = st.selectbox(
        "1. 전구체 (Precursor)",
        options=available_precursors.values(), index=0
    )
    thickness = st.number_input(
        "2. 목표 박막 두께 (Thickness, nm)",
        min_value=1.0, max_value=200.0, value=15.0, step=1.0
    )
    target_ar = st.number_input(
        "3. 목표 종횡비 (Aspect Ratio, AR)",
        min_value=1.0, max_value=100.0, value=10.0, step=0.5
    )
    critical_dimension_nm = st.number_input(
        "4. 채널 폭 (Critical Dimension, CD, nm)",
        min_value=10.0, max_value=500.0, value=100.0, step=10.0
    )
    start_button = st.button("🚀 최적 레시피 생성하기", use_container_width=True, type="primary")

# --- 9. 버튼 클릭 시 최적화 실행 및 결과 표시 ---
if start_button:
    user_input = {
        "Precursor": selected_precursor_name,
        "Thickness (nm)": thickness,
        "Target AR": target_ar,
        "CD (nm)": critical_dimension_nm
    }
    
    with st.spinner(f"⏳ '{selected_precursor_name}' 전구체를 사용하여 최적 레시피를 탐색 중입니다... (SLSQP)"):
        try:
            optimal_recipe, predicted_results, validation_data, optimization_stats = \
                generate_optimal_recipe_from_model(user_input)
            
            st.success("✅ 최적화 완료!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🤖 AI 제안 최적 공정 레시피")
                st.dataframe(pd.Series(optimal_recipe).to_frame("값"))
                st.subheader("📈 최적화(SLSQP) 수렴 리포트")
                st.dataframe(pd.Series(optimization_stats).to_frame("상태"))
            with col2:
                st.subheader("💡 AI 예측 박막 특성 (8가지)")
                st.dataframe(predicted_results.to_frame("예측값"))
                st.subheader("🔬 물리 기반 검증 (SC)")
                st.dataframe(pd.Series(validation_data).to_frame("값"))

            st.markdown("---")
            st.info(f"**물리 모델 SC:** {validation_data['SC (Full Model)']} (최적화 제약조건으로 사용됨)\n\n"
                    f"**최적화 목표 오차 (Cost):** {optimization_stats['Final Cost (fun)']} (Thickness, GPC, Roughness 기준)")

        except Exception as e:
            st.error(f"최적화 중 오류가 발생했습니다: {e}")
            st.exception(e)
else:
    st.info("👈 왼쪽 사이드바에서 목표 조건을 입력하고 '최적 레시피 생성하기' 버튼을 클릭하세요.")