# ... (앞부분 import 및 클래스 정의는 동일) ...
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.exceptions import NotFittedError
import joblib
import os
import sys
import streamlit as st

# ==========================================
# 0. 환경 설정 및 상수 정의
# ==========================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
st.info(f"사용 장치: {DEVICE}")

L_CHARACTERISTIC_LENGTH_M = 0.01 
WAFER_SURFACE_AREA_M2 = np.pi * (0.05**2) 
K_LAMBDA = 1e-5 
K_DA = 1.0 
K_UTIL = 1e-6 
EPSILON = 1e-6

MIN_PRECURSOR_PULSE_S = 0.1
MIN_COREACTANT_PULSE_S = 0.1
MIN_PURGE_S = 0.5

EPOCHS = 300 
LEARNING_RATE = 0.001

# 파일 경로
MODEL_PATH = 'ald_hybrid_model_v2.pth'
SCALER_X_PATH = 'scaler_X_v2.pkl'
SCALER_Y_PATH = 'scaler_Y_v2.pkl'
ENCODER_PATH = 'encoder_precursor_v2.pkl'
DATA_FILE = 'AI_ALD1.csv.csv'

output_cols_all = [
    'GPC (A/cycle)', 'Utilization_Proxy', 'R_max (A/s)', 'Thickness (nm)'
]
process_cols_base = [
    'Precursor_Pulse Time (s)', 'Co-reactant_Pulse Time (s)', 'Cycles (n)',
    'Temperature (c)', 'Pressure (torr)', 'Purge Time (s)',
    'Purge Gas Flow Rate (cm3/min)', 'Precursor Flow Rate (cm3/min)',
    'Co-reactant Flow Rate (cm3/min)'
]
new_input_features = ['Knudsen_Number (Kn)', 'Damkohler_Number (Da)']
process_cols_all = process_cols_base + new_input_features

# ==========================================
# 1. 모델 클래스 정의 (이전 코드와 동일)
# ...
# ==========================================

# ==========================================
# 2. 물리적 일관성 손실 함수 클래스 (이전 코드와 동일)
# ...
# ==========================================

# ==========================================
# 3. 데이터 로딩 및 전처리 (수정: Scikit-learn 객체를 파일로 저장)
# ==========================================
@st.cache_resource
def load_and_preprocess_data():
    """데이터 로딩, 피처 엔지니어링, 스케일러/인코더 학습 후 저장, NumPy 배열 반환."""
    
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    file_path = os.path.join(script_dir, DATA_FILE)

    try:
        # ... (데이터 로딩, 클리닝, 피처 엔지니어링 로직은 동일) ...
        df = pd.read_csv(file_path, encoding='cp949')
        df.columns = df.columns.str.strip().str.replace('"', '', regex=False)
        numeric_cols = process_cols_base + ['Thickness (nm)', 'GPC (A/cycle)']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].replace('-', np.nan), errors='coerce')

        df_clean = df.dropna(subset=['GPC (A/cycle)', 'Thickness (nm)']).copy()
        df_clean[process_cols_base] = df_clean[process_cols_base].fillna(0)
        df_clean['Precursor'] = df_clean['Precursor'].fillna('Unknown')
        
        T_k = df_clean['Temperature (c)'] + 273.15
        P_torr = df_clean['Pressure (torr)']
        df_clean['Lambda (m)'] = K_LAMBDA * T_k / (P_torr + EPSILON)
        df_clean['Knudsen_Number (Kn)'] = df_clean['Lambda (m)'] / L_CHARACTERISTIC_LENGTH_M

        reaction_rate_proxy = df_clean['GPC (A/cycle)']
        transport_rate_proxy = df_clean['Precursor Flow Rate (cm3/min)'] / L_CHARACTERISTIC_LENGTH_M
        df_clean['Damkohler_Number (Da)'] = K_DA * reaction_rate_proxy / (transport_rate_proxy + EPSILON)

        df_clean['Total_Cycle_Time (s)'] = df_clean['Precursor_Pulse Time (s)'] + df_clean['Co-reactant_Pulse Time (s)'] + 2 * df_clean['Purge Time (s)']
        df_clean['R_max (A/s)'] = df_clean['GPC (A/cycle)'] / (df_clean['Total_Cycle_Time (s)'] + EPSILON)

        precursor_inflow_proxy = df_clean['Precursor Flow Rate (cm3/min)'] * df_clean['Precursor_Pulse Time (s)']
        deposited_proxy = df_clean['GPC (A/cycle)'] * WAFER_SURFACE_AREA_M2
        df_clean['Utilization_Proxy'] = K_UTIL * deposited_proxy / (precursor_inflow_proxy + EPSILON)

        df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_clean = df_clean.fillna(0)
        
        physically_non_negative_cols = [
            'Precursor_Pulse Time (s)', 'Co-reactant_Pulse Time (s)', 'Cycles (n)', 'Temperature (c)', 'Pressure (torr)', 'Purge Time (s)',
            'Purge Gas Flow Rate (cm3/min)', 'Precursor Flow Rate (cm3/min)', 'Co-reactant Flow Rate (cm3/min)',
            'Knudsen_Number (Kn)', 'Damkohler_Number (Da)', 'Total_Cycle_Time (s)', 'R_max (A/s)', 'Utilization_Proxy'
        ]
        for col in physically_non_negative_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].clip(lower=0)
        
        # 인코딩 및 스케일러 학습
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        precursor_encoded = encoder.fit_transform(df_clean[['Precursor']])
        
        # 스케일러 학습
        df_features = df_clean[process_cols_all].join(pd.DataFrame(precursor_encoded, columns=encoder.get_feature_names_out(['Precursor']), index=df_clean.index))
        df_outputs = df_clean[output_cols_all]
        scaler_X = MinMaxScaler()
        scaler_Y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(df_features.values)
        Y_scaled = scaler_Y.fit_transform(df_outputs.values)

        # 💡 [핵심 수정 1] Scikit-learn 객체는 즉시 파일로 저장하여 캐싱 경고를 피함
        joblib.dump(scaler_X, SCALER_X_PATH)
        joblib.dump(scaler_Y, SCALER_Y_PATH)
        joblib.dump(encoder, ENCODER_PATH)

        # 텐서 대신 NumPy 배열로 저장
        X_np = X_scaled
        Y_np = Y_scaled
        
        # 인덱스 설정
        all_input_features = df_features.columns.to_list()
        indices = {
            'cycles': all_input_features.index('Cycles (n)'),
            'pre_pulse': all_input_features.index('Precursor_Pulse Time (s)'),
            'co_pulse': all_input_features.index('Co-reactant_Pulse Time (s)'),
            'purge': all_input_features.index('Purge Time (s)'),
        }
        output_indices = {
            'gpc': output_cols_all.index('GPC (A/cycle)'),
            'util': output_cols_all.index('Utilization_Proxy'),
            'rmax': output_cols_all.index('R_max (A/s)'),
            'thick': output_cols_all.index('Thickness (nm)'),
        }
        precursor_map = {name.replace('Precursor_', ''): name for name in encoder.categories_[0]}

    except FileNotFoundError:
        st.error(f"오류: '{DATA_FILE}' 파일을 찾을 수 없습니다. 데이터 파일을 확인해주세요.")
        return None, None
    except ValueError as e:
        st.error(f"데이터 전처리 오류: {e}")
        return None, None

    # 💡 [핵심 수정 2] data_artifacts는 해시 가능한 NumPy 배열과 기본 타입만 포함
    data_artifacts = {
        'X_np': X_np, 'Y_np': Y_np, # NumPy 배열
        'indices': indices, 'output_indices': output_indices, 
        'precursor_map': precursor_map, 'all_input_features': all_input_features,
        'input_dim': df_features.shape[1]
    }
    return data_artifacts, df_features.shape[1]

# ==========================================
# 4. 모델 학습/로드 함수 (수정: Scikit-learn 객체를 파일에서 로드)
# ==========================================

# ... (train_model, evaluate_model 함수는 동일) ...

@st.cache_resource
def train_or_load_model(data_artifacts, input_dim):
    """모델 파일이 없으면 학습하고 저장, 있으면 로드"""
    
    # 💡 [핵심 수정 3] Scikit-learn 객체를 파일에서 로드
    try:
        scaler_X = joblib.load(SCALER_X_PATH)
        scaler_Y = joblib.load(SCALER_Y_PATH)
        encoder = joblib.load(ENCODER_PATH)
    except FileNotFoundError:
        # 데이터 전처리에서 파일 저장이 실패했거나 (첫 실행 시), 파일이 누락된 경우
        st.error("오류: 스케일러/인코더 파일 로드 실패. 데이터 전처리 (`load_and_preprocess_data`)를 확인하세요.")
        return None
    except Exception as e:
        st.error(f"오류: 스케일러/인코더 파일 로드 중 예외 발생: {e}")
        return None
    
    # 딕셔너리 업데이트 (최적화 함수에서 사용)
    data_artifacts['scaler_X'] = scaler_X
    data_artifacts['scaler_Y'] = scaler_Y
    data_artifacts['encoder'] = encoder
    
    if os.path.exists(MODEL_PATH):
        # 1. 모델 로드
        model = ALDHybridModel(input_dim, output_dim=2).to(DEVICE)
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model.eval()
            st.success("💾 학습된 모델과 스케일러를 성공적으로 로드했습니다.")
            return model
        except Exception as e:
            st.error(f"모델 파일 로드 오류: {e}. 다시 학습을 시도합니다.")
            os.remove(MODEL_PATH) 
            # continue to training
    
    # 2. 모델 학습 (파일이 없거나 로드 실패 시)
    st.warning("모델 파일이 없거나 오류가 발생했습니다. 새 모델 학습을 시작합니다. (약 1분 소요)")
    
    # NumPy 배열을 텐서로 변환
    X_tensor = torch.tensor(data_artifacts['X_np'], dtype=torch.float32).to(DEVICE)
    Y_tensor = torch.tensor(data_artifacts['Y_np'], dtype=torch.float32).to(DEVICE)
    
    dataset = TensorDataset(X_tensor, Y_tensor)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    
    if train_size == 0 or test_size == 0:
        st.error("데이터셋 크기가 너무 작아 학습을 진행할 수 없습니다.")
        return None
    
    train_data, test_data = random_split(dataset, [train_size, test_size])
    BATCH_SIZE = max(1, train_size // 4)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
    
    model = ALDHybridModel(input_dim, output_dim=2).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    x_mins, x_ranges = get_scaler_tensors(scaler_X, DEVICE) # 로드된 스케일러 사용
    y_mins, y_ranges = get_scaler_tensors(scaler_Y, DEVICE)
    criterion = SelfConsistentLoss(x_mins, x_ranges, y_mins, y_ranges, data_artifacts['indices'], data_artifacts['output_indices'], DEVICE)
    
    # Streamlit에서 진행 상황을 표시
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for epoch in range(EPOCHS):
        train_loss = train_model(model, train_loader, criterion, optimizer)
        
        if (epoch + 1) % 50 == 0 or epoch == EPOCHS - 1:
            test_loss = evaluate_model(model, test_loader, criterion)
            status_text.text(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
            
        progress_bar.progress((epoch + 1) / EPOCHS)
    
    # 3. 모델 저장 (스케일러/인코더는 이미 저장됨)
    torch.save(model.state_dict(), MODEL_PATH)
    
    status_text.success("🎉 모델 학습 완료 및 파일 저장 완료!")
    model.eval()
    return model

# ... (find_best_recipe_gradient 함수는 동일) ...
# find_best_recipe_gradient 함수는 data_artifacts 딕셔너리에서 스케일러를 사용하므로
# train_or_load_model에서 딕셔너리에 스케일러를 추가해야 합니다. (위에서 처리됨)
def find_best_recipe_gradient(model, data_artifacts, target_thickness, target_gpc, selected_precursor_name, weights, n_runs=50):
    
    # 딕셔너리에 추가된 스케일러 객체 로드
    scaler_X = data_artifacts['scaler_X']
    scaler_Y = data_artifacts['scaler_Y']
    encoder = data_artifacts['encoder']
    indices = data_artifacts['indices']
    output_indices = data_artifacts['output_indices']
    all_input_features = data_artifacts['all_input_features']
    
    # ... (나머지 최적화 로직은 동일하게 유지) ...
    x_mins_opt, x_ranges_opt = get_scaler_tensors(scaler_X, DEVICE)
    y_mins_opt, y_ranges_opt = get_scaler_tensors(scaler_Y, DEVICE)

    def unscale_X_torch(x_scaled, idx):
        return x_scaled[idx] * x_ranges_opt[idx] + x_mins_opt[idx]

    def unscale_Y_torch(y_scaled, idx):
        if idx == output_indices['gpc']:
            model_output_idx = 0
        elif idx == output_indices['util']:
            model_output_idx = 1
        else:
            raise ValueError(f"Invalid idx {idx} for unscaling model prediction.")
        return y_scaled[model_output_idx] * y_ranges_opt[idx] + y_mins_opt[idx]

    precursor_one_hot = encoder.transform([[selected_precursor_name]])[0]
    precursor_ohe_tensor = torch.tensor(precursor_one_hot, dtype=torch.float32, device=DEVICE)
    
    process_cols_count = len(process_cols_all)
    
    def get_scaled_min_torch(unscaled_min, feature_idx):
        scaled_min = (unscaled_min - x_mins_opt[feature_idx]) / x_ranges_opt[feature_idx]
        return torch.clamp(scaled_min, 0.0, 1.0)

    scaled_min_pre_pulse = get_scaled_min_torch(MIN_PRECURSOR_PULSE_S, indices['pre_pulse'])
    scaled_min_co_pulse = get_scaled_min_torch(MIN_COREACTANT_PULSE_S, indices['co_pulse'])
    scaled_min_purge = get_scaled_min_torch(MIN_PURGE_S, indices['purge'])
    
    best_cost = torch.tensor(float('inf'), device=DEVICE)
    best_recipe_scaled_process = None
    
    for run in range(n_runs):
        initial_guess_scaled_process = torch.rand(process_cols_count, device=DEVICE)
        initial_guess_scaled_process[indices['pre_pulse']] = torch.max(initial_guess_scaled_process[indices['pre_pulse']], scaled_min_pre_pulse)
        initial_guess_scaled_process[indices['co_pulse']] = torch.max(initial_guess_scaled_process[indices['co_pulse']], scaled_min_co_pulse)
        initial_guess_scaled_process[indices['purge']] = torch.max(initial_guess_scaled_process[indices['purge']], scaled_min_purge)
        
        X_process_guess = initial_guess_scaled_process.clone().detach().requires_grad_(True)
        optimizer_input = torch.optim.Adam([X_process_guess], lr=0.01)
        
        for step in range(300):
            with torch.no_grad():
                X_process_guess.clamp_(0.0, 1.0)
                X_process_guess[indices['pre_pulse']].clamp_(min=scaled_min_pre_pulse)
                X_process_guess[indices['co_pulse']].clamp_(min=scaled_min_co_pulse)
                X_process_guess[indices['purge']].clamp_(min=scaled_min_purge)

            X_full_scaled = torch.cat([X_process_guess, precursor_ohe_tensor])  
            Y_pred_scaled = model(X_full_scaled.unsqueeze(0)).squeeze(0)
            
            pred_gpc_unscaled = unscale_Y_torch(Y_pred_scaled, output_indices['gpc'])
            pred_util_unscaled = unscale_Y_torch(Y_pred_scaled, output_indices['util'])
            
            unscaled_cycles = unscale_X_torch(X_full_scaled, indices['cycles'])
            unscaled_pre_pulse = unscale_X_torch(X_full_scaled, indices['pre_pulse'])
            unscaled_co_pulse = unscale_X_torch(X_full_scaled, indices['co_pulse'])
            unscaled_purge = unscale_X_torch(X_full_scaled, indices['purge'])
            
            calc_total_time = unscaled_pre_pulse + unscaled_co_pulse + 2 * unscaled_purge
            calc_rmax = pred_gpc_unscaled / (calc_total_time + EPSILON)
            calc_thick = pred_gpc_unscaled * unscaled_cycles * 0.1
            
            thickness_error = (calc_thick - target_thickness)**2
            gpc_error = (pred_gpc_unscaled - target_gpc)**2
            rmax_score = -calc_rmax
            util_score = -pred_util_unscaled
            
            cost = (weights['w_T'] * thickness_error) + \
                   (weights['w_G'] * gpc_error) + \
                   (weights['w_R'] * rmax_score) + \
                   (weights['w_U'] * util_score)
            
            optimizer_input.zero_grad()
            cost.backward()
            optimizer_input.step()
        
        final_cost = cost.detach()
        if final_cost < best_cost:
            best_cost = final_cost
            best_recipe_scaled_process = X_process_guess.detach()

    if best_recipe_scaled_process is None:
        return "Optimization failed to find a valid solution."

    best_X_scaled_full = torch.cat([best_recipe_scaled_process, precursor_ohe_tensor])
    best_recipe_full_unscaled = scaler_X.inverse_transform(best_X_scaled_full.cpu().numpy().reshape(1, -1))[0]
    
    pred_Y_scaled = model(best_X_scaled_full.unsqueeze(0)).squeeze(0).detach()
    pred_gpc_unscaled = unscale_Y_torch(pred_Y_scaled, output_indices['gpc']).cpu().numpy()
    pred_util_unscaled = unscale_Y_torch(pred_Y_scaled, output_indices['util']).cpu().numpy()
    
    unscaled_cycles = best_recipe_full_unscaled[indices['cycles']]
    unscaled_pre_pulse = best_recipe_full_unscaled[indices['pre_pulse']]
    unscaled_co_pulse = best_recipe_full_unscaled[indices['co_pulse']]
    unscaled_purge = best_recipe_full_unscaled[indices['purge']]
    
    calc_thick = pred_gpc_unscaled * unscaled_cycles * 0.1
    calc_total_time = unscaled_pre_pulse + unscaled_co_pulse + 2 * unscaled_purge
    calc_rmax = pred_gpc_unscaled / (calc_total_time + EPSILON)
    
    kn_value = best_recipe_full_unscaled[all_input_features.index('Knudsen_Number (Kn)')]
    da_value = best_recipe_full_unscaled[all_input_features.index('Damkohler_Number (Da)')]
    phi_proxy = np.sqrt(pred_gpc_unscaled / (kn_value + EPSILON))

    recipe_data = {
        'Precursor': selected_precursor_name,
        'Calculated Thickness (nm)': f"{calc_thick:.2f} (Target: {target_thickness:.2f})",
        'Predicted GPC (Å/cycle)': f"{pred_gpc_unscaled:.2f} (Target: {target_gpc:.2f})",
        'Calculated R_max (Å/s)': f"{calc_rmax:.3f}",
        'Predicted Utilization (Proxy)': f"{pred_util_unscaled:.4f}",
        'Knudsen (Kn)': f"{kn_value:.4f}",
        'Damköhler (Da)': f"{da_value:.4f}",
        'Thiele Mod. (Φ proxy)': f"{phi_proxy:.4f}",
        'Cost': f"{best_cost.item():.4f}"
    }

    recipe_df = pd.DataFrame(best_recipe_full_unscaled[:len(all_input_features)].reshape(1, -1), columns=all_input_features)
    
    return recipe_data, recipe_df


# ==========================================
# 6. Streamlit 앱 메인 함수 (이전 코드와 동일)
# ==========================================
def main_app():
    st.set_page_config(page_title="ALD Recipe Optimizer", layout="wide")
    st.title("🧪 AI ALD 하이브리드 레시피 최적화 시스템")
    st.markdown("---")

    # 1. 데이터 로드 및 전처리 (NumPy 배열 및 메타데이터 반환)
    data_artifacts, input_dim = load_and_preprocess_data()
    if data_artifacts is None:
        return

    # 2. 모델 학습/로드 (파일이 없으면 학습 진행)
    # data_artifacts는 해시 가능한 객체만 포함
    model = train_or_load_model(data_artifacts, input_dim) 
    if model is None:
        st.error("모델 학습 및 로드에 실패했습니다. 데이터 파일을 확인해주세요.")
        return

    precursor_list = list(data_artifacts['precursor_map'].keys())
    
    # ... (나머지 UI 및 최적화 실행 로직은 동일) ...
    # 3. 사용자 입력 UI (사이드바)
    st.sidebar.header("🎯 목표 설정")
    selected_precursor_name = st.sidebar.selectbox("프리커서 선택", precursor_list)
    target_thickness = st.sidebar.number_input("목표 막 두께 (nm)", value=50.0, min_value=0.1, max_value=1000.0, step=1.0)
    target_gpc = st.sidebar.number_input("목표 GPC (Å/cycle)", value=1.5, min_value=0.01, max_value=5.0, step=0.01)
    
    st.sidebar.header("⚖️ 다중 목표 가중치")
    w_T = st.sidebar.slider("두께 오차 (w_T)", 0.0, 5.0, 1.0, 0.1)
    w_G = st.sidebar.slider("GPC 오차 (w_G)", 0.0, 5.0, 1.5, 0.1)
    w_R = st.sidebar.slider("R_max (생산성) (w_R)", 0.0, 5.0, 0.5, 0.1)
    w_U = st.sidebar.slider("이용률 (효율) (w_U)", 0.0, 5.0, 0.2, 0.1)
    user_weights = {'w_T': w_T, 'w_G': w_G, 'w_R': w_R, 'w_U': w_U}
    n_runs = st.sidebar.number_input("최적화 랜덤 시작점 (n_runs)", value=50, min_value=10, max_value=200, step=10)


    # 4. 최적화 실행 버튼
    if st.button("🚀 최적 레시피 검색 시작"):
        
        with st.spinner("그래디언트 기반 최적 레시피를 검색 중입니다..."):
            
            result = find_best_recipe_gradient(
                model, data_artifacts, 
                target_thickness, target_gpc, 
                selected_precursor_name, user_weights, n_runs
            )

            if isinstance(result, str):
                st.error(result)
            else:
                recipe_data, recipe_df = result
                
                st.success("✅ 최적화 레시피 검색 완료!")
                st.subheader("결과 요약 및 성능 지표")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.json(recipe_data)

                with col2:
                    st.subheader("최적화된 공정 조건 (입력 변수)")
                    display_cols = ['Precursor_Pulse Time (s)', 'Co-reactant_Pulse Time (s)', 'Purge Time (s)',
                                    'Cycles (n)', 'Temperature (c)', 'Pressure (torr)',
                                    'Purge Gas Flow Rate (cm3/min)', 'Precursor Flow Rate (cm3/min)',
                                    'Co-reactant Flow Rate (cm3/min)']
                    
                    recipe_display = recipe_df[display_cols].T
                    recipe_display.columns = ['Optimized Value']
                    recipe_display['Optimized Value'] = recipe_display['Optimized Value'].apply(lambda x: f"{x:.3f}")
                    
                    st.table(recipe_display)
                    
                st.markdown("---")
                st.info("💡 **Knudsen/Damköhler 수 해석:** 최적화된 레시피의 물리적 레짐을 확인하세요.")

if __name__ == "__main__":
    main_app()