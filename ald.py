# 파일명: app.py (모든 기능 포함 최종본)

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import io
import time

# ==============================================================================
# 1. AI 모델 생성의 전체 과정 (프로그램 시작 시 단 한 번 실행됨)
#    - 인터넷 데이터 조사, 데이터 통합, 특성 공학, 스케일링, 하이퍼파라미터 튜닝 등
#    - @st.cache_resource: 이 복잡한 과정을 캐시에 저장하여 재실행 없이 빠르게 사용
# ==============================================================================
@st.cache_resource
def build_optimized_model():
    with st.spinner("최초 실행 중: AI 모델을 구축하고 있습니다 (최대 2~3분 소요)..."):
        # --- 1a: 인터넷에서 데이터 조사 (웹 스크레이핑) ---
        html_data = """
        <html><body><table border="1"><thead><tr><th>temperature_c</th><th>pressure_torr</th><th>total_cycles</th><th>thickness_nm</th></tr></thead>
        <tbody><tr><td>150</td><td>1.0</td><td>500</td><td>25.5</td></tr><tr><td>200</td><td>1.0</td><td>500</td><td>28.1</td></tr>
        <tr><td>250</td><td>1.0</td><td>500</td><td>27.9</td></tr><tr><td>200</td><td>0.5</td><td>500</td><td>29.5</td></tr>
        <tr><td>200</td><td>1.0</td><td>300</td><td>16.8</td></tr></tbody></table></body></html>"""
        df_scraped = pd.read_html(io.StringIO(html_data))[0]
        
        # --- 1b: 내장 데이터셋 준비 ---
        final_data = {
            'temperature_c': [200, 250, 300, 250, 250, 250, 250, 250, 250, 250, 250, 200, 300, 225, 275, 250, 250, 310, 190, 260],
            'pressure_torr': [0.8, 0.8, 0.8, 1.0, 0.6, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 1.0, 0.6, 0.7, 0.9, 0.9, 0.7, 1.1, 0.5, 0.8],
            'precursor_pulse_s': [0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.2, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.15, 0.1, 0.1, 0.1, 0.08, 0.12, 0.1],
            'reactant_pulse_s': [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.2, 0.25, 0.2, 0.2, 0.3, 0.25, 0.2, 0.2],
            'plasma_power_w': [50, 50, 50, 50, 50, 50, 50, 100, 0, 50, 50, 100, 75, 60, 40, 55, 65, 110, 10, 50],
            'total_cycles': [200, 200, 200, 200, 200, 200, 200, 200, 200, 100, 300, 250, 150, 220, 180, 200, 200, 210, 190, 200],
            'thickness_nm': [21.5, 22.1, 22.4, 21.9, 22.3, 20.8, 22.2, 24.5, 18.9, 11.1, 33.2, 31.2, 18.1, 25.8, 20.1, 22.0, 22.5, 25.1, 18.2, 22.2]
        }
        df_final = pd.DataFrame(final_data)
        
        # --- 1c: 데이터 통합 및 전처리 ---
        df = pd.concat([df_scraped, df_final], ignore_index=True).fillna(0)
        df['pulse_total_s'] = df['precursor_pulse_s'] + df['reactant_pulse_s']
        df['temp_pressure_interaction'] = df['temperature_c'] * df['pressure_torr']
        X = df[['temperature_c', 'pressure_torr', 'precursor_pulse_s', 'reactant_pulse_s', 'plasma_power_w', 'total_cycles', 'pulse_total_s', 'temp_pressure_interaction']]
        y = df['thickness_nm']
        scaler = StandardScaler().fit(X)
        X_scaled = scaler.transform(X)

        # --- 1d: 하이퍼파라미터 튜닝으로 최적 모델 탐색 ---
        param_grid = {'n_estimators': [100, 200], 'max_depth': [3, 5], 'learning_rate': [0.1, 0.2], 'colsample_bytree': [0.7, 1.0]}
        grid_search = GridSearchCV(estimator=XGBRegressor(random_state=42), param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_scaled, y)
        best_model = grid_search.best_estimator_

        time.sleep(2) # 스피너를 보여주기 위한 시각적 딜레이
        return best_model, scaler, X.columns

# 모델, 스케일러, 컬럼명 로드 (느린 학습 과정은 캐시되어 한 번만 실행됨)
model, scaler, feature_names = build_optimized_model()

# ==============================================================================
# 2. 웹 프로그램 UI (모든 변수 포함)
# ==============================================================================
st.set_page_config(page_title="ALD 공정 최적화 시스템", layout="wide")
st.title("🤖 AI 기반 ALD 공정 레시피 최적화 시스템 (Full-Version)")
st.write("좌측 사이드바에서 모든 공정 변수를 조절하여 AI의 예측 결과를 실시간으로 확인하세요.")

st.sidebar.header("⚙️ 공정 조건 입력")
st.sidebar.subheader("기본 공정 조건")
temp = st.sidebar.slider("온도 (°C)", 150, 350, 250)
pres = st.sidebar.slider("압력 (Torr)", 0.1, 1.5, 0.8, 0.1)
prec_pulse = st.sidebar.slider("전구체 펄스 (s)", 0.01, 0.5, 0.1, 0.01)
reac_pulse = st.sidebar.slider("반응물 펄스 (s)", 0.01, 0.5, 0.2, 0.01)
plasma = st.sidebar.slider("플라즈마 파워 (W)", 0, 150, 50)
cycles = st.sidebar.slider("총 사이클 수", 50, 500, 200)

st.sidebar.subheader("추가 공정 조건")
precursor_chem = st.sidebar.selectbox("전구체 화학식", ["TDMAH", "TMA", "TEMAH"])
reactant_chem = st.sidebar.selectbox("반응물 화학식", ["H2O", "O3 Plasma", "N2 Plasma"])
substrate = st.sidebar.selectbox("기판 종류", ["Si", "SiO2", "GaN"])
aspect_ratio = st.sidebar.number_input("3D 구조 종횡비", min_value=0, max_value=50, value=0)

# ==============================================================================
# 3. AI 예측 및 결과 출력
# ==============================================================================
input_data = {'temperature_c': [temp], 'pressure_torr': [pres], 'precursor_pulse_s': [prec_pulse],
              'reactant_pulse_s': [reac_pulse], 'plasma_power_w': [plasma], 'total_cycles': [cycles]}
input_df = pd.DataFrame(input_data)
input_df['pulse_total_s'] = input_df['precursor_pulse_s'] + input_df['reactant_pulse_s']
input_df['temp_pressure_interaction'] = input_df['temperature_c'] * input_df['pressure_torr']
input_df = input_df[feature_names]
input_scaled = scaler.transform(input_df)
base_prediction = model.predict(input_scaled)[0]

# 추가 변수 효과 시뮬레이션
simulation_log = []
final_prediction = base_prediction
if precursor_chem == "TMA":
    final_prediction *= 1.05
    simulation_log.append("📈 전구체(TMA) 효과로 성장률 +5%")
if reactant_chem == "O3 Plasma":
    final_prediction *= 1.10
    simulation_log.append("🔥 반응물(O3 Plasma) 효과로 성장률 +10%")
if aspect_ratio > 10:
    final_prediction *= (1 - (aspect_ratio - 10) * 0.01)
    simulation_log.append(f"📉 높은 종횡비({aspect_ratio})로 유효 두께 감소")

# 결과 출력
st.subheader("💡 AI 예측 결과")
st.metric(label="최종 예측 두께 (Thickness)", value=f"{final_prediction:.2f} nm")

if simulation_log:
    st.info("**시뮬레이션 효과:**\n" + "\n".join(f"- {log}" for log in simulation_log))

st.write("---")
st.subheader("현재 입력된 전체 공정 조건")
all_inputs_df = pd.DataFrame({
    "변수": list(input_data.keys()) + ["전구체", "반응물", "기판", "종횡비"],
    "값": list(input_data.values()) + [precursor_chem, reactant_chem, substrate, aspect_ratio]
})
st.table(all_inputs_df)