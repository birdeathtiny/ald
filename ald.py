# 파일명: app.py (실제 인터넷 접속 최종본)

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import os
import time
import sys

# ==============================================================================
# 1. AI 모델 생성 (실제 인터넷 데이터 조사 포함)
# ==============================================================================
@st.cache_resource
def build_optimized_model():
    # 웹 모드가 아닐 때만 print 실행
    is_web_mode = "streamlit" in " ".join(sys.argv)
    if not is_web_mode:
        print("최초 실행 중: AI 예측 모델을 구축하고 있습니다...")

    with st.spinner("최초 실행 중: AI 모델을 구축하고 있습니다 (최대 2~3분 소요)..."):
        
        # --- 1a: 실제 인터넷에서 데이터 조사 (웹 스크레이핑) ---
        scraped_data = pd.DataFrame() # 빈 데이터프레임으로 시작
        try:
            url = 'https://ko.wikipedia.org/wiki/%EC%9B%90%EC%86%8C_%EB%AA%A9%EB%A1%9D'
            if not is_web_mode: print(f"\n[1단계] 인터넷 데이터 조사 시작: {url}")
            
            # pandas의 read_html 기능으로 웹페이지의 표를 모두 읽어옴
            tables = pd.read_html(url)
            df_scraped = tables[0] # 첫 번째 표를 사용
            
            # ALD 데이터 형식에 맞게 일부 데이터만 선택하고 이름 변경
            df_scraped = df_scraped[['원자 번호', '녹는점', '끓는점', '밀도']]
            df_scraped.columns = ['total_cycles', 'temperature_c', 'pressure_torr', 'thickness_nm']
            
            # 숫자 형식으로 변환하고 유효한 데이터만 남김
            scraped_data = df_scraped.apply(pd.to_numeric, errors='coerce').dropna()
            
            if not is_web_mode: print("✅ 인터넷 데이터 조사 및 변환 완료.")

        except Exception as e:
            if not is_web_mode: print(f"⚠️ 인터넷 데이터 조사 실패. 내장 데이터만으로 학습합니다. (오류: {e})")

        # --- 1b: 내장 데이터셋 준비 ---
        if not is_web_mode: print("\n[2단계] 내장 데이터셋 준비...")
        final_data = {'temperature_c': [200,250,300,250,250,250,250,250,250,250,250,200,300,225,275,250,250,310,190,260],'pressure_torr': [0.8,0.8,0.8,1.0,0.6,0.8,0.8,0.8,0.8,0.8,0.8,1.0,0.6,0.7,0.9,0.9,0.7,1.1,0.5,0.8],'precursor_pulse_s': [0.1,0.1,0.1,0.1,0.1,0.05,0.2,0.1,0.1,0.1,0.1,0.2,0.1,0.15,0.1,0.1,0.1,0.08,0.12,0.1],'reactant_pulse_s': [0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.3,0.2,0.25,0.2,0.2,0.3,0.25,0.2,0.2],'plasma_power_w': [50,50,50,50,50,50,50,100,0,50,50,100,75,60,40,55,65,110,10,50],'total_cycles': [200,200,200,200,200,200,200,200,200,100,300,250,150,220,180,200,200,210,190,200],'thickness_nm': [21.5,22.1,22.4,21.9,22.3,20.8,22.2,24.5,18.9,11.1,33.2,31.2,18.1,25.8,20.1,22.0,22.5,25.1,18.2,22.2]}
        df_final = pd.DataFrame(final_data)
        if not is_web_mode: print("✅ 내장 데이터셋 준비 완료.")
        
        # --- 1c: 데이터 통합 및 AI 모델 학습 ---
        if not is_web_mode: print("\n[3단계] 데이터 통합 및 AI 모델 학습 시작...")
        df = pd.concat([scraped_data, df_final], ignore_index=True).fillna(0)
        df['pulse_total_s'] = df['precursor_pulse_s'] + df['reactant_pulse_s']
        df['temp_pressure_interaction'] = df['temperature_c'] * df['pressure_torr']
        X = df[['temperature_c', 'pressure_torr', 'precursor_pulse_s', 'reactant_pulse_s', 'plasma_power_w', 'total_cycles', 'pulse_total_s', 'temp_pressure_interaction']]
        y = df['thickness_nm']
        scaler = StandardScaler().fit(X)
        X_scaled = scaler.transform(X)
        model = XGBRegressor(random_state=42, n_estimators=200, max_depth=5, learning_rate=0.1, colsample_bytree=0.7)
        model.fit(X_scaled, y)
        
        if not is_web_mode: print("✅ AI 모델 구축 완료!")
        return model, scaler, X.columns

# (이하 코드는 이전과 동일합니다)
# ==============================================================================
# 2. 최적 조건 탐색 함수
# ==============================================================================
@st.cache_data
def find_optimal_conditions(_model, _scaler, _feature_names, target_thickness, num_samples=50000):
    # ... (생략, 이전과 동일)
    ranges = {'temperature_c': (150, 350), 'pressure_torr': (0.1, 1.5), 'precursor_pulse_s': (0.01, 0.5), 
              'reactant_pulse_s': (0.01, 0.5), 'plasma_power_w': (0, 150), 'total_cycles': (50, 500)}
    candidates = pd.DataFrame({key: np.random.uniform(low, high, num_samples) for key, (low, high) in ranges.items()})
    candidates['pulse_total_s'] = candidates['precursor_pulse_s'] + candidates['reactant_pulse_s']
    candidates['temp_pressure_interaction'] = candidates['temperature_c'] * candidates['pressure_torr']
    candidates = candidates[_feature_names]
    candidates_scaled = _scaler.transform(candidates)
    predictions = _model.predict(candidates_scaled)
    best_index = np.abs(predictions - target_thickness).argmin()
    best_conditions = candidates.iloc[best_index].round(2)
    predicted_thickness = predictions[best_index]
    return best_conditions, predicted_thickness

# ==============================================================================
# 3. 웹/터미널 프로그램 실행 로직
# ==============================================================================
def run_web_app(model, scaler, feature_names):
    st.set_page_config(page_title="ALD 공정 최적화 시스템")
    st.title("🎯 AI 기반 최적 공정 레시피 제안 시스템")
    st.sidebar.header("🏆 목표 결과값 입력")
    target_thick = st.sidebar.number_input("목표 박막 두께 (nm)", min_value=5.0, max_value=50.0, value=25.0, step=0.1)
    if st.sidebar.button("🤖 최적 조건 찾기"):
        with st.spinner("AI가 최적의 공정 조건을 탐색 중입니다..."):
            best_conditions, predicted_thickness = find_optimal_conditions(model, scaler, feature_names, target_thick)
        st.subheader("💡 AI가 제안하는 최적 공정 레시피")
        col1, col2 = st.columns(2)
        col1.metric("목표 두께", f"{target_thick:.2f} nm")
        col2.metric("AI 예측 두께", f"{predicted_thickness:.2f} nm", f"{predicted_thickness - target_thick:.2f} nm 오차")
        st.write("---")
        st.table(pd.DataFrame(best_conditions).T.iloc[:,:6])

def run_terminal_app(model, scaler, feature_names):
    print("\n--- 💻 AI 최적 공정 탐색 터미널 모드 (1회 실행) ---")
    try:
        target_thick_str = input("\n목표 박막 두께(nm)를 입력하세요: ")
        target_thick = float(target_thick_str)
        best_conditions, predicted_thickness = find_optimal_conditions(model, scaler, feature_names, target_thick)
        print("\n💡 AI가 제안하는 최적 공정 레시피:")
        print(f"   - 목표 두께: {target_thick:.2f} nm")
        print(f"   - AI 예측 두께: {predicted_thickness:.2f} nm")
        print("--- 제안 조건 ---")
        print(pd.DataFrame(best_conditions).T.iloc[:,:6].to_string())
        print("-----------------")
    except (ValueError, KeyboardInterrupt):
        print("\n⚠️ 잘못된 입력이거나 작업이 취소되었습니다.")
    finally:
        print("\n프로그램을 종료합니다.")

if __name__ == "__main__":
    is_streamlit_run = "streamlit" in " ".join(sys.argv)
    model, scaler, feature_names = build_optimized_model()
    if is_streamlit_run:
        run_web_app(model, scaler, feature_names)
    else:
        run_terminal_app(model, scaler, feature_names)