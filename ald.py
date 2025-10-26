# 파일명: app.py (Live 탐사 AI 최종 완성본)

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import os
import time
import sys
import requests
import io
import pdfplumber

try:
    from serpapi import GoogleSearch
    SERPAPI_ENABLED = True
except ImportError:
    SERPAPI_ENABLED = False

# ==============================================================================
# 0. 데이터 분석 및 정제 함수
# ==============================================================================
def analyze_and_clean_data(df):
    rename_map = {}
    for col in df.columns:
        col_lower = str(col).lower()
        if 'temp' in col_lower: rename_map[col] = 'temperature_c'
        elif 'press' in col_lower: rename_map[col] = 'pressure_torr'
        elif 'cycle' in col_lower: rename_map[col] = 'total_cycles'
        elif 'thick' in col_lower or 'rate' in col_lower or 'gpc' in col_lower: rename_map[col] = 'thickness_nm'
    
    df_std = df.rename(columns=rename_map)
    
    # 베테랑 수사관의 능력: '온도'와 '두께'만 있어도 유효 데이터로 인정
    required_cols = ['temperature_c', 'thickness_nm']
    if all(col in df_std.columns for col in required_cols):
        # 숫자 변환이 가능한 데이터만 남김
        for col in required_cols:
            df_std[col] = pd.to_numeric(df_std[col], errors='coerce')
        df_std = df_std.dropna(subset=required_cols)
        
        if not df_std.empty:
            return df_std
    return None

# ==============================================================================
# 1. Live 탐사 AI 모델 생성
# ==============================================================================
@st.cache_resource(show_spinner=False) # 스피너를 직접 제어하기 위해 False로 설정
def build_live_explorer_ai():
    is_web_mode = "streamlit" in " ".join(sys.argv)
    
    with st.spinner("Live 탐사 AI가 임무를 시작합니다..."):
        if not is_web_mode: print("Live 탐사 AI가 임무를 시작합니다...")
        all_valid_dfs = []

        # --- 1단계: 로컬 파일 탐사 ---
        if not is_web_mode: print("\n[1단계] 로컬 파일을 정밀 분석합니다...")
        for filename in os.listdir('.'):
            if filename.lower().startswith(('app', 'requirement', '~$')): continue
            try:
                if filename.lower().endswith('.xlsx'):
                    df = pd.read_excel(filename)
                elif filename.lower().endswith(('.csv', '.cell')):
                    df = pd.read_csv(filename, sep=None, engine='python', on_bad_lines='skip', encoding='utf-8')
                else:
                    continue
                
                valid_df = analyze_and_clean_data(df)
                if valid_df is not None:
                    all_valid_dfs.append(valid_df)
                    if not is_web_mode: print(f"  ✅ '{filename}'에서 유효 데이터 {len(valid_df)}개 확보.")
            except Exception:
                pass

        # --- 2단계: 인터넷 Live 탐사 ---
        if not is_web_mode: print("\n[2단계] 인터넷 Live 탐사를 시작합니다...")
        SERPAPI_API_KEY = "52b9b85163f1d3b8819e9aae64c928bf034b99d9e5be51b39374712e8d32318b"
        
        if SERPAPI_ENABLED and SERPAPI_API_KEY != "...":
            search_queries = [
                'atomic layer deposition experimental data filetype:csv',
                'ALD process parameters "growth rate" filetype:pdf'
            ]
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

            for query in search_queries:
                try:
                    if not is_web_mode: print(f"\n  -> 탐사 임무: '{query}'")
                    params = {"engine": "google", "q": query, "api_key": SERPAPI_API_KEY, "num": 5} # 5개 결과만 확인
                    search = GoogleSearch(params)
                    results = search.get_dict().get('organic_results', [])
                    
                    for result in results:
                        url = result.get('link')
                        if not url: continue

                        if not is_web_mode: print(f"    -> '{url[:60]}...' 탐사 중...")
                        
                        try:
                            response = requests.get(url, headers=headers, timeout=15)
                            response.raise_for_status()

                            if url.endswith('.csv'):
                                df = pd.read_csv(io.StringIO(response.text), on_bad_lines='skip')
                                valid_df = analyze_and_clean_data(df)
                                if valid_df is not None: all_valid_dfs.append(valid_df)
                            
                            elif url.endswith('.pdf'):
                                with pdfplumber.open(io.BytesIO(response.content)) as pdf:
                                    for page in pdf.pages:
                                        for table in page.extract_tables():
                                            df = pd.DataFrame(table[1:], columns=table[0])
                                            valid_df = analyze_and_clean_data(df)
                                            if valid_df is not None: all_valid_dfs.append(valid_df)
                        except Exception:
                            pass # 실패 시 조용히 다음으로 넘어감
                except Exception as e:
                    if not is_web_mode: print(f"  🔥 탐사 임무 실패: {e}")
        else:
            if not is_web_mode: print("  ⚠️ 탐사 허가증(SerpApi 키)이 없어 인터넷 탐사를 진행할 수 없습니다.")

        # --- 3단계: 보고 및 최종 분석 ---
        if not all_valid_dfs:
            raise ValueError("탐사 실패: 어떠한 유효 데이터도 확보하지 못했습니다. 인터넷 연결이나 로컬 파일을 확인해주세요.")

        if not is_web_mode: print("\n[3단계] 모든 탐사 결과를 종합하여 최종 보고서를 작성합니다...")
        master_df = pd.concat(all_valid_dfs, ignore_index=True)
        
        # 누락된 정보 추리
        final_cols = ['temperature_c', 'pressure_torr', 'total_cycles', 'thickness_nm']
        for col in final_cols:
            if col not in master_df.columns: master_df[col] = np.nan
        
        master_df = master_df[final_cols].apply(pd.to_numeric, errors='coerce')
        for col in ['pressure_torr', 'total_cycles']:
            if master_df[col].isnull().any():
                master_df[col].fillna(master_df[col].median(), inplace=True)
        master_df.dropna(inplace=True)

        if len(master_df) < 5:
            raise ValueError(f"보고서 작성 실패: 최종 유효 데이터가 {len(master_df)}개뿐입니다. 분석을 진행하기에 정보가 너무 부족합니다.")

        X = master_df[['temperature_c', 'pressure_torr', 'total_cycles']]
        y = master_df['thickness_nm']
        scaler = StandardScaler().fit(X); X_scaled = scaler.transform(X)
        model = XGBRegressor(random_state=42, n_estimators=100, max_depth=3).fit(X_scaled, y)
        
        if not is_web_mode: print(f"✅ 총 {len(master_df)}개의 데이터를 기반으로 최종 보고서(AI 모델) 작성 완료!")
        return model, scaler, X.columns

# ==============================================================================
# 2. 최적 조건 탐색 및 3. UI 실행 로직 (이전과 동일)
# ==============================================================================
@st.cache_data
def find_optimal_conditions(_model, _scaler, _feature_names, target_thickness, num_samples=50000):
    ranges = {'temperature_c': (100, 400), 'pressure_torr': (0.01, 2.0), 'total_cycles': (50, 1000)}
    candidates = pd.DataFrame({key: np.random.uniform(low, high, num_samples) for key, (low, high) in ranges.items()})
    candidates_scaled = _scaler.transform(candidates[_feature_names])
    predictions = _model.predict(candidates_scaled)
    best_index = np.abs(predictions - target_thickness).argmin()
    best_conditions = candidates.iloc[best_index].round(2)
    predicted_thickness = predictions[best_index]
    return best_conditions, predicted_thickness

def run_web_app(model, scaler, feature_names):
    st.set_page_config(page_title="ALD 공정 최적화 시스템")
    st.title("🎯 AI 기반 최적 공정 레시피 제안 시스템")
    st.sidebar.header("🏆 목표 결과값 입력")
    target_thick = st.sidebar.number_input("목표 박막 두께 (nm)", min_value=1.0, max_value=100.0, value=25.0, step=0.1)
    if st.sidebar.button("🤖 최적 조건 찾기"):
        with st.spinner("AI가 최적의 공정 조건을 탐색 중입니다..."):
            best_conditions, predicted_thickness = find_optimal_conditions(model, scaler, feature_names, target_thick)
        st.subheader("💡 AI가 제안하는 최적 공정 레시피")
        st.metric("목표 대비 AI 예측 두께", f"{predicted_thickness:.2f} nm", f"{predicted_thickness - target_thick:.2f} nm 오차")
        st.table(pd.DataFrame(best_conditions).T)

def run_terminal_app(model, scaler, feature_names):
    print("\n--- 💻 AI 최적 공정 탐색 터미널 모드 (1회 실행) ---")
    try:
        target_thick_str = input("\n목표 박막 두께(nm)를 입력하세요: ")
        target_thick = float(target_thick_str)
        best_conditions, predicted_thickness = find_optimal_conditions(model, scaler, feature_names, target_thick)
        print("\n💡 AI가 제안하는 최적 공정 레시피:")
        print(f"   - AI 예측 두께: {predicted_thickness:.2f} nm (목표: {target_thick:.2f} nm)")
        print("--- 제안 조건 ---")
        print(pd.DataFrame(best_conditions).T.to_string())
    except (ValueError, KeyboardInterrupt):
        print("\n⚠️ 잘못된 입력이거나 작업이 취소되었습니다.")
    finally:
        print("\n프로그램을 종료합니다.")

if __name__ == "__main__":
    is_streamlit_run = "streamlit" in " ".join(sys.argv)
    model, scaler, feature_names = build_live_explorer_ai()
    if is_streamlit_run:
        run_web_app(model, scaler, feature_names)
    else:
        run_terminal_app(model, scaler, feature_names)