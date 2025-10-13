# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import warnings

warnings.filterwarnings('ignore')

# --- 데이터 로딩 및 모델 학습 함수 (캐시 사용으로 속도 향상) ---
@st.cache_data
def load_and_train_model():
    # 1. 실제 데이터 로드 및 정제
    file_name = 'HCDS Data sample - HCDS Data Sample.csv'
    df = pd.read_csv(file_name, encoding='cp949')
    
    if 'T Level' in df.columns:
        df = df.drop(columns=['T Level'])
    for col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.extract(r'(\d+\.?\d*)', expand=False), errors='coerce')
    df.dropna(inplace=True)
    
    # AI 모델 학습
    features = ['T (oC)', 'P (mTorr)', 'F (sccm)']
    target = 'Depo Rate (nm/cycle)'
    X = df[features]
    y = df[target]
    model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
    
    return model, X

# --- 웹 UI 구성 ---
st.title('🤖 ALD 공정 최적 조건 추천 AI')

# 모델 로드 및 학습
model, X = load_and_train_model()

st.sidebar.header('목표 조건 입력')
target_depth = st.sidebar.number_input(
    '목표 Deposition Rate (nm/cycle)를 입력하세요:', 
    min_value=0.01, 
    max_value=0.20, 
    value=0.10, 
    step=0.01
)

# 최적화 실행 버튼
if st.sidebar.button('최적 조건 계산하기'):

    # --- 베이즈 최적화 실행 ---
    search_space = [
        Real(X['T (oC)'].min(), X['T (oC)'].max(), name='T (oC)'),
        Real(X['P (mTorr)'].min(), X['P (mTorr)'].max(), name='P (mTorr)'),
        Real(X['F (sccm)'].min(), X['F (sccm)'].max(), name='F (sccm)')
    ]

    @use_named_args(search_space)
    def objective_function(**params):
        input_df = pd.DataFrame([params])
        predicted_depth = model.predict(input_df)[0]
        return (predicted_depth - target_depth)**2

    with st.spinner('AI가 최적 조건을 탐색 중입니다...'):
        result = gp_minimize(
            func=objective_function,
            dimensions=search_space,
            n_calls=50,
            random_state=42
        )
    
    optimal_conditions = result.x
    final_predicted_depth = np.sqrt(result.fun) + target_depth

    st.success('최적 조건 탐색 완료!')
    
    # --- 최종 결과 출력 ---
    st.header('AI 추천 최적 공정 레시피')
    st.metric(label="요청한 목표 Depo Rate", value=f"{target_depth:.4f} nm/cycle")
    st.metric(label="AI 예상 최적 Depo Rate", value=f"{final_predicted_depth:.4f} nm/cycle")

    st.subheader('추천 최적 조건 (정밀)')
    results_df = pd.DataFrame([optimal_conditions], columns=[dim.name for dim in search_space])
    results_df = results_df.round(2)
    st.dataframe(results_df)

else:
    st.info('왼쪽 사이드바에서 목표 조건을 입력하고 버튼을 눌러주세요.')