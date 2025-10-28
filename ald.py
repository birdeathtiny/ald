import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import re
import plotly.graph_objects as go
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers.schedules import ExponentialDecay

# Keras 로드 시 경고 메시지 방지
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- 0. 전역 변수 및 파일 경로 설정 ---
DATA_FILE = 'ald_data.csv'  # ⭐ 이 이름으로 파일이 존재해야 합니다.
MODEL_PATH = 'improved_ald_mimo_model.h5'
PREPROCESSOR_PATH = 'ald_preprocessor.joblib'
Y_SCALER_PATH = 'y_minmax_scaler.joblib' 

# 입력/출력 컬럼 정의
NUMERICAL_FEATURES = [
    'Precursor_Pulse_Time', 'Co_reactant_Pulse_Time', 'Cycles', 'Temperature',
    'Pressure', 'Purge_Time', 'Purge_Gas_Flow_Rate', 'Aspect_Ratio'
]
CATEGORICAL_FEATURES = ['Precursor', 'Co-reactant', 'Purge Gas']
X_COLS_ORDER = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

TARGET_FEATURES = ['Thickness', 'Step_Coverage', 'Uniformity_Range', 'GPC', 'Density']
TARGET_FEATURES_DISPLAY = {
    'Thickness': 'Thickness (nm)', 
    'Step_Coverage': 'Step Coverage (%)', 
    'Uniformity_Range': 'Uniformity Range (%)', 
    'GPC': 'GPC (A/cycle)', 
    'Density': 'Density (g/cm³)'
}
DEFAULT_PRECURSORS = ['TMA', 'SiH4', 'TiCl4']
DEFAULT_COREACTANTS = ['H2O', 'O2', 'O3 + N2']
DEFAULT_PURGE_GASES = ['N2', 'Ar']


# --- 안전한 NumPy 배열 변환 함수 ---
def to_dense(X):
    if hasattr(X, 'toarray'):
        return X.toarray()
    return X


# --- 1. 데이터 클리닝 및 모델 학습/저장 로직 (함수들) ---

def normalize_col_name(col):
    # 파일명 클리닝 로직 (생략하지 않고 완전하게 포함)
    col_name = re.sub(r'\s*\([^)]*\)', '', col).strip() 
    col_name = re.sub(r'[^a-zA-Z0-9_]', '_', col_name).strip('_')
    
    if 'Precursor' == col_name or 'Precursor_Pulse_Time' in col_name: return 'Precursor_Pulse_Time' if 'Pulse_Time' in col_name else 'Precursor'
    if 'Co_reactant' == col_name or 'Co_reactant_Pulse_Time' in col_name: return 'Co_reactant_Pulse_Time' if 'Pulse_Time' in col_name else 'Co-reactant'
    if 'Purge_Gas' == col_name: return 'Purge Gas'
    if 'Temperature' in col_name: return 'Temperature'
    if 'Pressure' in col_name: return 'Pressure'
    if 'Aspect_Ratio' in col_name: return 'Aspect_Ratio'
    if 'Cycles' in col_name: return 'Cycles'
    if 'Purge_Time' in col_name: return 'Purge_Time'
    if 'Purge_Gas_Flow_Rate' in col_name: return 'Purge_Gas_Flow_Rate'
    if 'Thickness' in col_name: return 'Thickness'
    if 'GPC' in col_name: return 'GPC'
    if 'Density' in col_name: return 'Density'
    if 'Uniformity' in col_name: return 'Uniformity_Raw'
    if 'Step_Coverage' in col_name: return 'Step_Coverage_Raw'
    
    return col_name

def clean_data(df_raw):
    df_raw.columns = [normalize_col_name(col) for col in df_raw.columns]
    df = df_raw.copy()
    
    df = df.drop(columns=['Aspect_Ratio', 'Leakage_Current_Density', 'Dielectric_Constant', 
                          'Breakdown_Field', 'Paper_Title', 'Surface_Roughness', 
                          'Precursor_Flow_Rate', 'Co_reactant_Flow_Rate', 'ID'], errors='ignore')

    cols_to_convert = ['Precursor_Pulse_Time', 'Co_reactant_Pulse_Time', 'Cycles', 'Thickness']
    for col in cols_to_convert:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    def extract_aspect_ratio_from_step_coverage(raw_value):
        if pd.isna(raw_value): return np.nan
        s = str(raw_value)
        if 'AR' in s:
            match = re.search(r'AR[\s=]*([\d\.\s,]+)', s)
            if match:
                number_str = match.group(1).replace(' ', '').replace(',', '')
                numbers = re.findall(r'[\d\.]+', number_str)
                float_numbers = [float(n) for n in numbers if n]
                return max(float_numbers) if float_numbers else np.nan
        return np.nan
    
    if 'Aspect_Ratio' not in df.columns or df['Aspect_Ratio'].isnull().all():
        df['Aspect_Ratio'] = df['Step_Coverage_Raw'].apply(extract_aspect_ratio_from_step_coverage)
    else:
        df['Aspect_Ratio'] = pd.to_numeric(df['Aspect_Ratio'], errors='coerce')
        df['AR_Extracted'] = df['Step_Coverage_Raw'].apply(extract_aspect_ratio_from_step_coverage)
        df['Aspect_Ratio'] = df['Aspect_Ratio'].fillna(df['AR_Extracted'])
        df = df.drop(columns=['AR_Extracted'], errors='ignore')

    def extract_step_coverage_robust(raw_value):
        if pd.isna(raw_value): return np.nan
        s = re.sub(r'[^0-9\.\,\-\s\(\)%]', '', str(raw_value).strip()).replace('%', '')
        numbers = re.findall(r'[\d\.]+', s)
        try:
            float_numbers = [float(n) for n in numbers if n]
            return max(float_numbers) if float_numbers else np.nan
        except:
            return np.nan
    df['Step_Coverage'] = df['Step_Coverage_Raw'].apply(extract_step_coverage_robust)
    
    def extract_uniformity_range(raw_value):
        if pd.isna(raw_value): return np.nan
        s = re.sub(r'[^\d\.]', '', str(raw_value).strip())
        try:
            return float(s)
        except:
            return np.nan
    df['Uniformity_Range'] = df['Uniformity_Raw'].apply(extract_uniformity_range)
    
    df = df.drop(columns=['Uniformity_Raw', 'Step_Coverage_Raw'], errors='ignore')

    df_final = df[X_COLS_ORDER + TARGET_FEATURES].copy()
    df_clean = df_final.dropna(subset=X_COLS_ORDER + TARGET_FEATURES).copy()
    
    return df_clean

@st.cache_resource
def train_and_save_model():
    """모델 학습 및 파일 저장 (Streamlit 캐시 활용)"""
    st.info(f"데이터셋 로드 및 AI 모델 학습을 시작합니다. ({DATA_FILE})")
    
    try:
        df_raw = pd.read_csv(DATA_FILE, encoding='utf-8')
    except:
        df_raw = pd.read_csv(DATA_FILE, encoding='cp949')

    df_clean = clean_data(df_raw)
    
    X_train, X_test, Y_train, Y_test = train_test_split(df_clean[X_COLS_ORDER], df_clean[TARGET_FEATURES], test_size=0.2, random_state=42)

    # 1. X 입력 데이터 전처리 (StandardScaler)
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    preprocessor = ColumnTransformer(
        transformers=[('num', numerical_transformer, NUMERICAL_FEATURES),('cat', categorical_transformer, CATEGORICAL_FEATURES)]
    )
    X_train_processed = preprocessor.fit_transform(X_train)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    # 2. Y 타겟 데이터 스케일링 (MinMaxScaler for 0-1 range)
    Y_scaler = MinMaxScaler()
    Y_train_scaled = Y_scaler.fit_transform(Y_train)
    joblib.dump(Y_scaler, Y_SCALER_PATH)

    # 3. 딥러닝 모델 설계 및 학습
    input_dim = to_dense(X_train_processed).shape[1] 
    output_dim = Y_train.shape[1]
    
    improved_model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.2), Dense(64, activation='relu'), Dropout(0.2), Dense(32, activation='relu'),
        Dense(output_dim, activation='relu') 
    ])
    improved_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # 학습
    improved_model.fit(to_dense(X_train_processed), Y_train_scaled, epochs=150, batch_size=32, validation_split=0.2, verbose=0)
    save_model(improved_model, MODEL_PATH)
    
    st.success(f"✅ AI 모델 학습 완료! (총 {len(df_clean)}개 데이터 사용)")
    return improved_model, preprocessor, Y_scaler

# --- 2. AI 예측 함수 (통합) ---

@st.cache_resource
def load_ai_assets():
    """모델 파일 로드 또는 학습/생성 후 로드"""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH) or not os.path.exists(Y_SCALER_PATH):
        # 파일이 없으면 학습 함수를 호출하여 파일을 생성합니다.
        return train_and_save_model()
    else:
        # 파일이 있으면 로드합니다.
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            preprocessor = joblib.load(PREPROCESSOR_PATH)
            y_scaler = joblib.load(Y_SCALER_PATH)
            return model, preprocessor, y_scaler
        except Exception as e:
            st.error(f"❌ 저장된 모델 로드 실패. 파일을 삭제하고 재실행하세요. 오류: {e}")
            st.stop()

def predict_ald_properties(input_df, model, preprocessor, y_scaler):
    """입력 데이터프레임을 받아 AI 예측을 수행하고 결과를 반환합니다."""
    # X 입력 전처리
    X_processed = preprocessor.transform(input_df[X_COLS_ORDER])
    
    # 예측 수행
    Y_predicted_scaled = model.predict(to_dense(X_processed))[0]
    
    # 결과 역변환 (실제 물리적 단위로 복원)
    Y_predicted_original = y_scaler.inverse_transform(Y_predicted_scaled.reshape(1, -1))[0]
    
    results_df = pd.DataFrame({'특성': list(TARGET_FEATURES_DISPLAY.values()),'예측 값': [f"{val:.4f}" for val in Y_predicted_original]})
    
    return results_df, Y_predicted_original


# --- 3. Streamlit UI 구성 (메인 실행) ---

st.set_page_config(page_title="ALD 공정 AI 예측 시스템", layout="wide")
st.title("🧪 3D 반도체 ALD 공정 AI 예측 시스템")

# 모델 로드/학습 (가장 먼저 실행되며, Streamlit의 캐시 덕분에 한 번만 실행됨)
model, preprocessor, y_scaler = load_ai_assets()

# --- 사이드바: 입력 패널 ---
with st.sidebar:
    st.header("공정 조건 입력 (X)")
    st.markdown("---")

    # 입력 필드
    precursor = st.selectbox("Precursor", DEFAULT_PRECURSORS, index=0)
    co_reactant = st.selectbox("Co-reactant", DEFAULT_COREACTANTS, index=0)
    purge_gas = st.selectbox("Purge Gas", DEFAULT_PURGE_GASES, index=0)
    
    st.markdown("---")
    st.subheader("수치형 변수")
    
    temperature = st.slider("Temperature (℃)", min_value=100, max_value=400, value=300, step=1)
    pressure = st.number_input("Pressure (torr)", min_value=0.01, max_value=10.0, value=0.3, step=0.01)
    aspect_ratio = st.number_input("Aspect Ratio (AR)", min_value=1.0, max_value=500.0, value=10.0, step=1.0)
    
    st.markdown("---")
    
    prec_pulse = st.number_input("Precursor Pulse Time (s)", min_value=0.01, max_value=10.0, value=0.1, step=0.01)
    co_pulse = st.number_input("Co-reactant Pulse Time (s)", min_value=0.01, max_value=10.0, value=0.1, step=0.01)
    cycles = st.number_input("Cycles (n)", min_value=1, max_value=1000, value=500, step=10)
    purge_time = st.number_input("Purge Time (s)", min_value=0.1, max_value=20.0, value=5.0, step=0.1)
    purge_flow = st.number_input("Purge Gas Flow Rate (ccm)", min_value=50.0, max_value=500.0, value=200.0, step=10.0)


# --- 메인 페이지: 예측 결과 출력 ---

# 1. 입력 데이터를 DataFrame으로 통합
input_data_df = pd.DataFrame({
    'Precursor_Pulse_Time': [prec_pulse], 'Co_reactant_Pulse_Time': [co_pulse], 'Cycles': [cycles], 'Temperature': [temperature],
    'Pressure': [pressure], 'Purge_Time': [purge_time], 'Purge_Gas_Flow_Rate': [purge_flow], 'Aspect_Ratio': [aspect_ratio],
    'Precursor': [precursor], 'Co-reactant': [co_reactant], 'Purge Gas': [purge_gas]
})

# 2. 예측 버튼
if st.button("AI 예측 계산 실행", type="primary", use_container_width=True):
    # 예측 함수 호출
    results_df, Y_predicted_original = predict_ald_properties(input_data_df, model, preprocessor, y_scaler)

    if results_df is not None:
        st.subheader("예측된 박막 특성 결과")
        
        col1, col2 = st.columns(2)
        
        # 3. 테이블 출력
        with col1:
            st.dataframe(results_df, hide_index=True, use_container_width=True)
            st.success("✅ AI 예측 계산 완료")

        # 4. Plotly 레이더 차트 출력
        with col2:
            fig = go.Figure(data=[
                go.Scatterpolar(
                    r=Y_predicted_original,
                    theta=list(TARGET_FEATURES_DISPLAY.values()),
                    fill='toself',
                    name='AI 예측 결과'
                )
            ],
            layout=go.Layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, np.max(Y_predicted_original) * 1.2]) 
                ),
                showlegend=False,
                height=450
            ))
            st.plotly_chart(fig, use_container_width=True)
            
        st.markdown("---")
        st.caption("결과 해석: Min-Max Scaling과 ReLU 활성화 함수가 적용되어 GPC/Step Coverage 등은 0 미만으로 예측되지 않습니다. 결과가 0에 가깝다면, 해당 조건은 증착에 비효율적임을 의미합니다.")