import pandas as pd
from sklearn.linear_model import LinearRegression
import streamlit as st # streamlit 라이브러리 불러오기

# --- AI 모델 학습 (1, 2단계 내용) ---
# 실제 환경에서는 모델을 매번 학습시키지 않고, 학습된 모델 파일을 불러와서 사용합니다.
# 하지만 여기서는 간단하게 전체 과정을 보여주기 위해 코드를 포함합니다.

def train_model():
    """데이터를 읽고 AI 모델을 학습시키는 함수"""
    try:
        df = pd.read_csv('ald_data.csv')
        X = df[['Temperature', 'Pressure', 'Precursor_A_Pulse', 'Precursor_B_Pulse']]
        Y = df[['Thickness', 'Uniformity']]
        
        model = LinearRegression()
        model.fit(X, Y) # 전체 데이터를 사용해 최종 모델 학습
        return model
    except FileNotFoundError:
        return None

# 모델 학습 실행
model = train_model()

# --- 웹 애플리케이션 UI 구성 (3단계) ---

# st.title() : 웹 페이지의 제목을 설정합니다.
st.title("🤖 ALD 공정 최적화 AI 파트너")

st.write("---") # 구분선

# st.header() : 소제목을 추가합니다.
st.header("🔬 공정 조건을 입력하세요")

# st.sidebar를 사용하면 옆에 사이드바를 만들 수 있습니다. 여기서는 메인 화면에 바로 입력창을 만듭니다.
# st.number_input() : 숫자 입력창을 만듭니다.
temp = st.number_input("온도 (Temperature, °C)", min_value=200, max_value=400, value=330)
pressure = st.number_input("압력 (Pressure, Torr)", min_value=0.0, max_value=1.0, value=0.15, format="%.2f")
pulse_a = st.number_input("Precursor A 펄스 (ms)", min_value=10, max_value=200, value=65)
pulse_b = st.number_input("Precursor B 펄스 (ms)", min_value=10, max_value=200, value=65)

# st.button() : 클릭 가능한 버튼을 만듭니다.
if st.button("결과 예측하기"):
    if model is not None:
        # 사용자가 입력한 값들을 모델이 학습한 데이터 형식(2D 배열)으로 변경
        input_conditions = [[temp, pressure, pulse_a, pulse_b]]
        
        # 모델로 결과 예측
        predicted_properties = model.predict(input_conditions)
        
        thickness = predicted_properties[0][0]
        uniformity = predicted_properties[0][1]
        
        st.write("---")
        st.header("💡 예측 결과")
        
        # st.metric() : 주요 지표를 강조해서 보여주는 UI 요소
        col1, col2 = st.columns(2) # 화면을 2개의 컬럼으로 분할
        col1.metric("🎯 예측 두께 (Thickness)", f"{thickness:.2f} Å")
        col2.metric("✨ 예측 균일도 (Uniformity)", f"{uniformity:.2f} %")
    else:
        # ald_data.csv 파일이 없을 경우 에러 메시지 표시
        st.error("오류: 'ald_data.csv' 파일을 찾을 수 없습니다. app.py와 같은 폴더에 파일이 있는지 확인해주세요.")