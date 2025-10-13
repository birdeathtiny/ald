import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
import warnings

warnings.filterwarnings('ignore')

# --- 1. 데이터 로딩 및 클리닝 함수 ---
@st.cache_data # 파일 내용이 같으면 캐시된 데이터를 사용해 속도 향상
def load_and_clean_data(uploaded_file):
    """업로드된 CSV 파일을 불러오고 데이터 클리닝을 수행합니다."""
    if uploaded_file is None:
        return None
    
    try:
        df = pd.read_csv(uploaded_file)
        
        numeric_cols_to_clean = [
            'T (oC)', 'P (mTorr)', 'F (sccm)', 'Knudsen Number (Kn)', 
            'Sticking Coefficient (s)', 'Step Coverage (SC, %)', 'Depo Rate (nm/cycle)', 
            'C/H Impurity (at.%)', 'Particle Density (cm-3)'
        ]

        for col in numeric_cols_to_clean:
            if col in df.columns:
                df[col] = df[col].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
        
        return df
    except Exception as e:
        st.error(f"파일 처리 중 오류 발생: {e}")
        return None

# --- 2. 모델 학습 함수 ---
@st.cache_resource # 모델처럼 큰 객체는 리소스 캐시 사용
def train_model(df):
    """정제된 데이터로 AI 모델을 학습시키고, 모델과 평가 결과를 반환합니다."""
    try:
        input_columns = ['T (oC)', 'P (mTorr)', 'F (sccm)', 'T Level', 'Knudsen Number (Kn)', 'Sticking Coefficient (s)']
        output_columns = ['Step Coverage (SC, %)', 'Depo Rate (nm/cycle)', 'C/H Impurity (at.%)', 'Particle Density (cm-3)']
        categorical_features = ['T Level']

        df_cleaned = df.dropna(subset=input_columns + output_columns)
        if len(df_cleaned) < 10:
            st.warning("학습에 필요한 유효 데이터가 부족합니다.")
            return None, None

        X = df_cleaned[input_columns]
        Y = df_cleaned[output_columns]
        
        preprocessor = ColumnTransformer(
            transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
            remainder='passthrough'
        )

        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', MultiOutputRegressor(xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)))
        ])

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        model_pipeline.fit(X_train, Y_train)
        
        # 모델 성능 평가
        predictions = model_pipeline.predict(X_test)
        evaluation_metrics = {}
        for i, col_name in enumerate(output_columns):
            mae = mean_absolute_error(Y_test.iloc[:, i], predictions[:, i])
            evaluation_metrics[col_name] = mae
            
        return model_pipeline, X, evaluation_metrics

    except KeyError as e:
        st.error(f"KeyError: CSV 파일에 필요한 열({e})이 없습니다.")
        return None, None
    except Exception as e:
        st.error(f"모델 학습 중 오류 발생: {e}")
        return None, None

# --- 3. 최적 조건 탐색 함수 ---
def find_optimal_conditions(model, original_X, optimization_target, num_simulations):
    """학습된 모델을 이용해 최적 조건을 탐색합니다."""
    sim_data = {}
    for col in original_X.columns:
        if col in ['T Level']:
            sim_data[col] = np.random.choice(original_X[col].unique(), num_simulations)
        else:
            min_val, max_val = original_X[col].min(), original_X[col].max()
            sim_data[col] = np.random.uniform(min_val, max_val, num_simulations)
    
    simulation_df = pd.DataFrame(sim_data)
    sim_predictions = model.predict(simulation_df)

    output_columns = ['Step Coverage (SC, %)', 'Depo Rate (nm/cycle)', 'C/H Impurity (at.%)', 'Particle Density (cm-3)']
    for i, col_name in enumerate(output_columns):
        simulation_df[f'predicted_{col_name}'] = sim_predictions[:, i]

    # 최적화 목표에 따라 정렬 방향 결정
    # 불순물, 파티클 밀도는 낮을수록 좋음
    ascending_order = optimization_target in ['C/H Impurity (at.%)', 'Particle Density (cm-3)']
    optimal_conditions = simulation_df.sort_values(by=f'predicted_{optimization_target}', ascending=ascending_order)
    
    return optimal_conditions

# --- Streamlit 웹 UI 구성 ---
st.set_page_config(layout="wide")
st.title("🏭 ALD 공정 최적화 AI 시뮬레이터")

# 1. 파일 업로드
uploaded_file = st.file_uploader("CSV 파일을 업로드하세요.", type="csv")

if uploaded_file:
    df = load_and_clean_data(uploaded_file)
    
    if df is not None:
        st.success("✅ 데이터 로딩 및 정제 완료!")
        st.dataframe(df.head())

        # 2. 모델 학습
        model, X_data, eval_metrics = train_model(df)

        if model:
            st.success("✅ AI 모델 학습 완료!")
            with st.expander("모델 성능 평가 결과 보기 (평균 절대 오차)"):
                st.json({k: round(v, 4) for k, v in eval_metrics.items()})

            # 3. 최적화 조건 설정 및 실행
            st.header("⚙️ 최적 공정 조건 탐색")
            optimization_target = st.selectbox(
                "어떤 값을 최적화하시겠습니까?",
                ('Step Coverage (SC, %)', 'Depo Rate (nm/cycle)', 'C/H Impurity (at.%)', 'Particle Density (cm-3)'),
                help="Step Coverage와 Depo Rate는 최대화, Impurity와 Particle Density는 최소화하는 조건을 찾습니다."
            )
            num_simulations = st.slider("시뮬레이션 횟수", 1000, 50000, 10000, 1000)

            if st.button("🚀 최적 조건 탐색 시작!"):
                with st.spinner("AI가 수만 개의 가상 조건으로 시뮬레이션 중입니다..."):
                    optimal_df = find_optimal_conditions(model, X_data, optimization_target, num_simulations)
                
                st.success("✨ 최적 조건 탐색 완료!")
                st.dataframe(optimal_df.head(10))