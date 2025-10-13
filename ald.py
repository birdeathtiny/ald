import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import re
import warnings

warnings.filterwarnings('ignore')

def load_and_clean_data(file_name):
    """CSV 파일을 불러오고, 숫자 열에 포함된 문자를 제거하는 클리닝 작업을 수행합니다."""
    print("="*50)
    print("### 1단계: 데이터 불러오기 및 클리닝 ###")
    print("="*50)
    
    try:
        df = pd.read_csv(file_name)
        print(f"✅ '{file_name}' 파일에서 데이터를 성공적으로 불러왔습니다.")
    except FileNotFoundError:
        print(f"🚨 '{file_name}' 파일을 찾을 수 없습니다. 코드와 같은 폴더에 파일이 있는지 확인해주세요.")
        return None

    # 숫자여야 하지만 문자가 섞여 있을 수 있는 모든 열의 목록
    numeric_cols_to_clean = [
        'T (oC)', 'P (mTorr)', 'F (sccm)', 'Knudsen Number (Kn)', 
        'Sticking Coefficient (s)', 'Step Coverage (SC, %)', 'Depo Rate (nm/cycle)', 
        'C/H Impurity (at.%)', 'Particle Density (cm-3)'
    ]

    for col in numeric_cols_to_clean:
        if col in df.columns:
            df[col] = df[col].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
            
    print("✨ 데이터 클리닝 완료!")
    return df

def train_model(df):
    """정제된 데이터를 받아 AI 모델을 학습시키고, 학습된 모델을 반환합니다."""
    print("\n" + "="*50)
    print("### 2단계: AI 모델 학습 및 평가 ###")
    print("="*50)
    
    try:
        input_columns = ['T (oC)', 'P (mTorr)', 'F (sccm)', 'T Level', 'Knudsen Number (Kn)', 'Sticking Coefficient (s)']
        output_columns = ['Step Coverage (SC, %)', 'Depo Rate (nm/cycle)', 'C/H Impurity (at.%)', 'Particle Density (cm-3)']
        categorical_features = ['T Level']

        # 학습에 필요한 데이터가 없는 행 제거
        df.dropna(subset=input_columns + output_columns, inplace=True)
        if len(df) < 10: # 학습에 필요한 최소 데이터 수
            print("🚨 유효 데이터가 부족하여 학습을 진행할 수 없습니다.")
            return None, None

        X = df[input_columns]
        Y = df[output_columns]
        
        preprocessor = ColumnTransformer(
            transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
            remainder='passthrough'
        )

        from sklearn.multioutput import MultiOutputRegressor
        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', MultiOutputRegressor(xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)))
        ])

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        print("🚀 모델 학습 시작...")
        model_pipeline.fit(X_train, Y_train)
        print("✨ 모델 학습 완료!")

        # 모델 성능 평가
        predictions = model_pipeline.predict(X_test)
        print("\n--- 모델 성능 평가 (Mean Absolute Error) ---")
        for i, col_name in enumerate(output_columns):
            mae = mean_absolute_error(Y_test.iloc[:, i], predictions[:, i])
            print(f"{col_name} 예측 오차: {mae:.2f}")
        
        return model_pipeline, X # 최적화 탐색을 위해 X 데이터도 함께 반환

    except KeyError as e:
        print(f"🚨 KeyError: CSV 파일에 필요한 열({e})이 없습니다.")
        return None, None
    except Exception as e:
        print(f"🚨 모델 학습 중 오류 발생: {e}")
        return None, None

def find_optimal_conditions(model, original_X, optimization_target, num_simulations=10000):
    """학습된 모델을 이용해 가상 실험을 하고 최적 조건을 탐색합니다."""
    print("\n" + "="*50)
    print("### 3단계: 최적 공정 조건 탐색 (시뮬레이션) ###")
    print("="*50)

    # 시뮬레이션을 위한 가상 데이터 생성
    sim_data = {}
    for col in original_X.columns:
        if col in ['T Level']: # 범주형 데이터 처리
             sim_data[col] = np.random.choice(original_X[col].unique(), num_simulations)
        else: # 숫자형 데이터 처리
            min_val = original_X[col].min()
            max_val = original_X[col].max()
            sim_data[col] = np.random.uniform(min_val, max_val, num_simulations)
    
    simulation_df = pd.DataFrame(sim_data)

    print(f"{num_simulations}개의 가상 조건으로 시뮬레이션 시작...")
    # 가상 조건들로 결과 예측
    sim_predictions = model.predict(simulation_df)

    # 예측 결과를 DataFrame에 추가
    output_columns = ['Step Coverage (SC, %)', 'Depo Rate (nm/cycle)', 'C/H Impurity (at.%)', 'Particle Density (cm-3)']
    for i, col_name in enumerate(output_columns):
        simulation_df[f'predicted_{col_name}'] = sim_predictions[:, i]

    # 목표 변수를 기준으로 최적 조건 정렬
    optimal_conditions = simulation_df.sort_values(by=f'predicted_{optimization_target}', ascending=False)
    
    print("\n✨ AI가 예측한 최적의 공정 조건 ✨")
    print(f"(목표: '{optimization_target}' 최대화)")
    print(optimal_conditions.head())

# --- 메인 실행 부분 ---
if __name__ == "__main__":
    
    # 1단계: 데이터 불러오기 및 클리닝
    file_name = 'HCDS Data sample - HCDS Data Sample.csv'
    cleaned_df = load_and_clean_data(file_name)

    if cleaned_df is not None:
        # 2단계: AI 모델 학습
        trained_model, X_data = train_model(cleaned_df)

        if trained_model is not None:
            # 3단계: 최적 조건 탐색
            # 목표: 'Step Coverage (SC, %)'를 최대로 만드는 조건 찾기
            find_optimal_conditions(trained_model, X_data, optimization_target='Step Coverage (SC, %)')