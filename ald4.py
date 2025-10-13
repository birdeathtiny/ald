import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import warnings

warnings.filterwarnings('ignore')

# --- 1. 실제 데이터 로드 및 정제 ---
try:
    file_name = 'HCDS Data sample - HCDS Data Sample.csv'
    df = pd.read_csv(file_name, encoding='cp949')
    
    # 데이터 정제 (이전 단계에서 확정된 로직)
    if 'T Level' in df.columns:
        df = df.drop(columns=['T Level'])
    for col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.extract(r'(\d+\.?\d*)', expand=False), errors='coerce')
    df.dropna(inplace=True)
    
    print("--- 'HCDS Data sample - HCDS Data Sample.csv' 파일 로드 및 정제 완료 ---")

    # AI 모델 학습: Depo Rate를 예측하도록 설정
    features = ['T (oC)', 'P (mTorr)', 'F (sccm)']
    target = 'Depo Rate (nm/cycle)'
    X = df[features]
    y = df[target]
    model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
    print("--- 실제 데이터 기반 AI 모델 학습 완료 ---")

    # --- 2. 사용자 목표값 입력 ---
    target_depth = float(input("목표 Deposition Rate (nm/cycle)를 입력하세요: "))

    # --- 3. 베이즈 최적화를 위한 설정 ---
    # 탐색할 변수들의 범위를 정의합니다.
    search_space = [
        Real(X['T (oC)'].min(), X['T (oC)'].max(), name='T (oC)'),
        Real(X['P (mTorr)'].min(), X['P (mTorr)'].max(), name='P (mTorr)'),
        Real(X['F (sccm)'].min(), X['F (sccm)'].max(), name='F (sccm)')
    ]

    # 최적화할 목적 함수 정의
    @use_named_args(search_space)
    def objective_function(**params):
        input_df = pd.DataFrame([params])
        predicted_depth = model.predict(input_df)[0]
        return (predicted_depth - target_depth)**2

    # --- 4. 베이즈 최적화 실행 ---
    print("\n--- 베이즈 최적화 시작 (최적 조건 탐색 중...) ---")
    result = gp_minimize(
        func=objective_function,
        dimensions=search_space,
        n_calls=50, # 50번의 지능적인 탐색 시도
        random_state=42
    )
    
    # --- 5. 최종 결과 출력 ---
    optimal_conditions = result.x
    final_predicted_depth = np.sqrt(result.fun) + target_depth

    print("\n" + "="*50)
    print("     AI가 추천하는 최적 공정 레시피 (베이즈 최적화)")
    print("="*50)
    print(f"\n> 사용자가 요청한 목표 Depo Rate: {target_depth:.4f} nm/cycle")
    print(f"> AI가 찾은 최적 조건에서의 예상 Depo Rate: {final_predicted_depth:.4f} nm/cycle")
    print("\n> 추천 최적 조건 (정밀):")
    for dim, value in zip(search_space, optimal_conditions):
        print(f"  - **{dim.name}: {value:.2f}**")

except FileNotFoundError:
    print(f"오류: '{file_name}' 파일을 찾을 수 없습니다.")
except Exception as e:
    print(f"데이터 처리 중 오류가 발생했습니다: {e}")