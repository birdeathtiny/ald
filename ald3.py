import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings('ignore')

# --- 1. 데이터 준비 및 가상 시간 계산 ---
try:
    file_name = 'HCDS Data sample - HCDS Data Sample.csv'
    df = pd.read_csv(file_name, encoding='cp949')

    # 데이터 정제
    if 'T Level' in df.columns:
        df = df.drop(columns=['T Level'])
    for col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.extract(r'(\d+\.?\d*)', expand=False), errors='coerce')
    df.dropna(inplace=True)

    # 가상 관계식으로 시간 계산
    assumed_avg_pulse, assumed_avg_purge = 2.0, 4.0
    avg_depo_rate = df['Depo Rate (nm/cycle)'].mean()
    k = avg_depo_rate * (assumed_avg_pulse + assumed_avg_purge)
    
    total_cycle_time = k / (df['Depo Rate (nm/cycle)'] + 1e-9)
    random_ratios = np.random.uniform(1.5, 3.0, len(df))
    
    df['pulse_time'] = total_cycle_time / (1 + random_ratios)
    df['purge_time'] = total_cycle_time - df['pulse_time']
    
    # 시간과 온도의 '상호작용' 관계 생성
    temp_min, temp_max = df['T (oC)'].min(), df['T (oC)'].max()
    ideal_pulse_time = 5.0 - ((df['T (oC)'] - temp_min) / (temp_max - temp_min)) * 3.0
    penalty = 1 - 0.03 * abs(df['pulse_time'] - ideal_pulse_time)
    df['Step Coverage (SC, %)'] = df['Step Coverage (SC, %)'] * penalty.clip(0.97, 1.0)
    
    print("--- 데이터 준비 및 변수 간 상호작용 생성 완료 ---")
    
    # --- 2. 사용자로부터 직접 시간 입력받기 ---
    while True:
        try:
            user_pulse_time = float(input("원하는 Pulse Time(초)을 입력하세요: "))
            user_purge_time = float(input("원하는 Purge Time(초)을 입력하세요: "))
            break
        except ValueError:
            print("오류: 숫자를 입력해주세요.")

    # --- 3. AI 모델 학습 ---
    process_conditions = [
        'T (oC)', 'P (mTorr)', 'F (sccm)', 
        'Knudsen Number (Kn)', # <--- 'Knuden'을 'Knudsen'으로 수정
        'Sticking Coefficient (s)', 'pulse_time', 'purge_time'
    ]
    target = 'Step Coverage (SC, %)'
    X = df[process_conditions]
    y = df[target]
    model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
    print("\n--- AI 모델 학습 완료 ---")

    # --- 4. 최적 공정 조건 탐색 ---
    print(f"\n--- Pulse: {user_pulse_time}s, Purge: {user_purge_time}s 조건에서 최적 레시피 탐색 ---")

    p_range = np.linspace(X['P (mTorr)'].min(), X['P (mTorr)'].max(), 15)
    t_range = np.linspace(X['T (oC)'].min(), X['T (oC)'].max(), 15)
    f_range = np.linspace(X['F (sccm)'].min(), X['F (sccm)'].max(), 15)

    best_score = -1
    best_params = {}
    
    fixed_params = X.mean().to_dict()

    for p in p_range:
        for t in t_range:
            for f in f_range:
                params = fixed_params.copy()
                params.update({
                    'P (mTorr)': p, 'T (oC)': t, 'F (sccm)': f,
                    'pulse_time': user_pulse_time, 'purge_time': user_purge_time
                })
                
                input_df = pd.DataFrame([params])
                predicted_sc = model.predict(input_df[process_conditions])[0]
                
                if predicted_sc > best_score:
                    best_score = predicted_sc
                    best_params = params
    
    # --- 5. 최종 결과 출력 ---
    print("\n" + "="*50)
    print("     AI가 추천하는 최적 공정 레시피 (맞춤 조건)")
    print("="*50)
    print(f"\n> 최대 예상 Step Coverage: {best_score:.2f} %")
    print(f"\n> 입력된 고정 조건:")
    print(f"  - **pulse_time: {user_pulse_time:.2f}**")
    print(f"  - **purge_time: {user_purge_time:.2f}**")
    print("\n> 추천 최적 조건:")
    for param, value in best_params.items():
        if param not in ['pulse_time', 'purge_time']:
            print(f"  - {param}: {value:.2f}")

except FileNotFoundError:
    print(f"오류: '{file_name}' 파일을 찾을 수 없습니다.")
except Exception as e:
    print(f"데이터 처리 중 오류가 발생했습니다: {e}")