import os
import sys
import joblib
import pandas as pd
import numpy as np
import re
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import EarlyStopping

# Keras 로드 시 경고 메시지 방지
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- 0. 전역 변수 및 파일 경로 설정 (파일명 일치 확인 완료) ---
DATA_FILE = 'ald_data.csv'  # ⭐ 이 이름으로 파일이 존재해야 합니다.
MODEL_PATH = 'improved_ald_mimo_model.h5'
PREPROCESSOR_PATH = 'ald_preprocessor.joblib'

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

# --- 안전한 NumPy 배열 변환 함수 ---
def to_dense(X):
    """객체가 'toarray' 메서드를 가지면 호출하고, 아니면 그대로 반환하여 오류를 방지합니다."""
    if hasattr(X, 'toarray'):
        return X.toarray()
    return X


# --- 1. 데이터 클리닝 및 전처리 함수 ---

def normalize_col_name(col):
    col_name = re.sub(r'\s*\([^)]*\)', '', col).strip() 
    col_name = re.sub(r'[^a-zA-Z0-9_]', '_', col_name).strip('_')
    
    # 명시적 이름 매핑 (강력한 호환성 확보)
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


# --- 2. AI 모델 학습 및 저장 함수 ---

def train_and_save_model():
    """데이터를 클리닝하고 AI 모델을 학습시킨 후 저장합니다."""
    print("--- 🛠️ AI 모델 학습/저장 프로세스 시작 ---")
    
    try:
        df_raw = pd.read_csv(DATA_FILE, encoding='utf-8')
    except:
        df_raw = pd.read_csv(DATA_FILE, encoding='cp949')

    df_clean = clean_data(df_raw)
    
    X = df_clean[X_COLS_ORDER]
    Y = df_clean[TARGET_FEATURES]
    
    print(f"✅ 데이터 클리닝 완료. 최종 학습 데이터셋 크기: {len(df_clean)}개")

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, NUMERICAL_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ]
    )
    X_train_processed = preprocessor.fit_transform(X_train)
    
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print("✅ 전처리 파이프라인 (.joblib) 저장 완료.")

    input_dim = to_dense(X_train_processed).shape[1] 
    output_dim = Y_train.shape[1]
    
    lr_schedule = ExponentialDecay(0.001, decay_steps=1000, decay_rate=0.96, staircase=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    improved_model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(output_dim, activation='linear')
    ])

    improved_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='mse', 
        metrics=['mae'] 
    )

    print("--- AI 모델 학습 시작 (150 Epochs) ---")
    improved_model.fit(
        to_dense(X_train_processed), 
        Y_train.values,
        epochs=150, 
        batch_size=32,
        validation_split=0.2, 
        callbacks=[early_stopping],
        verbose=0
    )
    print("--- AI 모델 학습 완료 ---")

    save_model(improved_model, MODEL_PATH)
    print(f"✅ 딥러닝 모델 ({MODEL_PATH}) 저장 완료.")
    
    return improved_model, preprocessor


# --- 3. 콘솔 계산 및 출력 함수 ---

def run_single_prediction_test(model, preprocessor):
    """특정 입력에 대한 AI 모델의 예측 결과를 터미널에 출력합니다."""
    
    # --- ⭐ 계산에 사용할 입력 조건 (이 부분을 원하는 값으로 수정하세요) ⭐ ---
    test_data = pd.DataFrame({
        'Precursor_Pulse_Time': [0.1], 
        'Co_reactant_Pulse_Time': [0.1], 
        'Cycles': [500.0], 
        'Temperature': [300],
        'Pressure': [0.3], 
        'Purge_Time': [5.0], 
        'Purge_Gas_Flow_Rate': [200.0], 
        'Aspect_Ratio': [100.0],
        'Precursor': ['TMA'], 
        'Co-reactant': ['H2O'], 
        'Purge Gas': ['N2']
    })
    
    # 전처리 및 예측
    X_test_processed = preprocessor.transform(test_data)
    Y_predicted = model.predict(to_dense(X_test_processed))[0]
    
    # --- 결과 출력 ---
    print("\n" + "="*50)
    print("--- 🧪 ALD 공정 AI 예측 계산 결과 (VS Code 터미널) ---")
    print("="*50)
    print("입력 조건:")
    for key, value in test_data.iloc[0].items():
        print(f"  {key:<25}: {value}")
        
    print("\n\n🔥 예측된 박막 특성 (Y):")
    for i, target in enumerate(TARGET_FEATURES):
        display_name = TARGET_FEATURES_DISPLAY.get(target, target)
        print(f"  {display_name:<25}: {Y_predicted[i]:.4f}")
    print("="*50)

# --- 4. 메인 실행 블록 ---

def load_ai_assets():
    """저장된 AI 모델과 전처리기를 로드"""
    global loaded_model, loaded_preprocessor
    try:
        loaded_model = load_model(MODEL_PATH) 
        loaded_preprocessor = joblib.load(PREPROCESSOR_PATH)
        return loaded_model, loaded_preprocessor
    except Exception as e:
        print(f"❌ 오류: 모델 로드 실패. 파일이 손상되었거나 Keras 비호환성 문제입니다. 에러: {e}")
        return None, None

if __name__ == '__main__':
    loaded_model, loaded_preprocessor = None, None
    
    # 모델 파일이 없으면 학습 및 저장 프로세스를 실행
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
        if not os.path.exists(DATA_FILE):
             print(f"❌ 오류: 학습 데이터 파일 ({DATA_FILE})이 현재 폴더에 없습니다. 파일을 확인하고 다시 시도하세요.")
             sys.exit(1)
        # 학습 및 저장 실행
        loaded_model, loaded_preprocessor = train_and_save_model()
    
    # 파일이 있거나 학습이 완료되었으면 로드 시도
    if loaded_model is None or loaded_preprocessor is None:
        loaded_model, loaded_preprocessor = load_ai_assets()

    # --- 최종 계산 실행 ---
    if loaded_model is not None and loaded_preprocessor is not None:
        run_single_prediction_test(loaded_model, loaded_preprocessor)
    else:
        print("❌ 심각한 오류: AI 모델이 최종적으로 로드되지 않았습니다. 프로그램을 종료합니다.")