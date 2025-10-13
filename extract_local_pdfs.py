import os
import glob
import fitz  # PyMuPDF
import re
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import warnings

warnings.filterwarnings('ignore')

def extract_data_from_local_pdfs(folder_path):
    """지정된 폴더의 PDF 파일에서 공정 변수 데이터를 추출하여 DataFrame으로 반환합니다."""
    print("="*50)
    print("### Part 1: 로컬 PDF에서 데이터 추출 시작 ###")
    print("="*50)
    
    pdf_files = glob.glob(os.path.join(folder_path, '*.pdf'))
    if not pdf_files:
        print("❌ 분석할 PDF 파일이 없습니다.")
        return None

    final_data = []
    for i, pdf_path in enumerate(pdf_files):
        file_name = os.path.basename(pdf_path)
        print(f"[{i+1}/{len(pdf_files)}] Processing: {file_name}...")
        try:
            full_text = ""
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    full_text += page.get_text()

            text_lower = full_text.lower()
            found_materials = []
            if 'al2o3' in text_lower or 'alumina' in text_lower: found_materials.append('Al2O3')
            if 'hfo2' in text_lower or 'hafnia' in text_lower: found_materials.append('HfO2')
            if 'tin' in text_lower or 'titanium nitride' in text_lower: found_materials.append('TiN')

            if found_materials:
                material_str = ", ".join(list(set(found_materials)))
                temperatures = re.findall(r'(\d+\.?\d*)\s*°?C\b', full_text, re.IGNORECASE)
                cycles = re.findall(r'(\d+)\s*(cycles?)\b', full_text, re.IGNORECASE)
                
                # 온도와 사이클 데이터가 모두 존재할 때만 의미있는 데이터로 간주
                if temperatures and cycles:
                    final_data.append({
                        'source_file': file_name,
                        'material': material_str,
                        'temperature_C': float(temperatures[0]),
                        'cycles': int(cycles[0][0])
                    })
                    print("  -> ✅ 데이터 추출 성공!")

        except Exception as e:
            print(f"  -> ❌ 파일 처리 중 오류 발생: {e}")
            
    if not final_data:
        return None
        
    return pd.DataFrame(final_data)

def train_ai_model(df):
    """추출된 데이터(DataFrame)를 받아 AI 모델을 학습하고 평가합니다."""
    print("\n" + "="*50)
    print("### Part 2: AI 모델 학습 및 평가 시작 ###")
    print("="*50)
    
    print("--- 원본 데이터 ---")
    print(df)
    
    # 1. 데이터 전처리 (결측치가 없는 깨끗한 데이터만 사용)
    df_clean = df.dropna()
    if len(df_clean) < 5: # 학습에 필요한 최소 데이터 수를 5개로 가정
        print("\n❌ 학습에 사용할 유효 데이터가 너무 적어 모델을 학습할 수 없습니다.")
        return

    print(f"\n총 {len(df_clean)}개의 유효 데이터로 학습을 시작합니다.")

    # 2. 학습용 데이터 분리
    # 입력(X): 공정 조건 (온도)
    X = df_clean[['temperature_C']]
    # 목표(Y): 예측하고 싶은 결과 (사이클 수)
    Y = df_clean['cycles']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # 3. XGBoost 모델 생성 및 학습
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
    
    print("\n🚀 XGBoost 모델 학습 중...")
    model.fit(X_train, Y_train)
    print("✨ 모델 학습 완료!")

    # 4. 모델 성능 평가
    predictions = model.predict(X_test)
    mae = mean_absolute_error(Y_test, predictions)
    print(f"\n📈 모델 예측 평균 절대 오차 (MAE): {mae:.2f} cycles")
    print("MAE는 실제값과 예측값의 평균적인 차이를 의미하며, 0에 가까울수록 좋습니다.")
    
    # 5. 간단한 예측 예시
    sample_temp = 200 # 200°C일 때 예상되는 사이클 수 예측
    predicted_cycles = model.predict([[sample_temp]])
    print(f"\n💡 예측 예시: 온도가 {sample_temp}°C일 때, 예상되는 사이클 수는 약 {predicted_cycles[0]:.0f}회 입니다.")


# --- 메인 실행 부분 ---
if __name__ == "__main__":
    # 1. 현재 폴더('.')에서 PDF를 읽어 데이터 추출
    extracted_df = extract_data_from_local_pdfs('.')

    # 2. 추출된 데이터가 있으면 AI 모델 학습 진행
    if extracted_df is not None and not extracted_df.empty:
        train_ai_model(extracted_df)
    else:
        print("\n--- 최종적으로 추출된 데이터가 없어 모델 학습을 진행할 수 없습니다. ---")