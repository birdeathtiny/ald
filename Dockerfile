# 1. 기본 이미지 지정: TensorFlow가 안정적으로 호환되는 Python 3.11 환경을 사용합니다.
FROM python:3.11-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 필수 파일 복사 (라이브러리 목록과 학습 데이터)
COPY requirements.txt .
COPY ald_data.csv .

# 4. Python 라이브러리 설치
# pip이 호환되는 버전(TensorFlow 2.20.0 이상)을 설치할 것입니다.
RUN pip install --no-cache-dir -r requirements.txt

# 5. 메인 실행 파일 복사
COPY ald.py .

# 6. 컨테이너 실행 명령어 정의
CMD ["python", "ald.py"]