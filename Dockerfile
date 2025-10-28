# 1. 기본 이미지 지정: Python 3.11 환경을 가진 경량 리눅스 기반 이미지 사용
FROM python:3.11-slim

# 2. 작업 디렉토리 설정: 컨테이너 내부의 모든 작업이 /app 폴더에서 이루어지도록 지정
WORKDIR /app

# 3. 필수 파일 복사 (라이브러리 목록과 학습 데이터)
# [FIX] 한글 파일명 대신 ald_data.csv를 복사
COPY requirements.txt .
COPY ald_data.csv .

# 4. Python 라이브러리 설치
RUN pip install --no-cache-dir -r requirements.txt

# 5. 메인 실행 파일 복사
COPY ald.py .

# 6. 컨테이너 실행 명령어 정의 (CMD)
# 컨테이너가 시작될 때 자동으로 'python ald.py' 명령을 실행하여 AI 계산을 수행합니다.
CMD ["python", "ald.py"]