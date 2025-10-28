# 1. 기본 이미지 지정
FROM python:3.11-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 필수 파일 복사 (라이브러리 목록과 학습 데이터)
# 두 파일 모두 도커 이미지 빌드 시 /app 경로에 복사됩니다.
COPY requirements.txt .
COPY 파라미터 정리1.csv .

# 4. Python 라이브러리 설치
# requirements.txt에 나열된 모든 패키지를 설치합니다.
RUN pip install --no-cache-dir -r requirements.txt

# 5. 메인 실행 파일 복사
# 메인 AI 실행 파일인 ald.py를 컨테이너에 복사합니다.
COPY ald.py .

# 6. 컨테이너 실행 명령어 정의 (CMD)
# 컨테이너가 실행될 때 자동으로 'python ald.py' 명령을 실행합니다.
# 이 명령어가 AI 계산을 수행하고 결과를 터미널에 출력합니다.
CMD ["python", "ald.py"]