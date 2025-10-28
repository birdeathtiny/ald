# 1. 기본 이미지 지정: Python 3.11 환경 사용
FROM python:3.11-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 필수 파일 복사 (캐시 무효화의 시작점)
COPY requirements.txt .
COPY ald_data.csv .

# ⭐ 캐시 무효화 레이어 삽입 ⭐
# 이 레이어는 매번 현재 날짜와 시간으로 새로운 파일을 생성하여,
# 다음 레이어(RUN pip install)가 캐시를 사용하지 않도록 강제합니다.
RUN echo "Trigger rebuild at $(date)" > rebuild.trigger 

# 4. Python 라이브러리 설치 (캐시 무효화 후 실행)
RUN pip install --no-cache-dir -r requirements.txt

# 5. 메인 실행 파일 복사
COPY ald.py .

# 6. 컨테이너 실행 명령어 정의
CMD ["python", "ald.py"]