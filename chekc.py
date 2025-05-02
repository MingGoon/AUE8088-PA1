import os
import sys
import inspect

# 1) 현재 작업 디렉터리
print("Current working directory:", os.getcwd())

# 2) 이 스크립트 파일의 절대 경로
print("This file is located at  :", os.path.abspath(__file__))

# 3) 모듈로 로드된 경우 (예: import src.network)
import src.network
print("src.network module path:", inspect.getfile(src.network))