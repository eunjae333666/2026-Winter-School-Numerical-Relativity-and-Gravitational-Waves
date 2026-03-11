# 2026 수치상대론 및 중력파 겨울학교

이 프로젝트는 **2026 Numerical Relativity and Gravitational Waves Winter School**에서 진행한 팀 프로젝트 결과물입니다.

## 📝 프로젝트 소개
* **주제**: 선형 분석법을 이용한 중성자 별의 진동 모드 분석
* **주요 성과**: 경진대회 전체 **우승(1위)**
* **문제 확인**: [겨울학교 공식 사이트 링크](https://school.gw.nr.re.kr/2026/winter/)
* **기간**: 2026.02.02 - 2026.02.06

## 📂 폴더 구조
* `source/`: 문제 해결을 위한 메인 파이썬 소스 코드
* `results/`: 시뮬레이션 결과 그래프 및 데이터 파일
* `visualization/`: 시각화 관련 소스 코드(겨울학교 주최 제공)
* `docs/`: 관련 문서

## 💻 개발 환경 및 요구 사항
* **OS**: Ubuntu (Linux) / Windows 11
* **Language**: Python 3.10+
* **Libraries**: `numpy`, `matplotlib`, `scipy`

## 🚀 실행 방법
각 문제는 의존성이 있으므로 순서대로 실행하거나 설명에 따라 코드를 수정해야 합니다.

1. **Problem 1-(1)**
   - `source/problem1.py` 하단 `main()`에서 `n` 값을 원하는 대로 수정 후 실행

2. **Problem 2-(2) & 3-(1)**
   - **중요**: `problem1.py`를 먼저 실행한 뒤 `problem2.py` 또는 `problem3.py`를 실행
   - `main()`의 `mode` 값을 필요에 따라 수정 가능

3. **Problem 3-(4)**
   - `problem1.py` 실행 후 `problem3.py` 실행
   - `main()` 함수에서 `dat_output(p, mode=1)`을 주석 처리하고, 아래의 `for`문 주석을 해제한 뒤 실행

4. **Problem 4-(1)**
   - `source/problem4.py` 하단 `main()`에서 `n` 값을 수정 후 실행

## 👥 팀원
* 박은재 (본인)
    * 프로젝트 코드 구조 설계
    * Shooting Method 구현
    * 데이터 및 그래프 생성 코드 작성
    * 물리 수식 전개

* 이대한
    * Newton-Raphson 알고리즘 구현
    * 물리 수식 전개 및 정리 
    * 제출용 문서 작성
    * 코드 논리 검증 및 레포지토리 관리 보조