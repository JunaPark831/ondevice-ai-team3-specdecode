# vast.ai 3090 측정 — 따라 하기

이 폴더 파일:
| 파일 | 하는 일 |
|---|---|
| `run_3090.sh` | 전부 자동: 저장소 받기 → GPU 수정 적용 → 설치 → 모델 → 스펙 카드 → 측정 → 요약 → 묶기 |
| `lang_cuda.patch` | `lang.py`를 맥(MPS)과 GPU(CUDA) 둘 다에서 돌게 고친 것. 측정 로직은 그대로 |
| `gpu_env.py` | 기기 스펙 카드 + 실효 메모리 대역폭 실측 |
| `summarize_3090.py` | M5 Pro와 3090 결과를 같은 표로. W04 표의 speedup·α·τ를 똑같이 재현함을 확인함 |
| `W05-3090-보고서.md` | 보고서 틀. **2절(예측)은 측정 전에 읽고 본인 말로 고쳐서 작성 시각을 적어 둘 것** |

## 0. 측정 전에 (5분)

`W05-3090-보고서.md` 2절을 읽고, 이해한 대로 고쳐 쓰고, 작성 시각을 적는다.

## 1. 매물 고르기 (cloud.vast.ai/create)

- `#GPUs: 1X`, GPU **RTX 3090** (24GB짜리)
- **Max CUDA 13.0 이상** (torch 2.14.1이 CUDA 13.0을 씀. 낮으면 3단계에서 멈춤)
- Disk **80GB** (나중에 못 바꿈. 모델 21GB + 패키지)
- **On-Demand** (Interruptible 금지 — 중간에 끊기면 조건이 바뀜)
- Reliability 99% 이상
- 가능하면 CPU가 `16/16`처럼 **분자=분모**인 단독 매물. 이번 예측은 "CPU 쪽 고정 오버헤드"가 핵심이라 CPU를 남과 나눠 쓰면 숫자가 흔들린다. 공유 매물을 쓰면 그 사실을 보고서에 적는다
- 템플릿: PyTorch (CUDA) 계열 아무거나 (스크립트가 파이썬 환경을 따로 만든다)

## 2. 접속 → 파일 올리기 → 실행

vast.ai `Instances`에서 SSH 명령(포트·주소)을 복사한다. 내 컴퓨터 터미널에서:

```bash
cd ~/Documents/University/2026-2_Semester/In_Depth
scp -P <포트> -r w05-3090 root@<주소>:/workspace/
ssh -p <포트> root@<주소>
```

인스턴스 안에서:

```bash
tmux new -s run                 # 노트북을 덮거나 인터넷이 끊겨도 계속 돌게
bash /workspace/w05-3090/run_3090.sh
```

- 연결이 끊기면 다시 `ssh` 접속 후 `tmux attach -t run`
- 중간에 멈췄으면 그냥 다시 `bash /workspace/w05-3090/run_3090.sh` — 끝난 부분은 건너뛴다
- (선택) 마지막에 GPU 클럭 고정이 되는지도 보려면: `CLOCK_CHECK=1 bash /workspace/w05-3090/run_3090.sh` — compute sweep 가능 여부(팀의 미해결 질문)를 알려 준다. 끝나면 자동 원복

예상 시간 🔶: 설치·모델 다운로드 20~40분(호스트 네트워크 속도에 따라), 측정 15~30분.

## 3. 결과 받기 → 인스턴스 삭제

스크립트 끝에 나온 `scp` 명령을 **내 컴퓨터 터미널**에서 실행해 `result_3090_*.tar.gz`를 받는다. 그다음 vast.ai 콘솔에서 **Delete** (Stop은 디스크 요금이 계속 나감).

## 4. 보고서 채우기

`summary_3090.md`의 표를 `W05-3090-보고서.md` 4절에 붙이고, 5절 판정을 쓴다. 결과가 예측과 달라도 그대로 쓰고 이유를 찾는다.

## 막히면

- `CUDA 안 됨` → 드라이버가 낮음. 인스턴스 Delete 후 Max CUDA 13.0 이상 매물로
- `git clone` 실패 → 저장소가 비공개로 바뀌었을 수 있음. 준하 님께 확인
- `CUDA out of memory` → 다른 프로세스가 GPU를 쓰는지 `nvidia-smi` 확인. 24GB 미만 카드인지 확인
- 그 외: 화면에 나온 마지막 30줄을 그대로 복사해서 물어보기
