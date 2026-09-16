# SafeWave-AI · M1 낙상감지 학습 레포

> **이 레포의 정식 입력은 `(B,5,64,100)` — 5노드 조기융합이다.
> 과거 per-node `(B,1,64,100)` 전제의 문서·체크포인트·ONNX는 전부 레거시다.**
> (전환일 2026-09-16, 브랜치 `feature/m1-5node` → `chore/refactor-5node`)

이 레포는 자체 수집 CSI로 M1(낙상) 모델을 학습하고 `m1_wifi_pose.onnx`를 만들어
**rp5 추론 레포**(`jinsan02/safewave-ai-ambient-monitoring`)에 넘긴다.

---

## 절대 규칙 (어기면 배포가 조용히 부서진다)

| # | 규칙 | 이유 |
|---|---|---|
| 1 | 입력 텐서는 **`(B,5,64,100)`** — 노드 5 × 서브캐리어 64 × 100프레임 | 서브캐리어는 **64 유지**, 노드는 **별도 축**(conv 입력채널)으로 쌓는다. 서브캐리어축 **192 concat은 금지**. `config.py`가 로드 시 64·100을 강제하고 `n_nodes`는 1~8만 허용 |
| 2 | **ONNX에 Sigmoid를 넣지 마라** | rp5가 `_infer_onnx`에서 sigmoid를 한다. 넣으면 **이중 적용** → 출력이 0.5~0.73으로 뭉개져 임계 0.7이 무의미해진다 |
| 3 | 모델 출력은 **`fall_logit (B,1)` pre-sigmoid** | 학습은 `BCEWithLogitsLoss`가 내부에서 sigmoid 한다. 모델에 sigmoid를 붙이면 학습이 깨진다 |
| 4 | `CSI_FS = 100` 고정 | ESP 송신 rate == rp5 `CSI_FS` == M2 학습 fs, 3자 결속. 하나만 바꾸면 호흡·심박 대역이 전부 틀어진다 |
| 5 | 정규화는 **per-frame peak** 만 | ESP가 패킷 시점에 이미 한다. 재정규화·절대진폭 복원 금지. 5노드에서도 **노드별·프레임별 독립** 적용 (노드 간 공통 정규화 금지) |
| 6 | 체크포인트 로드는 **`strict=True`** | `strict=False`는 키가 하나도 안 맞아도 조용히 성공한다 → 스크래치가 돌면서 "사전학습 효과 없음"이라는 오진에 도달 |
| 7 | **M1이 5노드를 직접 조기융합(early fusion)한다** | 노드축 = 첫 conv의 입력채널. `dataset.py`가 `stream_id` epoch 그리드로 5노드를 `(5,64,100)`으로 정렬해 한 창으로 만든다. (예전 "M1은 per-node, 융합은 M5" 규칙은 **폐기**) |
| 8 | ONNX opset **17**, input `csi_data` | rp5 I/O 계약 |
| 9 | 5노드 시간 정렬은 **`stream_id`의 epoch** 기준 | `stream_id`("1781165460004-0")의 앞부분 = rp5 수신 시각 = **유일한 공통 시계**. `ts_ms`는 보드마다 다른 장치 시계(32비트 랩)라 노드 간 정렬에 쓰면 어긋난다 |

---

## 5노드 입력이 만들어지는 방식 (`dataset.py`)

```
CSV 한 세션 (stream_id, node_id, ts_ms, raw_0..63)
  → 행마다 epoch = stream_id 앞부분
  → 세션 전체를 공통 100Hz epoch 그리드로 (10ms 간격)
  → 노드별 프레임을 최근접 슬롯(±5ms)에 매핑, 충돌 시 더 가까운 쪽
  → 결측 노드 슬롯은 0으로 채움          → (T_grid, 5, 64)
  → 노드·프레임별 per-frame peak 정규화 + guard 서브캐리어 0
  → 창 100프레임 (stride 25, 낙상 근처 dense stride 5)
  → 창 텐서 (5, 64, 100)                → 배치 시 (B, 5, 64, 100)
```

**낙상 라벨은 epoch 도메인**이다 (`mark_falls.py`가 epoch ms로 찍는다).
`fall_ts_ms` 값이 **2^32 이상이면 epoch**, 미만이면 레거시 `ts_ms` 도메인
(`verify_sync_v2.py` 출력)으로 판별해 파일별 오프셋 `median(epoch - ts_ms)`로
자동 변환한다. 두 도메인은 크기가 약 400배 차이라 오판 여지가 없다.

---

## 함정 — 실제로 겪은 것

**GitHub `main` 브랜치는 stale하다.** rp5의 실제 작업은 `develop` / `feature/*`에 있다.
main만 보고 세운 가정이 대부분 틀렸다:

| main만 보고 내린 판단 | develop 실제 (2026-08-25 정찰 시점) |
|---|---|
| 192는 레거시 PulseFi 잔재 | **3노드 × 64 concat**, 의도된 설계 |
| `ai/main.py`에 롤링 버퍼를 새로 만들어야 함 | **이미 구현돼 있음** (노드별 deque, 150ms 초과 시 리셋) |
| `_preprocess`의 192를 64로 고쳐야 함 | **ONNX에서 shape을 직접 읽음** |
| Redis `csi:raw`는 `data` 단일 필드 | `data_raw` / `data_resp` / `data_heart` **3필드** |

> ⚠️ **위 오른쪽 열은 2026-08-25 정찰 기록이고, 이번 5노드 전환 이후 재확인하지 않았다 — 미검증.**
> rp5가 `(B,5,64,100)`을 어떻게 받을지(전처리·버퍼·노드 정렬)는 **이 레포에서 확인할 수 없다.**
> → **rp5 코드를 논할 때는 반드시 `develop`을 직접 읽어라. 추측 금지.**

---

## 검증된 사실

### 5노드 전환 (2026-09-16 실측)

```
model     입력 (4,5,64,100) -> 출력 (4,1) · 학습 파라미터 135,137
          (= 단일노드 134,561 + conv1 입력채널 1→5 확장분 576)
dataset   합성 5노드 2분 데이터로 (N,5,64,100) 창 생성 확인
          레거시 ts_ms 도메인 라벨 자동 변환 경로도 함께 확인
train     2 epoch 완주 · best.pt 저장 · strict=True 재로드 왕복 성공
          5노드 ckpt를 1노드 모델에 넣으면 size mismatch로 거부됨 (규칙 6 의도대로)
```

> 위는 **합성 데이터** 스모크다. 실측 CSI로는 아직 학습하지 않았다.

### 레거시 단일노드 아티팩트 (2026-08-25 측정, 2026-09-16 재확인)

**`runs/legacy_singlenode/best_B2_final.pt`** — ⚠️ **단일노드. 5노드 모델에 로드 불가.**
```
27 keys · epoch 14 · cfg=configs/m1_fall_pretrain.yaml
conv1.weight = (16, 1, 3, 3)  ← 입력채널 1  (5노드는 (16,5,3,3) 필요 → strict 실패)
pr_auc 0.6961 | fall_recall 0.6209 | precision 0.5519 | f1 0.5844 | train_loss 0.1319
학습 파라미터 134,561  (텐서 총합 134,788 = +BN 버퍼 224 +counter 3)
```

**`runs/legacy_singlenode/m1_wifi_pose.onnx`** — ⚠️ **단일노드 입력.**
```
opset 17 · INPUT csi_data [batch,1,64,100] · OUTPUT fall_logit [batch,1]
Sigmoid 노드 없음 (의도된 것 — 규칙 2 참조)
parity vs best_B2_final.pt : max|torch-onnx| = 2.861e-06  → 동일 체크포인트
```

→ 5노드는 **스크래치 재학습이 정상 경로**다. 이 둘로 워밍스타트하려 하지 마라
(`config.py`가 `pretrain` + `n_nodes != 1` 조합을 명시적으로 거부한다).

### 04 rp5 실측 (2026-08-02, ESP 1노드, 임계 0.7, 0.6초 간격)

| 조건 | n | p50 | ≥0.7 | 3연속 발동 |
|---|---|---|---|---|
| 빈 방 안정구간 | 561 | 0.001 | 0.18% | 0회 |
| 가만히 앉기 | 100 | 0.001 | 0% | 0회 |
| 천천히 걷기 | 100 | 0.153 | 32% | 5회/60초 |
| **빠르게 앉기** | 100 | **0.757** | **54%** | **10회/60초** |

→ 모델이 낙상이 아니라 **모션 강도**를 보고 있다. 원인은 02 어댑터에서 낙상 시각 추정에
실패해 **5초 세그먼트 전체를 양성**으로 라벨링한 것. 임계값·연속성 규칙으로는 안 풀린다.
**05 하드 네거티브 수집이 유일한 해법.**

`fall_detected` 임계 역산: True 최소 0.7035 / False 최대 0.6993 → 0.7 적용 확인.
baseline CSV에 exact 0.0/1.0이 **0건** → **측정 당시 sigmoid는 이미 적용돼 있었다.**

> 이 측정은 **단일노드 모델**의 것이다. 5노드 모델의 동작은 아직 측정되지 않았다.

---

## 레포 구조

```
configs/m1_fall.yaml           본 학습용 (n_nodes 5, fall_stride 5, primary_metric fall_recall)
configs/m1_fall_pretrain.yaml  사전학습용 — 레거시 단일노드(n_nodes 미지정 = 1)
data/labels.csv                학습이 읽는 라벨 — ⚠️ 현재 1002행(csibench 994 + 자체 8)이고
                               전부 단일노드 시절 것이다. stream_id가 없어 5노드 로더에
                               못 들어간다. 수집 후 교체 필요
data/labels_csibench.csv       공개 데이터 라벨 원본 994행 (단일노드 시절)
data/labels_self.csv           자체 수집 — 헤더만 (5노드 수집 후 채워진다)
data/labels_self_legacy.csv    6/11 단일노드 시절 NORMAL 8세션 (보관용)
data/node_layout.csv           노드 좌표 (수집 기록)
data/session_log.csv           세션 기록 (수집 기록)
data/raw/                      원본 CSI CSV — git 제외, 현재 비어 있음
src/m1_fall/{config,dataset,model,train,export}.py
scripts/mark_falls.py          낙상 시각 마킹 — epoch ms (표준 라이브러리만)
scripts/verify_sync_v2.py      노드별 동기·샘플레이트 검증 (표준 라이브러리만)
scripts/session_report.py      수집 기록 검증 + 노드 커버리지 (표준 라이브러리만)
docs/COLLECTION_5NODE.md       5노드 현장 체크리스트 (현행)
docs/REFACTOR_5NODE.md         5노드 전환 정리 + 사람이 결정할 것
docs/archive/                  2026-08-25 시점 기록 (현재 지침 아님)
runs/                          체크포인트·ONNX — git 제외
runs/legacy_singlenode/        단일노드 .pt/.onnx — 5노드에선 로드 불가, 이력 보관
runs/rp5_patch/                rp5로 넘길 패치 번들 (export.py가 생성)
```

`scripts/` 의 3개는 **rp5에서도 돌아가야 하므로 표준 라이브러리만 쓴다.** 의존성 추가 금지.

---

## 자주 쓰는 명령

로컬 파이썬은 레포 안 `.venv` (3.11, torch 2.14 CPU).
**시스템 3.14는 numpy DLL이 깨져 있으니 쓰지 마라.**

```powershell
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe -m m1_fall.model                          # shape + 파라미터 수
.venv\Scripts\python.exe -m m1_fall.dataset configs/m1_fall.yaml   # split별 창 수 + 노드 커버리지
.venv\Scripts\python.exe -m m1_fall.train --config configs/m1_fall.yaml
.venv\Scripts\python.exe -m m1_fall.export --config configs/m1_fall.yaml --ckpt runs/best.pt
.venv\Scripts\python.exe scripts\session_report.py                 # 수집 기록 검증
```

**rp5 접속:** `ssh csi@192.168.1.2` (Debian 13, 키 인증). ESP는 `192.168.1.14`, SSID `TESTAP`.
