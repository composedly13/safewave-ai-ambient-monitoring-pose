# 5노드 수집 체크리스트 (M1 낙상)

현장에서 위에서 아래로 따라간다. **2단계 스모크를 통과하지 못하면 본 수집을 시작하지 마라.**
6.6시간을 모은 뒤 한 노드가 60Hz였다는 걸 발견하면 그 세션 전체가 날아간다.

---

## 이 문서가 있는 이유

M1은 **5노드를 한 입력으로 조기융합**한다 — 창 하나가 `(5, 64, 100)`이고,
노드축은 첫 conv의 입력채널이다. 낙상은 **세션 단위 사건**이라 `fall_ts_ms` 하나가
그 시각의 **창 전체**(5노드가 한 덩어리)에 양성으로 붙는다.
벽 너머에 있어 신호가 없는 노드도 같은 창 안에 그대로 들어간다.

M2에는 이걸 거르는 **SNR 6dB 게이트**가 있다 (`src/m2_vital/snr.py`).
**M1에는 없다.** 조기융합이라 모델이 노드를 알아서 가중하긴 하지만, 그 가중이
맞았는지 되짚으려면 **노드 배치와 낙상 지점이 수집 시점에 기록**돼 있어야 한다.
기록이 없으면 "어느 노드가 실제로 기여했는가"를 사후에 알 방법이 없다.

기록은 `data/node_layout.csv` + `data/session_log.csv` 두 파일에 남기고,
`scripts/session_report.py`가 검증한다.

---

## 0. 출발 전 (책상에서)

| | 항목 | 확인 |
|---|---|---|
| ☐ | **ESP 5대 전부 재플래시** — 04에서 1대만 `TARGET_IP`를 고쳤다. 나머지 4대는 팀원 PC(`192.168.1.11`)를 향하고 있을 수 있다 | `src/config.h` → `TARGET_IP "192.168.1.2"` |
| ☐ | 노드별 env로 굽고 **보드에 번호 라벨** | `pio run -e nodeN -t upload --upload-port COMx` |
| ☐ | 같은 `NODE_ID`로 두 대를 굽지 않았는지 | env가 다르면 자동으로 다름 |
| ☐ | `CSI_FS 100` 그대로 (M2 학습 fs와 3자 결속 — 건드리지 말 것) | |
| ☐ | Kconfig CSI 키 확인 | `CONFIG_ESP_WIFI_CSI_ENABLED` |
| ☐ | rp5 도달 확인 | `ssh csi@192.168.1.2` |
| ☐ | rp5 디스크 여유 ≥ 50GB | `df -h` |
| ☐ | USB 64GB 이상 | 5노드 1시간 ≈ 5.8GB |
| ☐ | 매트 (낙상 recall 검증에 필수 — 04에서 매트가 없어 미실시) | |
| ☐ | 줄자 (노드 좌표 실측용) | |

---

## 1. 현장 설치

1. 노드 5개를 배치한다. **한 곳에 몰지 말 것** — 근거리/중거리/차폐가 섞여야 노드별 비교가 가능하다.
2. 방 한 구석을 원점 `(0,0)`으로 잡고 **줄자로 노드 5개 좌표를 실측**한다.
3. `data/node_layout.csv`에 기입 (예시행 5줄은 지운다).

```csv
layout_id,node_id,x_m,y_m,z_m,mount,notes
L1,1,0.00,0.00,1.20,책상,
L1,2,3.40,0.00,1.10,선반,
...
```

> 배치를 바꾸면 **새 `layout_id`**(L2, L3…)를 딴다. 사전학습 데이터가 환경 E22 하나뿐이라
> 배치는 **2~3종**을 확보해야 한다.

4. **노드 예열 30초** — 호흡 대역 필터가 정착할 시간.

---

## 2. 5노드 스모크 — GO / NO-GO

수집기를 2분 돌린다. 그동안 rp5에서:

```bash
# 5노드 전부 살아있나
for n in 1 2 3 4 5; do echo "--- node $n"; redis-cli HGETALL node:$n:health; done
redis-cli --raw XLEN csi:raw
```

수집 종료 후 노트북에서:

```bash
python3 scripts/verify_sync_v2.py --rawdir data/raw --normal-only
python -m m1_fall.dataset configs/m1_fall.yaml
```

### 통과 기준 — 전부 만족해야 GO

| ☐ | 기준 | 실패 시 |
|---|---|---|
| ☐ | `verify_sync_v2` 출력에 **"노드 5"** | 안 오는 노드의 `TARGET_IP`·전원 확인 |
| ☐ | 5개 전부 **100Hz ± 15%** (85~115) | 부록 A-1 |
| ☐ | 5개 전부 **끊김 0초** | 부록 A-2 |
| ☐ | `ts_ms` 랩어라운드 0회 | 49.7일 주기 — 정상은 0 |
| ☐ | 노드별 `loss_rate` < 5% | 부록 A-1 |
| ☐ | 노드별 `rssi` -40 ~ -75 dBm | AP 거리/채널 조정 |
| ☐ | `dataset.py`가 창 shape **`(5, 64, 100)`**을 출력 | 아니면 `n_nodes` 설정·`node_id` 컬럼 확인 |
| ☐ | 세션별 **노드 커버리지 5개 전부 ≥ 90%** | 낮은 노드는 그리드에 0으로 패딩된다 = 사실상 죽은 노드 |

**하나라도 실패하면 본 수집 금지.** 원인을 잡고 스모크를 다시 돌린다.

> ⚠️ 5노드 동시 100Hz는 **한 번도 검증된 적이 없다.** 04는 ESP 1대였다.
> 레퍼런스 펌웨어(AI_HACK_CAMP_2026_CSI)는 **20Hz**를 택했었다 — 100Hz는 공격적인 설정이다.
> 여기가 이번 현장의 1순위 관찰 포인트다.

---

## 3. 반응 지연 실측 (10분, 한 번만)

`mark_falls.py`는 사람이 보고 키를 누를 때까지의 지연을 갖는다. 기본 300ms를 빼지만,
본인 지연을 한 번 재두면 라벨이 정확해진다.

1. 낙상 3~5회를 **영상으로 같이 찍으면서** 마킹
2. 영상의 실제 접지 시각 vs 마커 시각 차이의 중앙값을 구함
3. `verify_sync_v2.py --lag-ms <실측값>`으로 넘김

> 양성 구간이 ±1000ms라 치명적이진 않다. 300ms로 가도 된다.

---

## 4. 본 수집

### 목표 (04 실측 반영)

| 클래스 | 목표 | 비고 |
|---|---|---|
| **FALL** | 200~400회 | 방향 4종 × 시작자세 3종 × 반복 |
| **HARDNEG** | **150~200회** | 04에서 빠르게앉기 오탐 **54%** — 최우선 |
| NORMAL | 2~4시간 | 일상 동작 |

**하드 네거티브 종류** (빠르게 앉기는 3종으로 세분):
- 빠르게앉기 — 의자 털썩 / 침대 쓰러지듯 / 바닥 주저앉기
- 눕기 · 쭈그려 물건 줍기 · 물건 떨어뜨리기 · 비틀거리기

계획표가 필요하면:
```bash
python3 scripts/session_report.py --plan 200 180
# → data/session_plan.csv
```

### 녹화 절차 (매 세션)

```bash
# 터미널 1: 수집기
# 터미널 2:
python3 scripts/mark_falls.py --session csi_20260901_1430.csv
#   space / f  낙상 순간   x  직전 취소   b  나쁜 시도   q  종료
```

**녹화 하나당 이벤트 4회로 제한.** split이 세션 단위라 세션 수가 곧 다양성이다.

### 매 세션 `data/session_log.csv`에 기록 — 빠뜨리면 복구 불가

```csv
session_id,csv_prefix,date,layout_id,subject_id,class,scenario,fall_dir,start_pose,spot_x_m,spot_y_m,n_events,occlusion_by_node,operator,lag_ms,verify_pass,notes
F001,csi_20260901_1430,2026-09-01,L1,S1,FALL,낙상-전방-서기,front,stand,1.60,2.00,4,1:clear;2:clear;3:furniture;4:wall;5:clear,태연,300,y,
```

**핵심 3칸:**
- `spot_x_m` / `spot_y_m` — 낙상 지점 좌표. 노드별 거리는 여기서 자동 계산된다
- `occlusion_by_node` — `노드:등급` 세미콜론 구분. 등급은 `clear` / `furniture` / `door` / `wall` 넷 중 하나
- `csv_prefix` — 수집 파일명 접두. 나중에 라벨과 조인하는 키

> 수집기는 **60초마다 파일을 쪼갠다.** 4분 녹화 = 파일 4개.
> `session_log`는 **녹화 1건 = 1행**으로 적고, `csv_prefix`는 첫 파일 이름을 쓴다.

### 다양성 목표

- **피험자 5~8명** — 03 CV에서 피험자 간 변동(0.238)이 구성 효과(0.057)의 4배. 환경보다 우선
- **배치 2~3종**
- 우선순위: **하드네거티브 > 피험자 수 > 환경**. 시간이 부족하면 이 순서로 줄인다

---

## 5. 철수 전 필수 — 여기서 안 하면 다시 와야 한다

```bash
# 1) 라벨 생성 + 동기 검증
python3 scripts/verify_sync_v2.py --markers markers/*.markers.csv \
        --rawdir data/raw --out data/labels_self.csv --lag-ms 300

# 2) 로더 통과 + 윈도우 수
python -m m1_fall.dataset configs/m1_fall.yaml

# 3) 기록 검증 + 노드 커버리지
python3 scripts/session_report.py

# 4) 라벨에 노드 거리/차폐 부착
python3 scripts/session_report.py --join data/labels_self.csv
```

| ☐ | 확인 |
|---|---|
| ☐ | `verify_sync_v2`에서 실패 세션 0건 |
| ☐ | `session_report.py` **치명 0** |
| ☐ | 노드별 거리 분포가 한쪽으로 안 쏠림 (≤2m / 2-4m / >4m + 차폐가 고루) |
| ☐ | `session_log.csv`의 `verify_pass`를 전부 `y`로 채움 |
| ☐ | 계획 대비 실제 이벤트 수 확인 |

---

## 6. 회수

```bash
# rp5에서 압축 (3~5배)
gzip -r data/raw/
du -sh data/raw/
```

USB로 회수. `session_log.csv` · `node_layout.csv` · `markers/` · `labels_self.csv`를 **같이** 가져온다.
**raw만 가져오면 라벨이 없는 데이터가 된다.**

---

## 부록 A — 실패 대응

**A-1. 100Hz 미달 / loss_rate 높음** (가장 흔함)
1. 노드를 1대씩 켜가며 몇 대부터 무너지는지 본다 → WiFi 경합이면 대수에 비례해 악화
2. AP 채널 변경 (2.4GHz 1/6/11 중 빈 채널)
3. 노드 간 거리를 벌린다
4. 그래도 안 되면 — `CSI_FS`는 M2 학습 fs와 묶여 있어 **임의로 못 낮춘다.** 백엔드팀과 재논의 사안

**A-2. 특정 노드만 끊김**
- 그 노드 RSSI 확인 → -75 이하면 AP에서 너무 멈
- 전원(USB 허브 전류 부족) 확인
- 재플래시

**A-3. CSI 콜백 자체가 없음**
- `info->len == 128` 인지 시리얼로 확인. 다르면 서브캐리어 매핑 가정이 깨진 것
- WiFi 채널/AP 의심 (레퍼런스는 채널 13 고정이었음)
- amplitude 추출·더미 트리거는 레퍼런스로 검증된 부분이라 의심 대상이 아니다

**A-4. `verify_sync_v2`가 마커를 못 붙임**
- 마커 epoch가 파일 시간 범위 밖 → 수집기보다 마커를 먼저/나중에 돌린 경우
- `stream_id` 컬럼 존재 확인 (오프셋 계산의 근거)

---

## 부록 B — 기록 파일

| 파일 | 단위 | 용도 |
|---|---|---|
| `data/node_layout.csv` | 배치 × 노드 | 노드 좌표. 낙상 지점과의 거리 계산 근거 |
| `data/session_log.csv` | 녹화 1건 | 시나리오·피험자·낙상 지점·차폐 |
| `data/session_plan.csv` | 녹화 1건 | 사전 계획 (`--plan`으로 생성) |
| `data/labels_self.csv` | 파일 1개 | `verify_sync_v2`가 생성. 학습이 읽는 것 |
| `markers/*.markers.csv` | 녹화 1건 | `mark_falls`가 생성. 원본 마커 |

`session_report.py --join`이 `labels_self_nodes.csv`를 만들어 라벨에 노드 거리·차폐를 붙인다.
**파인튜닝 후 노드별 recall을 뽑을 때 이 파일이 있어야 한다.**
