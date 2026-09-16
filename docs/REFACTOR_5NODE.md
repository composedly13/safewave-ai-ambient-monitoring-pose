# 5노드 전환 정리 (2026-09-16)

M1 입력을 per-node `(B,1,64,100)`에서 **5노드 조기융합 `(B,5,64,100)`**으로 바꾸고,
레포 전체가 그 전제로 일관되게 읽히도록 정리한 기록이다.

브랜치: `develop`(베이스라인) → `feature/m1-5node`(코드 전환) → `chore/refactor-5node`(문서·정합)

---

## 1. 무엇이 바뀌었나

| 축 | before | after |
|---|---|---|
| 입력 텐서 | `(B, 1, 64, 100)` | **`(B, 5, 64, 100)`** |
| 노드 처리 | 노드마다 독립 창 (per-node 추론) | **조기융합** — 노드축 = 첫 conv 입력채널 |
| 융합 주체 | M5(Qwen)가 5노드 결과를 합침 | **M1이 직접** 5노드를 한 입력으로 받음 |
| 노드 시간 정렬 | 없음 (노드별 독립) | **`stream_id`의 epoch** 100Hz 그리드, ±5ms 최근접 |
| 결측 노드 | 해당 없음 | 그리드 슬롯을 **0으로 패딩** |
| 낙상 라벨 도메인 | `ts_ms` (장치 시계) | **epoch** (2³² 미만이면 레거시로 보고 자동 변환) |
| 학습 파라미터 | 134,561 | **135,137** (conv1 입력채널 1→5, +576) |
| 사전학습 | `best_B2_final.pt` 워밍스타트 | **스크래치 재학습** (입력채널이 달라 로드 불가) |

서브캐리어 64 · 프레임 100 · `CSI_FS 100` · per-frame peak 정규화 ·
`strict=True` · opset 17 · input `csi_data` · **그래프에 Sigmoid 없음** — 전부 그대로다.

---

## 2. 변경·이동·보관 파일

### 코드 (검증된 학습 로직은 손대지 않음)

| 파일 | 내용 |
|---|---|
| `src/m1_fall/config.py` | `TensorCfg.n_nodes` 추가(기본 1, 1~8). `pretrain` + `n_nodes != 1` 조합을 로드 시점에 거부 |
| `src/m1_fall/model.py` | 첫 conv 입력채널 = `n_nodes` |
| `src/m1_fall/dataset.py` | epoch 그리드 정렬로 재작성, 창 `(5,64,100)` |
| `src/m1_fall/train.py` | `build_model`에 `n_nodes` 전달 (한 줄) |
| `src/m1_fall/export.py` | 더미·parity 입력 `(1,5,64,100)`, `assert_no_sigmoid()` 추가, rp5 패치 생성기 5노드화, 생성 스크립트에 `dynamo=False` 폴백 |
| `configs/m1_fall.yaml` | `n_nodes: 5`, `pretrain` 블록 제거 |
| `configs/m1_fall_pretrain.yaml` | `n_nodes: 1` 명시 + 레거시 배너 |

### 문서

| 파일 | 내용 |
|---|---|
| `CLAUDE.md` | 규칙 1·7 교체, 규칙 9(epoch 정렬) 신설, 레거시 아티팩트 ⚠️ 표시, rp5 서술 미검증 표시 |
| `AGENTS.md` | **예전 CLAUDE.md의 바이트 단위 사본**이었다 → 포인터 문서로 교체 |
| `README.md` | 하드컨트랙트 표 갱신, per-node 행 2개 삭제, rp5 sync를 질문 4개로 재구성 |
| `data/README.md` | epoch 그리드·0패딩·`fall_ts_ms` 도메인 판별표 |
| `docs/COLLECTION_5NODE.md` | 라벨 설명 + GO/NO-GO 한 줄 정정 (현장 절차는 그대로) |
| `runs/README.md` | **신규** — 레거시 아티팩트 설명 |
| `docs/REFACTOR_5NODE.md` | 이 문서 |

### 이동 / 보관 (삭제한 것 없음)

| 대상 | 이동 후 |
|---|---|
| `docs/WORKORDER_M1.md` | `docs/archive/WORKORDER_M1.md` + 시점 배너 |
| `docs/PRECOLLECTION_STATUS.md` | `docs/archive/PRECOLLECTION_STATUS.md` + 시점 배너 |
| `docs/RP5_DEVELOP_AUDIT.md` | `docs/archive/RP5_DEVELOP_AUDIT.md` + 시점 배너 (**정찰 기준선은 여전히 유효**) |
| `runs/best_B2_final.pt` | `runs/legacy_singlenode/` (git 제외 폴더라 커밋에는 안 잡힘) |
| `runs/m1_wifi_pose.onnx` | `runs/legacy_singlenode/` (동일) |
| `data/labels_self.csv`의 6/11 NORMAL 8행 | `data/labels_self_legacy.csv` (원본은 헤더만 남김) |

### 손대지 않은 것

`scripts/{mark_falls,verify_sync_v2,session_report}.py` (표준 라이브러리, 구조 무관),
`data/{session_plan,node_layout,session_log}.csv`, `data/labels.csv`(레거시 보관).

---

## 3. 통과한 검증

| 검증 | 결과 |
|---|---|
| `python -m m1_fall.model` | `(4,5,64,100)` → `(4,1)`, 파라미터 **135,137** |
| config 로드 | `m1_fall.yaml` n_nodes=5 / `m1_fall_pretrain.yaml` n_nodes=1 둘 다 OK |
| `n_nodes=9` 거부 | OK (1~8 범위 검증) |
| `pretrain` + `n_nodes=5` 거부 | OK — `n_nodes=1` + pretrain은 통과(단일노드 복귀 경로 보존) |
| 합성 5노드 데이터 dataset | train 333창 / val 285창, 창 shape `(5,64,100)`, 노드 커버리지 출력 |
| 레거시 `ts_ms` 도메인 라벨 자동 변환 | OK — val 세션을 레거시 도메인으로 줬는데 양성 41건 검출(틀렸으면 0건) |
| 학습 2 epoch | 완주, `best.pt` 저장, pr_auc 0.217 → 0.481 |
| 체크포인트 strict 왕복 | 5노드 모델 로드 OK / 1노드 모델은 size mismatch 거부 |
| ONNX export | `csi_data [batch,5,64,100]` → `fall_logit [batch,1]`, opset 17 |
| **Sigmoid 노드 검사** | 없음 (`assert_no_sigmoid()`가 자동 강제) |
| torch ↔ onnxruntime parity | `max|diff| = 2.38e-07` (atol 1e-4) |
| **생성된 rp5 스크립트 실행** | 실행 성공, 그 ONNX가 레포 ONNX와 **출력 완전 동일 (max diff 0.0)** |
| 레거시 아티팩트 실측 | `best_B2_final.pt` conv1 = `(16,1,3,3)` → 단일노드 확정 |

**전부 합성 데이터 기준이다. 실측 CSI로는 아직 학습하지 않았다.**

---

## 4. 사람이 결정해야 할 것

### ① 원격 히스토리 — 임의로 합치지 않았다

로컬 레포는 원래 `.git`이 없어서 2026-09-16에 새로 `git init` 했다.
`origin`(`composedly13/safewave-ai-ambient-monitoring-pose`)을 붙여 **fetch만** 해봤다:

```
origin/main            effda1c 2026-08-02  labels: B2 config (ESP32 494 + HP 500)...
                       eaa85f4 2026-08-02  B2 backbone, ONNX export, rp5 patch bundle...
                       aac183b 2026-06-29  Merge PR #1
                       634c92f 2026-06-29  feat(m1): train.py + export.py
                       ea59ade 2026-06-12  chore: scaffold
origin/feature/m1-fall 634c92f (위와 동일 계보)

로컬 chore/refactor-5node ← 92471b5 (2026-09-16 root commit)
```

**`git merge-base`가 아무것도 반환하지 않는다 = 공통 조상이 없다.**
두 히스토리는 완전히 무관하다. `git diff 92471b5 origin/main` 요약:

| 방향 | 내용 |
|---|---|
| 로컬에만 있음 (~1990줄) | 8/25 데스크 작업 전부 — `docs/` 4종, `scripts/session_report.py`, `AGENTS.md`, `data/session_plan.csv`, `config.py`의 pretrain, `train.py`의 워밍스타트 로직. **GitHub에 한 번도 푸시되지 않았다.** |
| 원격에만 있음 (~121줄 + 바이너리) | `runs/best_B2_final.pt`(546KB), `runs/m1_wifi_pose.onnx`(543KB), `runs/rp5_patch/` 2개가 **git에 커밋돼 있다.** 로컬 `.gitignore`는 `runs/*`를 제외한다. |

**선택지 (사용자가 골라야 함 — 어느 것도 실행하지 않았다):**

1. **그대로 둔다** — 로컬은 로컬대로 쓰고 GitHub는 8/2 상태로 보존. 가장 안전.
2. **새 브랜치로 푸시** — `git push -u origin chore/refactor-5node`.
   무관한 히스토리라 GitHub에 고아 브랜치로 뜬다. PR을 열면 "공통 커밋 없음"이 된다.
3. **원격 위에 재정렬** — `origin/main`을 베이스로 로컬 작업을 얹는다.
   `runs/` 바이너리 추적 여부(로컬은 제외, 원격은 포함)를 먼저 정해야 한다.

> **`--force` 푸시, `reset --hard`, 무관 히스토리 강제 병합은 하지 않았다.**
> 되돌릴 일이 생기면 `revert`를 쓸 것.
> 전체 히스토리 백업: **`D:\pose-backup-20260916.bundle`** (4 refs, complete history, verify 통과).

### ② rp5 `develop` 정찰 — 5노드 전환으로 확인 범위가 넓어졌다

이 레포에서는 rp5 코드를 볼 수 없다. **추측 금지.** 확인할 질문은
[`../README.md`](../README.md)의 "rp5 sync" 절과 `runs/rp5_patch/PREPROCESS_AND_EXPORT.md`에
정리해 뒀다. 요약하면:

1. `_preprocess`가 `(1,5,64,100)`을 만드는가? (노드를 채널축으로 쌓는가)
2. **`ai/main.py`가 5노드를 무엇으로 시간 정렬하는가?** ← 가장 위험.
   rp5가 `ts_ms`로 정렬하면 **shape은 맞는데 내용이 틀린 입력**이 들어간다.
   추론이 실패하지 않고 조용히 엉뚱한 점수를 낸다.
3. 노드가 빠진 시점에 0을 채우는가? (이 레포 로더는 0 패딩)
4. `_infer_onnx`가 sigmoid를 몇 번 하는가? (0번이면 임계 0.7 무의미, 2번이면 뭉개짐)

### ③ 수집 직후 첫 할 일 — `labels.csv` 교체

**지금 `python -m m1_fall.dataset configs/m1_fall.yaml`을 돌리면 실패한다:**

```
FileNotFoundError: data\raw\csib_ESP32_Human_U08_Fall_E22_ESP32_10000.csv
```

`data/labels.csv`가 아직 csibench 1002행(단일노드, `stream_id` 없음)을 가리키고
`data/raw/`가 비어 있기 때문이다. **버그가 아니라 데이터가 없는 상태의 정상 동작**이지만,
에러 메시지가 원인을 설명해 주지는 않으니 내일 아침에 이 줄을 기억할 것.

수집 후 순서:

1. `verify_sync_v2.py`로 검증 → 라벨 줄 생성 → `data/labels_self.csv`에 기입
   (지금은 헤더만 있다)
2. `configs/m1_fall.yaml`의 `paths.labels_csv`를 `data/labels_self.csv`로 바꾸거나,
   `labels_self.csv` 내용을 `labels.csv`로 복사
3. `python -m m1_fall.dataset configs/m1_fall.yaml` → 창 수와 **노드 커버리지(%)** 확인
4. 커버리지가 낮은 노드가 있으면 학습 전에 동기·수집 문제를 먼저 잡을 것

> `mark_falls.py`는 **epoch ms**를 찍고 로더도 epoch를 기본으로 읽으므로,
> 새 라벨은 변환 없이 그대로 쓰면 된다. `verify_sync_v2.py`는 `ts_ms` 도메인으로
> 변환해 기록하는데, 그것도 로더가 자동 판별해 되돌린다(2³² 기준).

---

## 5. 알려진 한계 (정직하게)

- **실측 데이터로 검증된 것이 하나도 없다.** 모든 수치는 합성 5노드 데이터 기준이다.
- **rp5 쪽은 전부 미검증이다.** 배포 계약의 절반은 아직 가정이다.
- **5노드 조기융합 자체가 검증된 설계는 아니다.** 노드를 채널축으로 쌓는 것이
  per-node보다 나은지는 실측 학습으로 확인해야 한다. 04 실측에서 드러난
  "모델이 낙상이 아니라 모션 강도를 본다"는 문제는 **라벨 품질 문제**라
  구조를 바꿔도 그대로 남는다 — 하드 네거티브 수집이 여전히 필요하다.
- `docs/archive/`의 rp5 관련 서술은 2026-08-25 시점이고 그 뒤 재확인하지 않았다.
