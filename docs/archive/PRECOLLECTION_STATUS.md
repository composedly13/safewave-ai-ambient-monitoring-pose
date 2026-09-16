> ⚠️ **2026-08-25 시점 기록. 현재 지침이 아니다.**
> 이 문서는 입력이 per-node `(B,1,64,100)`이던 시절의 상태 보고다. 2026-09-16에 M1은
> **5노드 조기융합 `(B,5,64,100)`**으로 전환됐다 — 현행 지침은 `CLAUDE.md`와
> `docs/COLLECTION_5NODE.md`다. 여기 적힌 체크포인트·ONNX 상태는 **단일노드 레거시**
> (`runs/legacy_singlenode/`)를 가리킨다. 이력 보존용으로 남긴다.
> **`RP5_DEVELOP_AUDIT.md`의 rp5 정찰 기준선은 여전히 유효하다.**

---

# 현장 전 데스크 작업 결과 (2026-08-25)

## 요약

| 태스크 | 상태 | 비고 |
|---|---|---|
| T1 rp5 정찰 | **실패** | rp5에 접속 불가 (서브넷 불일치). Q1~Q5 전부 미확인 |
| T2 문서 정정 | **부분** | T1이 실패해 문구를 지우지 않고 `⚠️ 미검증` 표시만 붙임 (완료 조건대로) |
| T3 `--init-ckpt` | **완료** | 구현 + 검증 통과 (실패 0건) |
| T4 per-node 추론 루프 | **미수행** | 조건부 태스크. T1 Q4 미확인 + rp5 접속 불가 |
| T5 수집 계획표 + 리허설 | **완료** | 계획표 95건 생성, 4개 케이스 리허설 전부 기대대로 |
| T6 최종 보고 | **완료** | 이 문서 |

**먼저 알아야 할 것: 이 레포는 git 저장소가 아니다.** `.git` 이 없어서 `chore/pre-collection`
브랜치도, 태스크별 커밋도 만들지 못했다. 아래 "커밋" 절 참조.

---

## T1 핵심 발견

전문은 [`RP5_DEVELOP_AUDIT.md`](RP5_DEVELOP_AUDIT.md).

**rp5(`192.168.1.2`)에 접속하지 못했다.** 이 PC는 `192.168.0.41/24`(게이트웨이 `192.168.0.1`)에
있고 rp5는 `192.168.1.0/24`다 — 서로 다른 망이고 라우트가 없다. `192.168.0.x` 전 대역에
SSH 열린 호스트가 없고, Tailscale 노드도 둘 다 offline이다.

| # | 질문 | 답 |
|---|---|---|
| Q1 | sigmoid 적용 횟수 | **확인 못 함** |
| Q2 | `_preprocess` shape 출처 | **확인 못 함** |
| Q3 | `M1_MAX_NODES` 값·조립 방식 | **확인 못 함** |
| Q4 | 노드별 ONNX 루프 유무 | **확인 못 함** |
| Q5 | `ai:m1:latest` 노드별 분리 | **확인 못 함** |

**T4 필요 여부: 판정 불가 → 수행하지 않음.**

### 정찰은 실패했지만, 대신 나온 게 하나 있다

로컬 디스크에 배포 레포 스냅샷 두 개가 남아 있었다:

| 경로 | mtime | M1 전문가 |
|---|---|---|
| `D:\rp5` | 2026-05-07 | `m1_fall.py` (구세대) |
| `D:\safewave-ai-ambient-monitoring` | 2026-06-05 | `m1_wifi_pose.py` |

둘 다 git 메타데이터가 없어 **브랜치를 알 수 없다.** 2026-06-05 스냅샷은
`CLAUDE.md`가 develop의 사실로 적어 둔 항목들과 어긋나므로(`csi:raw`가 3필드가 아니라
`data` 단일 필드 등) **develop이 아니다.** 그래서 Q1~Q5의 답으로 쓰지 않았고,
`RP5_DEVELOP_AUDIT.md` 부록 A에 **정찰 재개 시 diff 기준선**으로만 넣어 두었다.

### 그 과정에서 드러난 진짜 문제

**이 레포 안의 문서 두 갈래가 develop에 대해 정반대를 주장하고 있고, 어느 쪽도 검증된 적이 없다.**

| 항목 | `CLAUDE.md` 함정 표 | `README.md` / `PREPROCESS_AND_EXPORT.md` | 2026-06-05 스냅샷 |
|---|---|---|---|
| `_preprocess` shape | ONNX에서 직접 읽음 → 수정 불필요 | 192 → 64로 고쳐야 함 | **192 하드코딩** |
| main.py 롤링 버퍼 | 이미 구현됨 (노드별 deque) | 변경 불필요 (upstream 조립) | **버퍼 없음** |
| sigmoid | rp5 `_infer_onnx`가 적용 | "(then sigmoid)" | **없음, `np.clip`만** |

세 열이 전부 다르다. `CLAUDE.md`의 "GitHub main만 보고 세운 가정이 대부분 틀렸다"는 교훈이
**함정 표 자체에도 적용될 수 있다** — 그 표가 어느 시점의 develop을 보고 쓴 건지 근거가 남아 있지 않다.

**배포 전에 반드시 rp5 develop을 직접 읽어야 한다.** 특히 sigmoid는 방향이 양쪽으로 틀릴 수 있다:

- rp5가 sigmoid를 **안 하면** → `fall_logit`이 `clip(0,1)`을 통과해 음수는 0.0, 양수는 1.0 근처로
  잘리고 임계 0.7이 무의미해진다
- rp5가 **두 번** 하면 → 출력이 0.5~0.73으로 뭉개진다 (`CLAUDE.md` 규칙 2)

참고로 `CLAUDE.md`에 기록된 04 실측(2026-08-02)의 baseline CSV에 exact 0.0/1.0이 0건이었다는
사실은 **측정 당시엔 sigmoid가 정확히 한 번 걸려 있었다**는 간접 증거다. 다만 그건 측정 시점의
이야기이고, 코드로 확인한 것이 아니다.

---

## T2 정정 내역

T1이 실패했으므로 완료 조건대로 **문구를 지우지 않고 `⚠️ 미검증 (2026-08-25)` 표시만** 붙였다.

`runs/rp5_patch/PREPROCESS_AND_EXPORT.md`
- 문서 최상단에 미검증 배너 + `CLAUDE.md`와 정반대라는 사실 명시
- "Unlike M2, no main.py buffer change is needed…" 문장에 미검증 표시 (지시서는 삭제를 지시했으나,
  근거가 될 T1이 없어 완료 조건 2의 폴백 규칙을 따랐다)
- `## 1. _preprocess 192 → 64` 에 "이 수정이 필요한지 자체가 미확인" 표시
- "(then sigmoid)" 에 미검증 표시 + 양방향 실패 모드 설명

`README.md`
- 하드 컨트랙트 표에 **2행 추가** (기존 행은 건드리지 않음):
  - `Sigmoid` — rp5 쪽에서만, ONNX 그래프에 넣지 말 것. 이중 적용 시 0.5~0.73으로 뭉개짐
  - `Node fan-out` — `M1_MAX_NODES`는 M1의 관심사가 아님. 노드 융합은 M5(Qwen)
- `⚠️ rp5 sync requirement` 절에 미검증 배너
- 같은 절에 "아직 답이 안 나온 세 번째 항목"(롤링 버퍼) 추가

### T2 정정이 오래가지 못한다 — 두 가지 이유

**(1) 그 파일은 자동 생성물이라 다음 export 때 덮어써진다.**
`src/m1_fall/export.py:199` 의 `emit_rp5_patch()` 가 `runs/rp5_patch/PREPROCESS_AND_EXPORT.md` 를
인라인 템플릿에서 통째로 다시 쓴다. 템플릿 원문에 문제의 문장들이 그대로 들어 있다:

```
export.py:160-161   (Unlike M2, no main.py buffer change is needed — M1 infers on a single
                     1-node x 100-frame window already assembled upstream.)
export.py:163       ## 1. ai/experts/m1_wifi_pose.py._preprocess  (192 -> 64 subcarriers)
export.py:189       (then sigmoid), which is exactly our `{cfg.export.output_name}` (B,1).
```

즉 `python -m m1_fall.export` 를 한 번만 돌리면 **T2에서 붙인 `⚠️ 미검증` 표시가 전부 사라진다.**
근본 수정은 `export.py` 의 템플릿을 고치는 것인데, T2 완료 조건이 "코드 파일 변경 없이 문서만"이라
임의로 손대지 않았다. → 아래 "사용자 확인이 필요한 결정".

**(2) 그 파일은 git 추적 대상이 아니다.**
`.gitignore` 가 `runs/*` 를 무시한다 (`!runs/.gitkeep` 만 예외).
rp5에 넘길 패치 문서를 추적하려면 `!runs/rp5_patch/` 예외를 넣거나 `docs/` 로 옮겨야 한다.

`README.md` 쪽 정정은 두 문제 모두에 해당하지 않는다 — 손으로 쓰는 파일이고 추적된다.

---

## T3 구현 + 검증

`configs/m1_fall.yaml` — 선택적 `pretrain` 블록 추가 (`tensor:` · `export:` 는 손대지 않음)
```yaml
pretrain:
  ckpt: runs/best_B2_final.pt
  load: full             # full | backbone
  freeze_epochs: 5
  backbone_lr_mult: 0.1
```

`src/m1_fall/config.py` — `PretrainCfg` 추가, `Config.pretrain: Optional[PretrainCfg] = None`,
`load_config` 가 `raw.get("pretrain")` 로 **없으면 None**, `validate()` 에 4개 검사 추가

`src/m1_fall/train.py` — `load_pretrained` / `set_backbone_trainable` / `build_optimizer` /
`resolve_pretrain` 추가, CLI `--init-ckpt` 추가, 학습 루프에 동결·해동 분기 추가

### 검증 결과 (검증 스크립트는 `/tmp` 격 임시 폴더, 레포에 커밋 안 함)

`data/raw/` 가 비어 있어 학습은 못 돌린다. 로드·동결 부분만 떼어 확인했다. **실패 0건.**

```
1 · load=full · strict=True
[pretrain] runs/best_B2_final.pt | keys 27/27 | epoch 14 | pr_auc 0.6961
  [o] 키 27/27   [o] epoch 14   [o] pr_auc 0.6961   [o] 가중치가 실제로 바뀌었다

2 · 망가뜨린 state_dict → RuntimeError
  [o] 키 1개 삭제 / 키 이름 변조 / shape 변조 → 전부 RuntimeError
  [o] 빈 state_dict → RuntimeError   [o] 없는 파일 → RuntimeError

3 · load=backbone
[pretrain] runs/best_B2_final.pt | keys 21/27 (backbone) | epoch 14 | pr_auc 0.6961
  [o] convs 가 바뀜   [o] GRU 는 랜덤 유지   [o] 21/27

4 · 동결 / 해동 시 optimizer param group   (백본 23,520p / 전체 134,561p)
  동결: [optim] param groups -> head: 111,041p lr=1.0e-03 | frozen: backbone 12 tensors (excluded)
  해동: [optim] param groups -> backbone: 23,520p lr=1.0e-04, head: 111,041p lr=1.0e-03
  [o] 동결 중엔 group 이 head 하나뿐, 백본이 optimizer 에서 아예 빠짐
  [o] 해동 후 backbone lr = 1.0e-03 * 0.1 = 1.0e-04
  [o] 동결 상태로 한 스텝 밟으면 conv 가중치 불변 / head 만 변함

5 · CLI 우선순위
  [o] --init-ckpt 가 config 의 ckpt 를 이김 (나머지 노브는 config 값 유지)
  [o] pretrain 블록 없는 config → None, 그래도 --init-ckpt 는 기본 노브로 동작
```

전체 파라미터 134,561 은 `CLAUDE.md` 검증된 사실의 학습 파라미터 수와 일치한다.

### 완료 조건 대조

- [x] `pretrain` 블록 없는 기존 config 가 그대로 로드된다 — `configs/m1_fall_pretrain.yaml`
      `load_config` 성공, `cfg.pretrain is None`.
      `python -m m1_fall.dataset configs/m1_fall_pretrain.yaml` 은 **config 단계는 통과**하고
      그 뒤 `data/raw/csib_*.csv` 가 없어서 `FileNotFoundError` 로 죽는다 —
      `pretrain` 블록이 있는 config 도 **완전히 같은 지점에서 같은 이유로** 죽으므로 config 회귀는 없다
- [x] `--init-ckpt runs/best_B2_final.pt` 로 27/27 로드가 로그에 찍힌다
- [x] 일부러 망가뜨린 state_dict 로는 `RuntimeError` 가 난다 (4종 전부)
- [x] `strict=False` 가 코드 어디에도 없다 — grep 히트는 "쓰지 말라"는 설명 주석 2줄뿐
- [x] 동결/해동 시 optimizer param group 이 실제로 바뀌는지 확인할 로그가 있다 (`[optim]` 줄)

**미확인으로 남긴 것:** `python -m m1_fall.train --init-ckpt …` 를 끝까지 돌려 보지는 못했다.
`--help` 로 플래그가 붙은 건 확인했지만, 실제 실행은 `make_dataloaders` 가 빈 `data/raw/` 에서
죽어 `[pretrain]` 줄에 닿기 전에 중단된다. 로드 경로 자체는 위 검증 스크립트로 직접 확인했다.

---

## T4 — 수행하지 않음

조건부 태스크이고, 조건인 T1 Q4("루프 없음")를 확인하지 못했다. 더구나 T4는 rp5 레포에
feature 브랜치를 파고 `sensing/simulator.py --nodes 5` 로 검증하는 작업이라 rp5 접속 없이는
착수 자체가 불가능하다. **rp5에는 아무것도 하지 않았다** (읽기조차 못 했다).

---

## T5 수집 계획표 + 리허설

### 계획표
```
$ python3 scripts/session_report.py --plan 200 180
[plan] data/session_plan.csv: 녹화 95건 (FALL 50 / HARDNEG 45) · 이벤트 200+180
[plan] 녹화당 2분 가정 시 순수 수집 190분 ≈ 3.2시간 (설치·휴식 제외) · 5노드 데이터 약 18GB
```
**순수 녹화만 3.2시간**이다. 설치·이동·휴식·재촬영을 얹으면 하루로는 빠듯하다.
피험자 5~8명을 노린다면 표를 사람 수만큼 복제해야 하므로 **여러 날에 나눠야 한다.**

### 리허설 (가짜 세션 3건: FALL 2 + HARDNEG 1)

| 단계 | 내용 | exit | 기대 | 결과 |
|---|---|---|---|---|
| 1 | 차폐 등급 오타 `3:커튼` + `spot_y_m` 누락 + 노드 4 차폐 미기입 | 1 | 1 | OK |
| 2 | 전부 고친 기록 | 0 | 0 | OK |
| 3 | `--join data/labels_self.csv` | 0 | 0 | OK |
| 4 | 노드 4개짜리 배치 (`EXPECTED_NODES=5` 위반) | 1 | 1 | OK |

1단계에서 스크립트가 잡아낸 것:
```
[X] R01: 알 수 없는 차폐 등급 '커튼' (허용 ['clear', 'furniture', 'door', 'wall'])
[X] R01: 노드 ['3'] 차폐 미기입
[X] R02: spot_x_m / spot_y_m 미기입 — 노드 거리 계산 불가
[X] R03: 노드 ['4'] 차폐 미기입
[!] R01: verify_pass 미기입 — verify_sync_v2 통과 전
치명 4 · 경고 1
```
**오타는 잡힌다.** `clear`/`furniture`/`door`/`wall` 외의 값은 전부 치명으로 거부하고,
거부된 노드는 "차폐 미기입"으로 한 번 더 걸린다.

거리 버킷은 세 구간이 전부 나오도록 좌표를 잡았고, 실제로 분리돼 나왔다:
```
node 1: 2-4m 4, ≤2m 4
node 3: 2-4m 4, >4m/door 4
node 4: 2-4m/furniture 4, >4m/wall 4
```

`--join` 은 `data/labels_self_nodes.csv` 를 만들었고 **8/8행에 노드 거리·차폐가 붙었다**
(`csv_prefix` 를 `csi_20260611_17` 로 두어 접두 매칭 확인).

### 정리 후 상태
- `data/node_layout.csv` · `data/session_log.csv` — **예시행까지 전부 지우고 헤더 1줄만** 남김
- `data/labels_self_nodes.csv` — 리허설 산출물이라 삭제
- 헤더만 남은 상태로 `session_report.py` 재실행 → `치명 0 · 경고 1 · 통과 · exit 0`
  (경고는 "세션이 없음 — 수집 전이면 정상")

### 완료 조건 대조
- [x] `data/session_plan.csv` 생성됨 (건수·시간·용량이 로그에 출력)
- [x] 정상 케이스 exit 0, 오타·좌표 누락 exit 1
- [x] `--join` 이 `labels_self_nodes.csv` 를 만든다
- [x] `node_layout.csv` · `session_log.csv` 는 헤더만 남은 상태
- [x] 스크립트에 서드파티 import 없음 — 세 파일 전부 stdlib만
      (`argparse, csv, math, os, sys, time, glob, statistics, collections, termios/tty`)

### 사소한 관찰 (고치지 않음)
`session_report.py:196-197` — 차폐 등급이 빈 문자열일 때 버킷 라벨이 `>4m/` 처럼 슬래시로 끝난다.
이미 치명 판정이 난 실행에서만 보이는 표시상의 문제라 손대지 않았다.

---

## 남은 일 (사람이 해야 하는 것)

| | 작업 | 비고 |
|---|---|---|
| ☐ | **IRB 문서 발송** | 심의 대기가 있으니 제일 먼저 |
| ☐ | **rp5를 접속 가능한 상태로** | 이게 풀려야 T1·T4가 산다. rp5 전원 확인 + 이 PC를 `TESTAP`(192.168.1.0/24)에 붙이거나 rp5를 현재 망으로 |
| ☐ | 펌웨어 Kconfig 확인 — `CONFIG_ESP_WIFI_CSI_ENABLED` | |
| ☐ | ESP **5대** 플래시, `TARGET_IP` = `192.168.1.2` | 04에서 1대만 고쳤다 |
| ☐ | **5노드 동시성 데스크 테스트** | GO 기준: 5대 전부 rate 85~115Hz · loss < 5% · rssi -40~-75 · `info->len == 128` |
| ☐ | 테스트 통과 후 `TARGET_IP` 를 rp5로 되돌려 재플래시 | |
| ☐ | 수집 일정을 **여러 날로 분할** | 순수 녹화만 3.2시간 × 피험자 수 |

---

## 사용자 확인이 필요한 결정

1. **이 레포를 git 저장소로 만들 것인가.** `.git` 이 없어 지시서의 브랜치·커밋 규칙
   (`chore/pre-collection`, 태스크 단위 커밋)을 하나도 지키지 못했다. `git init` 은 되돌리기
   번거로운 작업이라 임의로 하지 않았다. 원격이 이미 있다면 `git init` 이 아니라 clone 위에
   작업을 옮기는 쪽이 맞을 수 있다.

2. **`configs/m1_fall.yaml` 의 `pretrain` 블록을 활성 상태로 둘지.** 지시서에 적힌 그대로
   활성으로 넣었다. 그 결과 `m1_fall.yaml` 로 학습하면 **기본으로** `runs/best_B2_final.pt` 를
   불러온다. 스크래치 학습이 기본이어야 한다면 블록을 주석 처리해야 한다.
   (`runs/` 는 git-ignore 대상이라 체크포인트가 없는 환경에서는 `RuntimeError` 로 즉시 멈춘다.)

3. **`src/m1_fall/export.py` 의 패치 문서 템플릿을 고칠지.** 이게 T2에서 제일 중요한 미결 사항이다.
   `emit_rp5_patch()` 가 `PREPROCESS_AND_EXPORT.md` 를 매번 새로 쓰므로, export를 한 번만 돌리면
   T2의 `⚠️ 미검증` 표시가 전부 날아간다. T2 완료 조건이 "문서만, 코드 변경 없이"라 손대지 않았다.
   **표시를 유지하려면 `export.py:160-189` 의 템플릿 문자열을 같이 고쳐야 한다.**
   덧붙여 `.gitignore` 의 `runs/*` 때문에 그 파일은 추적조차 되지 않는다 —
   `!runs/rp5_patch/` 예외를 넣거나 `docs/` 로 옮기는 게 맞다고 본다. 임의로 옮기지 않았다.

4. **`CLAUDE.md` 의 "함정" 표를 어떻게 할지.** 그 표가 develop의 어느 시점을 근거로 쓰인 건지
   확인되지 않았고, 레포 안 다른 문서와 정면으로 충돌한다. T2 범위가 세 파일로 한정돼 있어
   `CLAUDE.md` 는 손대지 않았다. 정찰 후 셋 중 하나로 통일해야 한다.

---

## 커밋

**없다. 이 레포에는 `.git` 이 없다** (`git status` → `fatal: not a git repository`).
`chore/pre-collection` 브랜치도, 태스크 단위 커밋도 만들지 못했다.

이번 작업으로 바뀐 파일 (git이 생기면 이 단위로 쪼개 커밋하면 된다):

| 태스크 | 파일 | 종류 |
|---|---|---|
| T1 | `docs/RP5_DEVELOP_AUDIT.md` | 신규 |
| T2 | `README.md` | 수정 (문서) |
| T2 | `runs/rp5_patch/PREPROCESS_AND_EXPORT.md` | 수정 (문서, **현재 gitignore 대상**) |
| T3 | `configs/m1_fall.yaml` | 수정 (`pretrain` 블록 추가) |
| T3 | `src/m1_fall/config.py` | 수정 |
| T3 | `src/m1_fall/train.py` | 수정 |
| T5 | `data/session_plan.csv` | 신규 (생성물) |
| T5 | `data/node_layout.csv`, `data/session_log.csv` | 수정 (헤더만 남김) |
| T6 | `docs/PRECOLLECTION_STATUS.md` | 신규 (이 문서) |

T2와 T3은 서로 다른 파일만 건드리므로 커밋 분리에 충돌은 없다.
`data/labels*.csv` 는 건드리지 않았다. `configs/` 의 `tensor:` · `export:` 블록도 그대로다.

### 레포 밖 임시물 (커밋 대상 아님)

`data/raw/` 가 비어 있어 T3 검증용으로 CPU 전용 torch venv를 임시로 만들었다:
`C:\Users\compo\AppData\Local\Temp\m1v` (약 500MB). 검증 스크립트는 세션 스크래치패드에 있다.
재검증할 일이 없으면 지워도 된다:
```
rm -rf "C:/Users/compo/AppData/Local/Temp/m1v"
```
사용자의 전역 파이썬 환경에는 아무것도 설치하지 않았다 (거기엔 torch·pandas가 없다).
