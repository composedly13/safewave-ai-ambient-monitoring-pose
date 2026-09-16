> ⚠️ **2026-08-25 시점 기록. 현재 지침이 아니다.**
> 이 문서는 입력이 per-node `(B,1,64,100)`이던 시절에 쓰였다. 2026-09-16에 M1은
> **5노드 조기융합 `(B,5,64,100)`**으로 전환됐다 — 현행 지침은 `CLAUDE.md`와
> `docs/COLLECTION_5NODE.md`다. 여기 적힌 텐서 형태·융합 주체·로더 동작은 낡았다.
> 이력 보존용으로 남긴다. **`RP5_DEVELOP_AUDIT.md`의 rp5 정찰 기준선은 여전히 유효하다.**

---

# 작업 지시서 — M1 현장 수집 전 데스크 작업

**발행** 2026-08-25 · **수행** Claude Code · **레포** `D:\safewave-ai-ambient-monitoring-pose`

목표: **05 자체 데이터 수집(현장)에 나가기 전에, 책상에서 끝낼 수 있는 모든 것을 끝낸다.**

작업 전 `CLAUDE.md`를 반드시 읽어라. 절대 규칙 8개와 함정이 거기 있다.

---

## 역할 분담

| | 담당 |
|---|---|
| T1 rp5 develop 정찰 | **Claude Code** |
| T2 레포 문서 정정 | **Claude Code** |
| T3 `train.py --init-ckpt` | **Claude Code** |
| T4 per-node 추론 루프 (조건부) | **Claude Code** |
| T5 수집 계획표 + 리허설 | **Claude Code** |
| T6 최종 보고 | **Claude Code** |
| — 펌웨어 5대 플래시 | 사람 (물리 작업) |
| — 5노드 동시성 데스크 테스트 | 사람 (물리 작업) |
| — IRB 문의 | 사람 |

---

## 공통 규칙

**브랜치·커밋**
- 작업 브랜치 `chore/pre-collection` 을 새로 파고 거기서만 작업한다
- `main` 에 직접 push 금지. 되돌릴 일이 생기면 `revert`를 쓴다 (`reset --hard` + force push 금지 — 이미 한 번 사고가 났다)
- 커밋은 태스크 단위로 쪼갠다. 한 커밋에 T2와 T3을 섞지 마라
- 커밋 메시지는 한국어로, 무엇을 왜 바꿨는지 한 줄

**정직성 — 이게 제일 중요하다**
- **확인하지 않은 것을 확인한 것처럼 쓰지 마라.** 이 프로젝트는 "GitHub main만 보고 세운 가정"이 대부분 틀려서 한 번 크게 헤맸다
- 파일을 못 읽었으면 "못 읽음", 확신이 없으면 "미확인"이라고 적어라
- 추측을 적어야 할 때는 반드시 `추정:` 접두를 붙여라

**금지**
- `configs/*.yaml` 의 `tensor:` 블록 수정 금지 (64/100/100Hz는 하드 컨트랙트)
- `export:` 에 sigmoid 관련 옵션 추가 금지 (`CLAUDE.md` 규칙 2)
- `scripts/` 3개 파일에 서드파티 의존성 추가 금지 (rp5에서 돌아야 함)
- `data/labels*.csv` 내용 수정 금지 (T5에서 새 파일을 만드는 건 허용)
- 학습을 실제로 돌리지 마라 (`data/raw/` 가 비어 있어 어차피 실패한다)

---

## T1 · rp5 develop 정찰

**목적** 배포 쪽에서 유일하게 확인 안 된 부분을 확정한다. T4의 필요 여부가 여기서 갈린다.

**중요: 읽기 전용이다. rp5의 어떤 파일도 수정하지 마라.**

```bash
ssh csi@192.168.1.2 'cd ~/safewave && git branch --show-current && git log --oneline -10'
ssh csi@192.168.1.2 'cd ~/safewave && cat ai/experts/m1_wifi_pose.py'
ssh csi@192.168.1.2 'cd ~/safewave && grep -n "M1_MAX_NODES\|deque\|wifi_pose\|_decode_csi\|active\|for node" ai/main.py'
ssh csi@192.168.1.2 'cd ~/safewave && grep -n "M1_MAX_NODES\|EXPERT\|ORT_USE_GPU" .env'
ssh csi@192.168.1.2 'cd ~/safewave && sed -n "1,60p" sensing/main.py'
```

`ai/main.py` 가 길면 M1 추론 구간 전후 80줄을 따로 떠서 읽어라.

**답해야 할 질문 5개**

| # | 질문 | 왜 중요한가 |
|---|---|---|
| Q1 | sigmoid가 **어디서 몇 번** 적용되는가 | 0번이면 logit이 그대로 나가 `clip(0,1)`에 뭉개진다. 2번이면 이중 적용 |
| Q2 | `_preprocess`가 입력 shape을 ONNX에서 읽는가, 하드코딩인가 | 192 하드코딩이 남았으면 수정 대상 |
| Q3 | `M1_MAX_NODES` 현재값과 텐서 조립 방식 | concat이면 노드 수만큼 shape이 커진다 |
| Q4 | **노드마다 ONNX를 돌리는 루프가 있는가** | 없으면 T4 필요 |
| Q5 | `ai:m1:latest` 가 노드별로 분리돼 있는가 | 단일 키면 5노드가 서로를 덮어쓴다 |

**산출물** `docs/RP5_DEVELOP_AUDIT.md`

형식:
```markdown
# rp5 develop 감사 (YYYY-MM-DD)
브랜치: <name> / 커밋: <hash>

## Q1 sigmoid
**답:** ...
**근거:** `ai/experts/m1_wifi_pose.py:LN` 의 다음 코드
```python
<실제 코드 인용>
```
(Q2~Q5 동일 형식)

## 판정
- T4 필요 여부: 필요 / 불필요 — 근거
- 추가로 발견한 문제: ...
```

**완료 조건**
- [ ] Q1~Q5 전부에 **실제 코드 인용**과 파일:줄번호가 붙어 있다
- [ ] 인용 없이 서술만 한 항목이 없다
- [ ] T4 필요 여부가 명시돼 있다

**ssh가 안 되면** 그 사실을 기록하고 T2로 넘어가라. IP가 바뀌었을 수 있다 (과거 `192.168.0.13` → `192.168.1.2`). 사용자에게 보고하되 다른 태스크를 멈추지는 마라.

---

## T2 · 레포 문서 정정

**목적** 지금 세 곳이 틀린 지시를 하고 있다. 다음 사람이 또 헛짚는다.

**고칠 곳**

1. `runs/rp5_patch/PREPROCESS_AND_EXPORT.md`
   - "1. `_preprocess` 192 → 64" — develop이 ONNX shape을 직접 읽으면 불필요. **T1 Q2 결과에 따라** 수정하거나 조건부 문구로 바꾼다
   - "(Unlike M2, no main.py buffer change is needed — ... already assembled upstream)" — **틀렸다.** develop에 이미 롤링 버퍼가 있고, 그 서술은 근거가 없다. 삭제하고 T1에서 확인한 실제 구조로 대체
2. `README.md` 의 "⚠️ rp5 sync requirement" 섹션 — 두 파일 동시 수정 요구가 이미 불필요하다. T1 결과 기준으로 다시 쓴다
3. `README.md` 의 하드 컨트랙트 표에 **2행 추가**
   - `M1_MAX_NODES` 정책 — M1은 per-node 추론. 배포에서 노드 융합은 M5 담당
   - **sigmoid는 rp5가 적용. ONNX에 굽지 말 것** — 이중 적용 시 출력이 0.5~0.73으로 뭉개진다

**금지** 하드 컨트랙트 표의 **기존 행을 수정하지 마라.** 추가만 한다.

**완료 조건**
- [ ] 세 파일 모두 T1의 실제 관찰과 모순되지 않는다
- [ ] T1을 못 했으면 해당 문구를 지우지 말고 `⚠️ 미검증 (2026-08-25)` 표시만 붙인다
- [ ] `git diff` 에 코드 파일 변경이 없다 (문서만)

---

## T3 · `train.py` 사전학습 체크포인트 진입로

**목적** `runs/best_B2_final.pt` 를 백본으로 이어받는 경로. 06 파인튜닝의 전제다.

**설계 (06 문서 기준)**

`configs/m1_fall.yaml` 에 **선택적** 블록 추가:
```yaml
pretrain:
  ckpt: runs/best_B2_final.pt
  load: full             # full | backbone
  freeze_epochs: 5       # 초반 N에폭 conv 동결, GRU+헤드만 학습
  backbone_lr_mult: 0.1  # 해동 후 백본 lr = lr * 0.1
```

`src/m1_fall/config.py`:
- `PretrainCfg` 데이터클래스 추가
- `Config.pretrain: Optional[PretrainCfg] = None`
- `load_config`에서 `raw.get("pretrain")` 로 **없으면 None** (기존 config가 그대로 로드돼야 한다)
- `validate()`에 검사 추가: `load`는 `full`/`backbone` 중 하나, `freeze_epochs >= 0`, `0 < backbone_lr_mult <= 1`

`src/m1_fall/train.py`:
- CLI `--init-ckpt` 추가. **CLI가 config보다 우선**
- 모델 생성 직후 로드. **`strict=True` 고정** (`CLAUDE.md` 규칙 6)
- 로드 직후 반드시 출력:
  ```
  [pretrain] runs/best_B2_final.pt | keys 27/27 | epoch 14 | pr_auc 0.6961
  ```
- 로드된 키가 0개면 `RuntimeError`로 즉시 중단
- `freeze_epochs > 0` 이면 `model.convs`의 `requires_grad=False`, N에폭 후 해동하고 **optimizer param group을 다시 구성**해 백본에 `lr * backbone_lr_mult` 적용
- 해동 시점을 로그에 남긴다: `[pretrain] epoch 6: 백본 해동, backbone_lr=1.0e-04`

**완료 조건**
- [ ] `pretrain` 블록 없는 기존 config가 그대로 로드된다 (`python -m m1_fall.dataset configs/m1_fall_pretrain.yaml` 이 config 에러 없이 시작)
- [ ] `--init-ckpt runs/best_B2_final.pt` 로 27/27 로드가 로그에 찍힌다
- [ ] 일부러 손상시킨 state_dict로는 `RuntimeError`가 난다
- [ ] `strict=False` 가 코드 어디에도 없다
- [ ] 동결/해동 시 optimizer param group이 실제로 바뀌는지 확인한 로그가 있다

**검증 방법 (데이터 없이)**
`data/raw/`가 비어 있어 학습은 못 돌린다. 대신 로드·동결 부분만 떼서 확인하는 임시 스크립트를 `/tmp`에 만들어 돌리고, **레포에는 커밋하지 마라.** 검증 결과는 T6 보고서에 적는다.

---

## T4 · per-node 추론 루프 (조건부)

**T1 Q4가 "루프 없음"일 때만 수행한다. "있음"이면 이 태스크를 건너뛰고 T6에 그렇게 적어라.**

**문제** 학습 계약은 per-node `(1,1,64,100)`인데 develop은 `(M1_MAX_NODES × 64, 100)`으로 concat한다.

| `M1_MAX_NODES` | 결과 |
|---|---|
| 5 | `(320,100)` → ONNX `(64,100)`과 불일치 → 터짐 |
| 1 | 노드 1개만 추론, 나머지 4개 무시 |
| **노드 루프** | 정답 — 노드마다 ONNX 1회, 융합은 M5 |

**작업 위치** rp5 레포. `~/safewave`에서 **새 feature 브랜치**를 파고 작업한다. `develop`·`main` 직접 수정 금지.

**요구사항**
- 활성 노드(버퍼가 100프레임 찬 노드)마다 ONNX를 1회씩 돌린다
- 결과를 `{node_id: {fall_score, latency_ms}}` 로 모아 M5에 넘긴다
- `ai:m1:latest` 가 단일 키면(T1 Q5) 노드별로 분리하는 안을 **제안만** 하고, 실제 변경은 사용자 승인 후에 한다 — 다른 서비스(api, monitor.html)가 이 키를 읽는다
- 기존 concat 경로는 지우지 말고 설정으로 전환 가능하게 남긴다 (되돌릴 수 있어야 한다)

**검증 — 하드웨어 없이 된다**
```bash
python sensing/simulator.py --nodes 5 --rate 100
redis-cli GET ai:m1:latest
redis-cli --raw XREVRANGE ai:result + - COUNT 3
```
- [ ] 5개 노드가 전부 추론되는가
- [ ] shape 에러 없이 도는가
- [ ] **CPU 사용률을 다시 측정**한다. 04의 "5노드 환산 13%"는 concat 1회 기준이라 무효다
- [ ] 노드당 latency_ms 기록

**완료 조건**
- [ ] simulator 5노드에서 shape 에러 0
- [ ] 노드 5개 각각의 `fall_score`가 관측된다
- [ ] CPU·latency 실측치가 T6 보고서에 있다
- [ ] rp5의 `develop`·`main`에 커밋이 올라가지 않았다

---

## T5 · 수집 계획표 + 리허설

**목적** 현장에서 양식 때문에 시간을 쓰지 않게 한다.

```bash
python3 scripts/session_report.py --plan 200 180
```

그다음 **가짜 세션 3건**(FALL 2 + HARDNEG 1)을 손으로 채워 전체 흐름을 돌려본다:
```bash
python3 scripts/session_report.py
python3 scripts/session_report.py --join data/labels_self.csv
```

- 노드 좌표는 임의로 넣되 **거리대가 섞이게** (≤2m / 2-4m / >4m 각각 하나 이상)
- `occlusion_by_node` 는 `clear`/`furniture`/`door`/`wall` 4종만 허용된다. 오타를 일부러 넣어 스크립트가 잡는지도 확인
- 확인 후 **가짜 데이터는 지우고 헤더만 남긴 상태로 커밋**한다

**완료 조건**
- [ ] `data/session_plan.csv` 생성됨 (녹화 건수·예상 시간·용량이 로그에 출력)
- [ ] `session_report.py` 가 정상 케이스에서 exit 0, 오타·좌표 누락에서 exit 1
- [ ] `--join` 이 `labels_self_nodes.csv` 를 만든다
- [ ] `data/node_layout.csv` · `data/session_log.csv` 는 **예시행이 지워진 헤더 상태**로 커밋
- [ ] 스크립트에 서드파티 import가 없다

---

## T6 · 최종 보고

**산출물** `docs/PRECOLLECTION_STATUS.md`

```markdown
# 현장 전 데스크 작업 결과 (YYYY-MM-DD)

## 요약
| 태스크 | 상태 | 비고 |
|---|---|---|
| T1 rp5 정찰 | 완료/부분/실패 | |
...

## T1 핵심 발견
(Q1~Q5 한 줄 요약 + T4 필요 여부 판정)

## 남은 위험
(사람이 해야 하는 것: 펌웨어 5대, 5노드 테스트, IRB)

## 사용자 확인이 필요한 결정
(내가 판단하지 않고 남겨둔 것)

## 커밋
(브랜치명 + 커밋 해시·제목 목록)
```

**완료 조건**
- [ ] 각 태스크의 완료 조건 체크리스트가 실제 결과로 채워져 있다
- [ ] 실패·미확인 항목이 숨겨지지 않고 그대로 적혀 있다
- [ ] 사용자 판단이 필요한 항목이 따로 모여 있다

---

## 사람이 할 일 (참고 — Claude Code는 수행하지 않는다)

| | 작업 | 비고 |
|---|---|---|
| ☐ | IRB 문의 발송 | **회신 대기가 있으니 제일 먼저** |
| ☐ | 펌웨어 Kconfig 확인 → `CONFIG_ESP_WIFI_CSI_ENABLED` | |
| ☐ | ESP **5대** 플래시, `TARGET_IP` = `192.168.1.2` | 04에서 1대만 고쳤다. 나머지는 팀원 PC를 향할 수 있음 |
| ☐ | **5노드 동시성 데스크 테스트** | `TARGET_IP`를 노트북으로 두고 `tools/udp_listener.py`. 1대 → 3대 → 5대로 늘려가며 각 2분 |
| ☐ | 테스트 통과 후 `TARGET_IP`를 rp5로 되돌려 재플래시 | |

**5노드 테스트 GO 기준:** 5대 전부 rate 85~115Hz · loss < 5% · rssi -40~-75 · `info->len == 128`

**NO-GO면** AP 채널 변경 → 노드 간 거리 → **노드 수 축소**(3노드 100Hz가 5노드 60Hz보다 낫다. 학습 코드는 노드 수를 가정하지 않는다) 순으로 시도. `CSI_FS`를 낮추는 건 최후이며 백엔드팀과 재논의 사안이다.

상세 절차는 `docs/COLLECTION_5NODE.md`.
