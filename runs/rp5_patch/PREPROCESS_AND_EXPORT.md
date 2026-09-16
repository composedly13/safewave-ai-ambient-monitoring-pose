# rp5 inference-side edits (ship together with the new ONNX)

> ⚠️ **rp5 미검증 — 적용 전 반드시 `develop`을 직접 읽어라.**
> 이 문서는 이 학습 레포가 **내보내는 계약**을 근거로 자동 생성된 것이고,
> rp5의 **실제 코드를 읽고 쓴 것이 아니다.** 2026-08-25 정찰(T1)에서 rp5
> (`192.168.1.2`)에 접속하지 못했고, 그 뒤 5노드로 전환하면서 확인 범위가 더 넓어졌다.
> 상세: `docs/archive/RP5_DEVELOP_AUDIT.md`.
> 아래는 **확정된 작업 목록이 아니라 확인해야 할 질문 목록**이다.

## 이 레포가 내보내는 계약 (확정)

```
INPUT   csi_data    (B, 5, 64, 100)   float32
        = 노드 5 × 서브캐리어 64 × 100프레임 (1초 @ 100Hz)
        노드축은 conv 입력채널(조기융합). 서브캐리어축 192 concat 아님.
OUTPUT  fall_logit  (B, 1)   pre-sigmoid logit
        그래프에 Sigmoid 노드 없음 — rp5가 sigmoid를 한다.
opset   17 · 동적 축은 batch 뿐
```

## rp5에서 확인해야 할 것 (전부 미검증)

### Q1. `ai/experts/m1_wifi_pose.py._preprocess` 가 `(1,5,64,100)`을 만드는가?

입력은 노드마다 64 서브캐리어이고, **노드는 채널축으로 쌓는다.**
`block_raw`는 ESP에서 이미 per-frame peak 정규화돼 있으므로 **재정규화 금지,
절대진폭 복원 금지** (노드별로 독립 정규화된 상태 그대로 써야 한다).

기대 형태:
```python
N_NODES = 5          # 노드축 = 채널축
N_SUBCARRIERS = 64   # 노드당 64 (레거시 192 아님)
N_FRAMES = 100       # 1초 @ 100Hz 고정
...
def _preprocess(self, signal_data):
    data = np.asarray(signal_data, dtype=np.float32)
    # -> (1, 5, 64, 100)
```

> 과거 스냅샷(2026-06-05)은 `192 * 100`을 하드코딩했고, `CLAUDE.md`의 함정 표는
> "develop은 ONNX에서 shape을 직접 읽으므로 수정 불필요"라고 적고 있다. **서로 모순이다.**
> 직접 읽어서 판정하라. shape을 ONNX에서 읽더라도, **노드를 어떤 순서·어떤 축으로
> 채우는지**는 별개 문제다.

### Q2. `ai/main.py`의 롤링 버퍼가 5노드를 **무엇으로 시간 정렬**하는가?  ← 가장 위험

이 레포는 **`stream_id`의 epoch**(rp5 수신 시각 = 유일한 공통 시계)로 100Hz 그리드를
깔고 노드를 ±5ms 최근접 슬롯에 맞춘다. `ts_ms`는 **보드마다 다른 장치 시계**(32비트 랩)라
노드 간 정렬에 쓰면 노드들이 서로 어긋난다.

rp5가 `ts_ms`로 정렬한다면 **shape은 맞는데 내용이 틀린 입력**이 들어간다 —
추론이 실패하지 않고 조용히 엉뚱한 점수를 낸다. 이게 가장 잡기 어려운 실패다.

### Q3. 노드가 빠진 시점에 rp5는 무엇을 채우는가?

이 레포 로더는 **0으로 패딩**한다. rp5가 직전 프레임을 복제하거나, 노드가 하나라도
없으면 추론을 건너뛴다면 학습 분포와 추론 분포가 갈린다.

### Q4. `_infer_onnx`가 sigmoid를 **몇 번** 하는가?

이 레포는 `fall_logit`을 **pre-sigmoid**로 내보낸다 (`CLAUDE.md` 규칙 3).
- rp5가 sigmoid를 **안 하면**: logit이 그대로 점수가 된다. 2026-06-05 스냅샷의
  `_infer_onnx`에는 sigmoid가 없고 `np.clip(score, 0.0, 1.0)`만 있었다 — 그 경우
  음수 logit은 전부 `0.0`, 양수는 `1.0` 근처로 잘려 **임계 0.7이 무의미**해진다.
- sigmoid가 **두 번** 걸리면: 출력이 0.5~0.73으로 뭉개진다 (`CLAUDE.md` 규칙 2).

어느 쪽인지 확인하기 전에는 **ONNX에 Sigmoid를 넣지도 빼지도 말고 현행(미포함)을 유지**하라.

## rp5 export 스크립트 교체

`scripts/export_m1_wifi_pose_onnx.py`를 이 폴더의 `export_m1_wifi_pose_onnx.py`로 교체.
기존 스크립트 대비 달라지는 점:
  - `dummy_input` (1, 1, 192, 100) -> (1, 5, 64, 100)
  - 랜덤 초기화 `DTPoseModel` -> 인라인 CNN-GRU (M1FallNet), 이 레포의 학습된
    `state_dict`를 `--ckpt best.pt`로 로드 (strict=True)
  - 출력 이름 `fall_logit`, opset 17, 동적 축은 batch 뿐

생성 근거 체크포인트: `runs/best.pt (5노드 학습 후 생성 예정)`
