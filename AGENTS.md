# SafeWave-AI · M1 낙상감지 학습 레포 — 에이전트 안내

> **정식 지침은 [`CLAUDE.md`](CLAUDE.md) 하나다. 작업 전에 그것을 읽어라.**
>
> 이 파일은 예전에 `CLAUDE.md`의 **사본**이었다. 2026-09-16 5노드 전환 때 사본 쪽만
> 낡아서 규칙 1·7이 코드와 정반대를 가리키고 있었다. 그래서 사본을 없애고
> 포인터로 바꿨다. **여기에 규칙을 다시 적지 마라** — 또 갈라진다.

## 틀리면 배포가 조용히 부서지는 것 (요약 — 전문은 `CLAUDE.md`)

- 입력 텐서는 **`(B, 5, 64, 100)`** = 노드 5 × 서브캐리어 64 × 100프레임(1초 @ 100Hz).
  서브캐리어는 64 고정, **노드는 별도 축**(첫 conv의 입력채널)이다.
  서브캐리어축에 노드를 이어붙인 **192 concat은 금지**(PulseFi 레거시).
- 5노드 시간 정렬은 **`stream_id`의 epoch** 기준. `ts_ms`는 보드마다 다른 장치 시계라
  노드 간 정렬에 쓰면 어긋난다.
- 모델 출력은 **`fall_logit (B,1)` pre-sigmoid**. **ONNX에 Sigmoid를 넣지 마라**
  (rp5가 한다 → 이중 적용).
- 정규화는 **per-frame peak**만, **노드별·프레임별 독립**.
- 체크포인트 로드는 **`strict=True`**. 단일노드 레거시 체크포인트
  (`runs/legacy_singlenode/`)는 5노드 모델에 로드되지 않는다 — 재학습이 정상 경로다.
- `CSI_FS = 100` 고정, ONNX opset **17**, input 이름 `csi_data`.

## rp5에 대해 말할 때

**이 레포에서는 rp5 코드를 확인할 수 없다.** rp5가 5노드 입력을 어떻게 만드는지
(노드 정렬 기준, 결측 노드 처리, sigmoid 적용 횟수)는 **전부 미검증**이다.
추측해서 단정하지 말고 "미검증"으로 남겨라. 확인해야 할 질문 목록은
[`README.md`](README.md)의 "rp5 sync" 절과 `runs/rp5_patch/PREPROCESS_AND_EXPORT.md`에 있다.

## 실행 환경

레포 안 `.venv` (Python 3.11, torch 2.14 CPU). **시스템 3.14는 numpy DLL이 깨져 있다.**

```powershell
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe -m m1_fall.model
```
