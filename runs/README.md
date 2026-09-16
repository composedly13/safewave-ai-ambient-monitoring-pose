# runs/ — 체크포인트 · ONNX · rp5 패치 번들

**이 폴더는 전부 git 제외다** (`.gitignore`의 `runs/*`, `*.pt`, `*.onnx`).
`.gitkeep`만 추적된다. 여기 있는 파일은 백업·전달을 수동으로 챙겨야 한다.

```
runs/
├── best.pt                     학습이 저장하는 최고 체크포인트 (5노드)
├── train_summary.json          학습 요약 (best epoch, history)
├── m1_wifi_pose.onnx           export.py 산출물 — rp5에 넘길 것 (5노드)
├── legacy_singlenode/          ⚠️ 단일노드 레거시 — 5노드에선 로드 불가
└── rp5_patch/                  export.py가 생성하는 rp5 패치 번들
```

## ⚠️ `legacy_singlenode/` — 단일노드 레거시, 5노드 모델에 로드 불가

| 파일 | 정체 |
|---|---|
| `best_B2_final.pt` | 2026-08-25 사전학습 체크포인트. `conv1.weight = (16, 1, 3, 3)` — **입력채널 1** |
| `m1_wifi_pose.onnx` | 위 체크포인트의 ONNX. `INPUT csi_data [batch, 1, 64, 100]` |

5노드 모델은 `conv1.weight = (16, 5, 3, 3)`을 요구하므로 **`strict=True` 로드가
반드시 실패한다**(의도된 것 — `CLAUDE.md` 규칙 6). `config.py`도 `pretrain` 블록과
`n_nodes != 1` 조합을 명시적으로 거부한다.

**5노드는 스크래치 재학습이 정상 경로다.** 이 둘로 워밍스타트하지 마라.

삭제하지 않는 이유: 04 rp5 실측(2026-08-02)의 근거 아티팩트이고, 단일노드
`n_nodes: 1` 설정으로 돌아갈 경우 여전히 유효하기 때문이다.

## 자주 쓰는 명령

```powershell
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe -m m1_fall.train  --config configs/m1_fall.yaml
.venv\Scripts\python.exe -m m1_fall.export --config configs/m1_fall.yaml --ckpt runs/best.pt
```

`export.py`는 ONNX를 쓰고, **그래프에 Sigmoid가 없는지 검사**하고, torch↔onnxruntime
parity를 확인한 뒤 `rp5_patch/`를 새로 생성한다.
`rp5_patch/` 안의 두 파일은 **매번 덮어써지므로 손으로 고치지 마라** — 고칠 곳은
`src/m1_fall/export.py`의 생성기다.
