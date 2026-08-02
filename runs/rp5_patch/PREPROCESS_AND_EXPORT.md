# rp5 inference-side edits (ship together with the new ONNX)

Two edits switch rp5 from the random `DTPoseModel` to this trained CNN-GRU.
(Unlike M2, no main.py buffer change is needed — M1 infers on a single
1-node x 100-frame window already assembled upstream.)

## 1. ai/experts/m1_wifi_pose.py._preprocess  (192 -> 64 subcarriers)

The model input is (1, 1, 64, 100). block_raw is already per-frame
peak-normalized on the ESP, so DO NOT re-normalize and DO NOT reconstruct absolute
amplitude. Only the subcarrier count changes: the legacy PulseFi 192 -> 64.

Before:
```python
N_SUBCARRIERS = 192
...
def _preprocess(self, signal_data):
    data = np.asarray(signal_data, dtype=np.float32)
    # ... reshape / pad to (1, 1, 192, 100) ...
```

After:
```python
N_SUBCARRIERS = 64     # HARD CONTRACT: block_raw 64 subcarriers (was 192)
N_FRAMES = 100         # fixed 1 s @ 100 Hz
...
def _preprocess(self, signal_data):
    data = np.asarray(signal_data, dtype=np.float32)
    # ... reshape / pad to (1, 1, 64, 100) ...
```
Everything downstream is identical — `_infer_onnx` already feeds name
`csi_data` and reads the else-branch output as `out.reshape(-1)[0]`
(then sigmoid), which is exactly our `fall_logit` (B,1).

## 2. scripts/export_m1_wifi_pose_onnx.py

Replace with `export_m1_wifi_pose_onnx.py` in this folder. Changes vs the old script:
  - `dummy_input` (1, 1, 192, 100) -> (1, 1, 64, 100)
  - the random `DTPoseModel` -> inline CNN-GRU (M1FallNet) loading this repo's
    trained `state_dict` (`--ckpt best.pt`)
  - output name `fall_logit`, opset 17, batch-only dynamic axis
