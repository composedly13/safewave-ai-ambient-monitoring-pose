> ⚠️ **2026-08-25 시점 기록. 현재 지침이 아니다.**
> M1 입력은 2026-09-16에 per-node `(B,1,64,100)` → **5노드 조기융합 `(B,5,64,100)`**으로
> 전환됐다. 이 문서의 텐서 서술은 그 이전 것이다.
>
> **다만 이 문서의 rp5 정찰 기준선(무엇을 확인했고 무엇을 확인 못 했는지)은 여전히 유효하다.**
> rp5는 그때도 지금도 **확인되지 않았고**, 5노드 전환으로 확인해야 할 범위가 오히려 넓어졌다
> (노드를 무엇으로 시간 정렬하는가, 결측 노드에 무엇을 채우는가).
> 정찰을 재개할 때 **이 문서의 질문 목록부터 시작하라** — `README.md`의 "rp5 sync" 절과
> `runs/rp5_patch/PREPROCESS_AND_EXPORT.md`가 5노드 기준으로 그 질문을 갱신해 두었다.

---

# rp5 develop 감사 (2026-08-25)

브랜치: **확인 못 함** / 커밋: **확인 못 함**

## 결론 먼저

**rp5에 접속하지 못했다. Q1~Q5 중 develop 기준으로 답할 수 있는 항목은 0개다.**

아래 "부록 A"에 로컬에 남아 있던 배포 레포 스냅샷(`D:\safewave-ai-ambient-monitoring`,
파일 mtime **2026-06-05**)에서 읽은 실제 코드를 인용해 두었다. 이건 develop이 아니고,
develop보다 최소 한 세대 이전이라는 증거가 있다(부록 B). **답으로 쓰면 안 된다.**

---

## 접속 시도 기록

| 시도 | 결과 |
|---|---|
| `ssh csi@192.168.1.2` | `connect to host 192.168.1.2 port 22: Connection timed out` |
| `ping 192.168.1.2` | 요청 시간 만료, 패킷 2/2 손실 (100%) |
| `ping 192.168.0.13` (과거 IP) | `대상 호스트에 연결할 수 없습니다` — 게이트웨이가 unreachable 응답 |
| TCP 22 스캔: `192.168.0.1 / .13 / .40 / .44 / .61` (ARP 테이블에 살아 있던 전 호스트) | 전부 closed |
| Tailscale (`tailscale status`) | 노드 2개(`gram`, `desktop`) 전부 **offline**, 좌표 서버 연결 실패. rp5 노드 없음 |

**원인:** 이 PC의 Wi-Fi는 `192.168.0.41/24`, 게이트웨이 `192.168.0.1`.
rp5는 문서상 `192.168.1.2`, 즉 **다른 서브넷**이고 라우트가 없다.
`192.168.0.x` 대역에는 SSH가 열린 호스트가 하나도 없다.

**추정:** rp5가 붙는 AP(`TESTAP`, `192.168.1.0/24`)와 이 PC가 붙은 공유기(`192.168.0.0/24`)가
서로 다른 망이다. 같은 AP에 붙거나 rp5를 켜야 정찰이 가능하다.

---

## Q1~Q5

| # | 질문 | 답 |
|---|---|---|
| Q1 | sigmoid가 어디서 몇 번 적용되는가 | **확인 못 함** |
| Q2 | `_preprocess`가 입력 shape을 ONNX에서 읽는가, 하드코딩인가 | **확인 못 함** |
| Q3 | `M1_MAX_NODES` 현재값과 텐서 조립 방식 | **확인 못 함** |
| Q4 | 노드마다 ONNX를 돌리는 루프가 있는가 | **확인 못 함** |
| Q5 | `ai:m1:latest` 가 노드별로 분리돼 있는가 | **확인 못 함** |

---

## 판정

- **T4 필요 여부: 판정 불가.** 작업 지시서상 T4는 "Q4 = 루프 없음"일 때만 수행한다.
  Q4를 확인 못 했으므로 **T4는 수행하지 않았다.** 덧붙여 T4는 rp5 레포에 브랜치를 파고
  `sensing/simulator.py`로 검증하는 작업이라, rp5 접속 없이는 애초에 착수 자체가 불가능하다.

- **추가로 발견한 문제 (이게 제일 중요하다):**
  **이 레포 안의 문서 두 개가 develop에 대해 서로 반대되는 주장을 하고 있고, 둘 다 미검증이다.**

  | 항목 | `CLAUDE.md` "함정" 표의 주장 | `README.md` / `runs/rp5_patch/PREPROCESS_AND_EXPORT.md` 의 주장 |
  |---|---|---|
  | `_preprocess` shape | "**ONNX에서 shape을 직접 읽음**" → 수정 불필요 | "192 → 64로 **고쳐야 함**" (README:70, PREPROCESS:7) |
  | main.py 롤링 버퍼 | "**이미 구현돼 있음** (노드별 deque)" | "no main.py buffer change is needed — 이미 upstream에서 조립됨" (PREPROCESS:4-5) |
  | sigmoid | rp5 `_infer_onnx`가 적용 (규칙 2) | "(then sigmoid)" (PREPROCESS:33), "→ sigmoid → fall_score" (README:16) |

  2026-06-05 스냅샷은 **`_preprocess` 192 하드코딩 + sigmoid 없음 + 버퍼 없음** 쪽이다(부록 A).
  즉 `CLAUDE.md`의 함정 표가 서술하는 develop과 다르다. 셋 중 무엇이 현재 develop인지는
  **rp5를 직접 읽기 전까지 확정할 수 없다.** T2에서 세 파일 모두에 `⚠️ 미검증` 표시를 붙였다.

---

# 부록 A · 로컬 스냅샷(2026-06-05)에서 실제로 읽은 코드

> ⚠️ **경고: 아래는 develop이 아니다.** 출처는 `D:\safewave-ai-ambient-monitoring`
> (git 메타데이터 없음, 브랜치 불명, 모든 파일 mtime `2026-06-05`, `VERSION` = `ver.0.0.1`).
> Q1~Q5의 답으로 인용 금지. 정찰 재개 시 대조용 기준선으로만 쓸 것.

## A1 · sigmoid — 스냅샷에는 **0번**

`ai/experts/m1_wifi_pose.py:39-51`

```python
    def _infer_onnx(self, data):
        input_name = self.session.get_inputs()[0].name
        output = self.session.run(None, {input_name: data})[0]
        output_arr = np.asarray(output)
        if output_arr.ndim == 3 and output_arr.shape[-1] == 2:
            # DT-Pose keypoint 좌표 [1, 17, 2]를 fall score로 매핑
            keypoints = output_arr[0]
            motion_energy = float(np.mean(np.linalg.norm(keypoints, axis=-1)))
            score = motion_energy / 1.5
        else:
            score = float(output_arr.reshape(-1)[0])
        score = float(np.clip(score, 0.0, 1.0))
        return {"fall_score": score, "fall_detected": score >= 0.7}
```

sigmoid 호출이 없다. `np.clip(score, 0, 1)` 뿐이다.
레포 전체 grep에서 `sigmoid` 히트는 M2 export 스크립트 2줄이 전부다:

```
scripts/export_m2_frenel_vital_onnx.py:51:            nn.Sigmoid(),  # 0-1 정규화
scripts/export_m2_frenel_vital_onnx.py:59:            nn.Sigmoid(),  # 0-1 정규화
```

이 코드에 우리 `fall_logit`(pre-sigmoid)을 그대로 먹이면 음수 logit은 전부 0.0으로,
양수 logit은 1.0 근처로 잘린다. `CLAUDE.md` 규칙 2가 상정한 rp5 동작과 다르다.

## A2 · `_preprocess` — 스냅샷은 **192 하드코딩**

`ai/experts/m1_wifi_pose.py:26-37`

```python
    def _preprocess(self, sensor_data):
        data = np.asarray(sensor_data, dtype=np.float32)
        if data.ndim == 4:
            return data

        flat = data.reshape(-1)
        target = 192 * 100
        if flat.size < target:
            flat = np.pad(flat, (0, target - flat.size), mode="constant")
        else:
            flat = flat[:target]
        return flat.reshape(1, 1, 192, 100)
```

ONNX 세션에서 shape을 읽지 않는다. 4-D 입력이면 통과시키지만,
M1에는 4-D가 들어오지 않는다(A4 참조) → 항상 `(1,1,192,100)`으로 간다.

## A3 · `M1_MAX_NODES` — 스냅샷에 **존재하지 않음**

레포 전체 grep 결과 히트 0건. 노드 concat 코드도 없다.
`ai/main.py`는 `csi:raw` 스트림 메시지를 **한 건씩** 처리하고, 각 메시지는 노드 하나에 속한다.

`ai/main.py:652-665`

```python
            for _stream, messages in entries:
                for msg_id, fields in messages:
                    try:
                        node_id = int(fields.get(b"node", 0))
                    except Exception:
                        node_id = 0
                    ...
                    if active_nodes and node_id not in active_nodes and node_id != 0:
                        last_id = msg_id
                        continue
```

## A4 · 노드별 추론 루프 — 스냅샷은 "메시지당 1회"

노드를 순회하는 루프는 없다. 대신 스트림 메시지 하나 = 노드 하나 = 전문가 1회 실행이다.

`ai/main.py:681-690`

```python
                    raw = fields.get(b"data", b"")
                    input_data = _decode_csi_payload(raw)

                    context_window = _build_context_window(r, ts_ms)
                    audio_events = _load_recent_audio_events(r, node_id, ts_ms, M4_AUDIO_WINDOW_MS)
                    expert_inputs, audio_result = _build_expert_inputs(input_data, audio_events)
                    expert_results, expert_latency_ms = ai_engine.process_experts(
                        input_data,
                        expert_inputs=expert_inputs,
                    )
```

`_build_expert_inputs`는 `"fall"` 키를 만들지 않는다 (`ai/main.py:423-429`):

```python
def _build_expert_inputs(default_data, audio_events: list[dict]) -> tuple[dict, dict | None]:
    latest_audio = audio_events[-1] if audio_events else None
    expert_inputs = {
        "env_sound": _merge_audio_window(audio_events, M3_AUDIO_WINDOW_MS),
        "speech_ko": _merge_audio_window(audio_events, M4_AUDIO_WINDOW_MS) or latest_audio,
    }
    return expert_inputs, latest_audio
```

따라서 `_run_expert("fall", ...)`은 fallback인 `data`를 쓴다 (`ai/main.py:204`):

```python
        expert_input = expert_inputs.get(name, data) if expert_inputs else data
```

`data`는 `TurboQuant.optimize()`를 거친 **1-D float32 배열**이다
(`ai/utils/__init__.py:17-21` — dtype 변환과 빈 배열 방어만 하고 shape은 그대로 반환).

**즉 스냅샷 기준으로는 M1에 100프레임 롤링 윈도우가 들어가지 않는다.**
UDP 패킷 1개가 그대로 1회 추론이 된다. 노드별 deque는 없다.

**따라오는 귀결 (추정, 스냅샷 한정):** 이 경로에 우리 `(1,1,64,100)` ONNX를 얹으면
`_preprocess`가 `(1,1,192,100)`을 만들어 shape 불일치로 ORT가 예외를 던지고,
`ai/main.py:236-239`의 `except Exception`이 이를 삼켜 `fall` 결과가 매번 `{}`가 된다.
즉 **에러 로그 한 줄만 남고 낙상 판정이 조용히 사라진다.**

## A5 · `ai:m1:latest` — 스냅샷은 **단일 키**

`ai/main.py:22-27`

```python
EXPERT_LATEST_KEYS = {
    "fall": "ai:m1:latest",
    "vital": "ai:m2:latest",
    "env_sound": "ai:m3:latest",
    "speech_ko": "ai:m4:latest",
}
```

`ai/main.py:584-596`

```python
def _write_expert_latest(r, expert_name: str, node_id: int, ts_ms: int, output: dict, latency_ms: float):
    key = EXPERT_LATEST_KEYS.get(expert_name)
    if not key:
        return

    payload = {
        "ts_ms": int(ts_ms),
        "node_id": int(node_id),
        "expert": expert_name,
        "latency_ms": round(float(latency_ms), 2),
        "data": output,
    }
    r.set(key, json.dumps(payload, ensure_ascii=False), ex=EXPERT_LATEST_TTL_SECONDS)
```

`node_id`는 **payload 안에만** 있고 키 이름에는 없다.
노드 N개가 돌면 마지막에 쓴 노드가 앞의 것을 전부 덮는다. 작업 지시서 Q5가 우려한 그대로다.

---

# 부록 B · 이 스냅샷이 develop이 아니라는 증거

`CLAUDE.md`가 develop의 사실로 기록해 둔 항목들과 스냅샷을 대조하면 어긋난다.

| `CLAUDE.md` 가 develop이라고 적은 것 | 2026-06-05 스냅샷 실제 |
|---|---|
| Redis `csi:raw`가 `data_raw`/`data_resp`/`data_heart` **3필드** | **`data` 단일 필드** — `sensing/main.py:96-105`의 `xadd`가 `node`/`ts_ms`/`data` 3개만 쓴다. grep `data_raw|data_resp|data_heart` 히트 0건 |
| `M1_MAX_NODES` 로 노드 concat | 심볼 자체가 없음 (A3) |
| 노드별 deque 롤링 버퍼 존재 | 없음 (A4) |
| rp5가 `_infer_onnx`에서 sigmoid 적용 | 없음 (A1) |

`sensing/main.py:94-105` — 단일 `data` 필드 증거:

```python
            processed = preprocess_csi(samples, fs=FS)

            r.xadd(
                STREAM_NAME,
                {
                    "node":  node_id,
                    "ts_ms": ts_ms,
                    "data":  processed.tobytes(),
                },
                maxlen=STREAM_MAXLEN,
                approximate=True,
            )
```

`processed`는 `sensing/filters/__init__.py:42-60`에서
`concatenate([normed, resp, heart, fft_feats])` 로 **하나로 이어 붙인** 결과다.
develop이 이걸 3필드로 쪼갰다면 그건 이 스냅샷 이후의 변경이다.

또 하나: 스냅샷의 전문가 모듈명은 `m2_frenel_vital` / `m3_ast_base` / `m4_whisper_small`
(`ai/main.py:11`)로, 그보다 더 오래된 `D:\rp5`(2026-05-07) 의 `m1_fall` / `m2_vital` /
`m3_activity` / `m4_occupancy` 와도 다르다. 즉 이 배포 레포는 **최소 3세대**를 거쳤고
로컬에 있는 건 그중 두 번째다.

---

# 정찰 재개 시 할 일

```bash
ssh csi@192.168.1.2 'cd ~/safewave && git branch --show-current && git log --oneline -10'
ssh csi@192.168.1.2 'cd ~/safewave && cat ai/experts/m1_wifi_pose.py'
ssh csi@192.168.1.2 'cd ~/safewave && grep -n "M1_MAX_NODES\|deque\|wifi_pose\|_decode_csi\|active\|for node" ai/main.py'
ssh csi@192.168.1.2 'cd ~/safewave && grep -n "M1_MAX_NODES\|EXPERT\|ORT_USE_GPU" .env'
ssh csi@192.168.1.2 'cd ~/safewave && sed -n "1,60p" sensing/main.py'
```

접속 전 확인: 이 PC를 `TESTAP`(192.168.1.0/24)에 붙이거나, rp5 전원·네트워크 상태 확인.
부록 A를 diff 기준선으로 쓰면 무엇이 바뀌었는지 바로 나온다.
