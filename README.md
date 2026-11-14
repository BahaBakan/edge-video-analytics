# Edge Video Analytics

A real-time video analytics system that runs entirely on-device: it detects and tracks
people in a video stream, turns region crossings into structured events, stores them,
and lets you query that history in natural language — all without sending a single
frame to the cloud.

## What it does

```
video stream → YOLOv8 detection → ByteTrack tracking → ROI event logic
                                                            ↓
natural language question → local LLM → SQL → SQLite ← event storage
```

- **Detection & tracking** — YOLOv8n with ByteTrack, persistent IDs across frames
- **Event logic** — an object entering the configured region produces one event; re-entries
  by the same track ID are not double-counted
- **Storage** — SQLite with parameterised queries
- **REST API** — FastAPI with auto-generated Swagger docs
- **Natural language queries** — a locally hosted LLM (Ollama) converts questions into
  SQL; the query is validated and executed by the application, never by the model
- **Quantised inference** — an INT8 model that is smaller *and* faster than the original,
  with the accuracy cost measured rather than assumed

## Architecture

| Module | Responsibility |
|---|---|
| `config.py` | All tunable settings in one place |
| `detector.py` | Model inference and tracking — knows nothing about events |
| `events.py` | Turns detections into domain events — knows nothing about the model |
| `storage.py` | Persistence and queries |
| `llm.py` | Natural language → SQL, with schema context and output parsing |
| `api.py` | HTTP interface and security checks |
| `main.py` | Wires the vision pipeline together, and times inference |
| `evaluation.py` | Measures text-to-SQL accuracy |
| `export_onnx.py` | PyTorch → ONNX export |
| `quantize_int8.py` | Dynamic INT8 quantisation (ONNX Runtime) |
| `quantize_static.py` | Static INT8 quantisation with calibration (ONNX Runtime) |
| `export_openvino.py` | INT8 export through Ultralytics + NNCF (OpenVINO) |
| `compare_models.py` | Measures what quantisation costs in accuracy |

Each module has a single reason to change: swap the model and only `detector.py` moves;
change the database and only `storage.py` moves.

---

## Model optimisation

The project is called *edge* analytics, so the model has to be small and fast enough to
run somewhere that is neither. Four routes to INT8 were tried. Two of them failed, and
the failures are reported here because they are the part that took the work.

### Results

Intel Core i5-12450H, CPU only, 640px input, 1919 frames of a crowded street scene.

| Model | Size | ms / frame | FPS | Detections vs FP32 | Match rate | Mean IoU |
|---|---|---|---|---|---|---|
| PyTorch FP32 (`.pt`) | 6.2 MB | 74.6 | 13.4 | reference | — | — |
| ONNX FP32 | 12.3 MB | 124.3 | 8.0 | unchanged | — | — |
| ONNX dynamic INT8 | 3.3 MB | slower¹ | — | −9.8% | 88.7% | 0.956 |
| ONNX static INT8 | 3.4 MB | — | — | **no detections at all** | — | — |
| **OpenVINO INT8** | **3.6 MB** | **67.7** | **14.8** | **−1.1%** | **96.4%** | **0.972** |

¹ Timed before the measurement protocol below was adopted, so the figure is not
comparable and is omitted rather than reported.

**The winner is 42% smaller and 9% faster than the original, and loses 1.1% of
detections.** Accuracy is measured against the unquantised model's own output: for every
box the FP32 model finds, the INT8 model is checked for a box in the same place
(IoU ≥ 0.5). This answers "what did quantisation change", not "what is the ground truth" —
there is no labelled data for this video, and the README should not pretend otherwise.

### Why the first attempts failed

**ONNX FP32 was slower than PyTorch.** Converting a model does not make it faster by
itself. PyTorch's CPU path is already well optimised, and the Ultralytics ONNX path adds
a NumPy↔Torch conversion on every frame for post-processing. The conversion was kept
anyway, because quantisation needs a computation graph to operate on.

**Dynamic INT8 was dramatically slower.** Dynamic quantisation computes activation
scales at runtime and wraps every convolution in quantise/dequantise nodes. Without an
optimised INT8 convolution kernel underneath, the data is converted to INT8 and straight
back to FP32 — the arithmetic never gets faster, and the conversion cost is paid on every
layer. YOLOv8 is almost entirely convolutions, so the penalty compounds.

**Static INT8 through ONNX Runtime produced a model that detected nothing.** The
generic quantiser treats every layer alike, including the detection head that regresses
box coordinates. That head is the most sensitive part of the network; quantising it
destroys the output. Percentile calibration instead of min/max did not help, because the
calibration method was never the problem.

**OpenVINO worked because the tooling knows the model.** Ultralytics exports through
NNCF, which recognises YOLOv8's architecture and leaves the detection head in floating
point — the export log lists each skipped operation under `model.22`, the detect head.
It also applies bias correction, compensating for the error quantisation accumulates
layer by layer. The runtime matters too: OpenVINO uses the INT8 instructions this CPU
actually has, which ONNX Runtime's CPU provider did not.

The lesson is not "OpenVINO is better". It is that a generic tool applied to a model it
does not understand will quantise the one layer it should have left alone.

### Why the speedup is only 9%

Inference time is not all model. Letterboxing, NMS and the tracker run in floating point
and are untouched by quantisation. The model itself is considerably faster; the pipeline
around it sets the ceiling — Amdahl's law, in a benchmark table.

### Measurement conditions

The first round of measurements was taken on a laptop that was sometimes on battery,
with a browser open. Those numbers said ONNX was 9% faster than PyTorch. Repeating the
baseline under controlled conditions showed the same `.pt` model running **3.2× faster**
than it had measured an hour earlier, and reversed the conclusion entirely.

Every figure in the table above was therefore taken with:

- the machine on mains power, Windows set to best-performance
- no browser or other application running
- all runs back-to-back in one session
- the first 5 frames discarded as model warm-up
- only the `track()` call timed — decoding, drawing and display excluded

A benchmark without its conditions is not a measurement. The discarded first round is
the reason this section exists.

---

## Text-to-SQL accuracy

qwen2.5-coder:1.5b, 20-question evaluation set.

| Stage | Accuracy |
|---|---|
| Initial | 72% |
| After clarifying ambiguous questions and adding few-shot examples | 88% |

Measured with *execution accuracy*: the generated query's result is compared against a
hand-written reference query's result. Figures are the mean of three runs — LLM output is
non-deterministic and a single run varies by about ±5%.

## Security

The model was observed generating `DELETE FROM events` despite an explicit prompt rule
forbidding it. Prompt instructions are a request, not a guarantee, so the application
layer rejects any query that is not a `SELECT`. In production a read-only database user
would add a third layer.

## Known limitations

- Detection accuracy after quantisation is measured against the FP32 model, not against
  labelled ground truth — it reports the cost of quantisation, not absolute accuracy
- The 14.8 FPS figure is for a laptop CPU; no measurement has been taken on actual edge
  hardware, so no claim is made about one
- Questions requiring aggregation arithmetic ("average events per hour") exceed what a
  1.5B parameter model handles reliably
- ByteTrack IDs restart each run; there is no `session_id`, so IDs are not unique across sessions
- SQLite allows a single writer; a multi-process deployment would need PostgreSQL
- Video source is a file; RTSP support is not implemented yet

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

ollama pull qwen2.5-coder:1.5b
```

Run the vision pipeline:
```bash
python main.py
```

Run the API:
```bash
uvicorn api:app --reload
# http://127.0.0.1:8000/docs
```

Run the text-to-SQL evaluation:
```bash
python evaluation.py
```

Reproduce the quantisation results:
```bash
python export_openvino.py     # build the INT8 model
python main.py                # time it (set MODEL_PATH in config.py)
python compare_models.py      # measure what it costs in accuracy
```

## Roadmap

- [x] INT8 quantisation with a measured speed/accuracy trade-off
- [ ] Train on a custom dataset instead of using COCO weights
- [ ] Run the benchmark on real edge hardware (Android / Raspberry Pi)
- [ ] Android client with on-device inference (Jetpack Compose)
- [ ] Richer event logic: line crossing, dwell time, direction
- [ ] RTSP input and multi-stream support
- [ ] Comparison against a larger LLM (qwen2.5-coder:7b)
