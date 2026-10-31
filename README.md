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

## Architecture

| Module | Responsibility |
|---|---|
| `config.py` | All tunable settings in one place |
| `detector.py` | Model inference and tracking — knows nothing about events |
| `events.py` | Turns detections into domain events — knows nothing about the model |
| `storage.py` | Persistence and queries |
| `llm.py` | Natural language → SQL, with schema context and output parsing |
| `api.py` | HTTP interface and security checks |
| `main.py` | Wires the vision pipeline together |
| `evaluation.py` | Measures text-to-SQL accuracy |

Each module has a single reason to change: swap the model and only `detector.py` moves;
change the database and only `storage.py` moves.

## Measurements

**Vision pipeline** (PyTorch, CPU, 640px input)

| Configuration | FPS |
|---|---|
| Detection only | 9–10 |
| Detection + tracking, crowded scene | 7.1 |

**Text-to-SQL accuracy** (qwen2.5-coder:1.5b)

| Stage | Accuracy |
|---|---|
| Initial | 72% |
| After clarifying ambiguous questions and adding few-shot examples | 88% |

Measured with *execution accuracy* on a 20-question evaluation set: the generated query's
result is compared against a hand-written reference query's result. Figures are the mean
of three runs — LLM output is non-deterministic and a single run varies by about ±5%.

## Security

The model was observed generating `DELETE FROM events` despite an explicit prompt rule
forbidding it. Prompt instructions are a request, not a guarantee, so the application
layer rejects any query that is not a `SELECT`. In production a read-only database user
would add a third layer.

## Known limitations

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

Run the evaluation:
```bash
python evaluation.py
```

## Roadmap

- [ ] ONNX Runtime inference + INT8 quantization, with a before/after benchmark table
- [ ] Docker Compose for one-command startup
- [ ] Android client (Jetpack Compose + Retrofit)
- [ ] RTSP input and multi-stream support
- [ ] Comparison against a larger LLM (qwen2.5-coder:7b)