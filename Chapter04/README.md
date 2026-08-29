# Chapter 4 - Monitoring AI Systems

This directory contains the educational synthetic monitoring example used in Chapter 4. It demonstrates how an application can instrument locally generated detector results with OpenTelemetry, expose them to Prometheus, and visualize them in Grafana.

> **Important:** All traffic, responses, model choices, detector decisions, scores, timings, false-positive and false-negative events, and dashboard values are synthetic or illustrative. Model names are labels for time-series data only. The named models and providers are not queried, and the displayed values must not be interpreted as real measurements, official rates, or model benchmarks.

No external AI API key is required for the default example.

## Architecture

```text
demo.py
  | sends synthetic HTTP requests
  v
local FastAPI application (main.py)
  | creates local sample responses and simulated detector outcomes
  v
OpenTelemetry Metrics SDK
  | instruments application-supplied measurements at /metrics
  v
Prometheus
  | scrapes and stores the metric time series
  v
Grafana
  | queries Prometheus and displays the provisioned dashboard
```

The simulated detector in `main.py` is the source of every detector flag and score. OpenTelemetry records measurements, Prometheus scrapes and stores them, and Grafana queries and visualizes them. OpenTelemetry, Prometheus, and Grafana do not determine whether text is factually hallucinated.

## Included files

- `main.py` - local FastAPI application, synthetic response generator, simulated application-side detector, and OpenTelemetry metric instruments
- `demo.py` - synthetic traffic generator used by the walkthrough
- `Dockerfile` - Python 3.11 image for the FastAPI application
- `docker-compose.yml` - local application, Prometheus, and Grafana services
- `prometheus.yml` - scrape configuration for the application's `/metrics` endpoint
- `grafana/provisioning/` - automatic Prometheus datasource and dashboard provisioning
- `grafana/dashboards/llm-monitoring.json` - preconfigured synthetic monitoring dashboard
- `requirements.txt` - pinned Python dependencies for the application and demo client

## Requirements

- Docker Desktop with Docker Compose
- Python 3.11 or newer for `demo.py`
- `pip`

The default project does not read an API key or contact an external LLM service.

## Run the walkthrough

Run all commands from the `Chapter04` directory.

1. Start Docker Desktop and wait until the Docker engine is ready.

2. Build and start the local stack:

   ```bash
   docker compose up -d
   ```

3. Install the Python dependencies used by the traffic generator:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Generate synthetic traffic:

   ```bash
   python demo.py
   ```

5. Stop and remove the containers when finished:

   ```bash
   docker compose down
   ```

The named Docker volumes retain Prometheus and Grafana data between ordinary `down` and `up` commands. Use `docker compose down --volumes` only when you intentionally want to remove that local demonstration data.

## Local services

| Service | URL | Purpose |
| --- | --- | --- |
| LLM app | <http://localhost:8000> | Local synthetic response and detector application |
| API documentation | <http://localhost:8000/docs> | FastAPI-generated endpoint documentation |
| Application metrics | <http://localhost:8000/metrics> | OpenTelemetry metrics in Prometheus format |
| Prometheus | <http://localhost:9090> | Scrapes and stores application metrics |
| Grafana | <http://localhost:3000> | Visualizes Prometheus data; sign in with `admin` / `admin` |

Grafana provisions the Prometheus datasource and the **Synthetic LLM Monitoring Dashboard** automatically from the checked-in files.

## What the demo records

- Synthetic request counts grouped by illustrative model label
- Simulated application response duration
- Simulated detector scores and detector runtime
- Simulated detector flags
- Manually generated false-positive and false-negative events
- Simulated active sessions

These values show the flow of monitoring data; they do not establish factual accuracy, provider quality, cost, latency, or safety characteristics for any real model.

## Manual verification

Check the application health endpoint:

```bash
curl http://localhost:8000/health
```

Send one local synthetic request:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Explain monitoring","model":"gpt-4o"}'
```

The `model` value above remains a label only. Inspect the exported metrics at <http://localhost:8000/metrics> or query them through Prometheus.

## Troubleshooting

Inspect container state and logs:

```bash
docker compose ps
docker compose logs llm-app
docker compose logs prometheus
docker compose logs grafana
```

If ports `8000`, `9090`, or `3000` are already occupied, stop the conflicting local service before starting this stack.

## Book section

This project supports the Chapter 4 walkthrough of application instrumentation, metric storage, dashboard provisioning, and interpretation of monitoring signals. It is intentionally small and synthetic; it is not a production detector, evaluation suite, or provider-comparison benchmark.

The Packt copy is based on `morfidon/monitoring-llm` snapshot `e27df76a43844d9e5e36490c91b7f9df6b972cb1`, with wording and implementation corrections that make the synthetic nature explicit.
