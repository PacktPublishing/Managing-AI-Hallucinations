"""Synthetic LLM monitoring application for the Chapter 4 walkthrough.

The application generates responses and detector outcomes locally. Model names
are labels for synthetic traffic only; no model provider is contacted.
"""

import logging
import random
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

import uvicorn
from fastapi import FastAPI, Response
from opentelemetry import metrics
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SIMULATION_NOTICE = (
    "Educational synthetic data only. No external model was queried, and the "
    "values are not model benchmarks."
)

# OpenTelemetry records application measurements. The Prometheus reader makes
# those measurements available to prometheus_client for the /metrics endpoint.
resource = Resource.create({SERVICE_NAME: "chapter04-synthetic-monitoring"})
prometheus_reader = PrometheusMetricReader()
meter_provider = MeterProvider(
    resource=resource,
    metric_readers=[prometheus_reader],
)
metrics.set_meter_provider(meter_provider)
meter = metrics.get_meter("chapter04.synthetic-monitoring", "1.0.0")

# Counter names omit the Prometheus `_total` suffix. The exporter adds it.
hallucination_counter = meter.create_counter(
    "hallucinations_detected",
    description="Simulated application-side detector flags",
)
llm_requests_total = meter.create_counter(
    "llm_requests",
    description="Synthetic requests handled by the local demonstration",
)
model_usage = meter.create_counter(
    "model_usage",
    description="Synthetic traffic grouped by illustrative model label",
)
false_positive_counter = meter.create_counter(
    "false_positives",
    description="Manually simulated false-positive events",
)
false_negative_counter = meter.create_counter(
    "false_negatives",
    description="Manually simulated false-negative events",
)
active_sessions_gauge = meter.create_up_down_counter(
    "active_sessions",
    description="Simulated active sessions",
)
llm_response_time = meter.create_histogram(
    "llm_response_duration_seconds",
    unit="s",
    description="Elapsed time for a locally simulated response",
)
hallucination_score = meter.create_histogram(
    "hallucination_score",
    description="Synthetic score produced by the application-side detector",
)
detection_latency = meter.create_histogram(
    "detection_latency_seconds",
    unit="s",
    description="Elapsed time for the simulated detector function",
)

app = FastAPI(
    title="Chapter 4 Synthetic LLM Monitoring Demo",
    version="1.0.0",
    description=SIMULATION_NOTICE,
)


class LLMRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4_000)
    model: str = Field(
        default="gpt-4o",
        description="Illustrative series label only; no provider is contacted",
    )
    user_id: Optional[str] = None
    context: Optional[str] = None


class LLMResponse(BaseModel):
    response: str
    model: str
    response_time: float
    hallucination_detected: bool
    hallucination_score: float
    timestamp: str
    simulation_notice: str


class HallucinationDetector:
    """Return deliberately simulated outcomes for monitoring practice."""

    detection_methods = (
        "simulated_semantic_similarity",
        "simulated_factual_consistency",
        "simulated_confidence_score",
    )
    trigger_phrases = (
        "hallucinate",
        "make up",
        "fictional",
        "false information",
    )

    def detect_hallucination(
        self,
        prompt: str,
        response: str,
        model: str,
    ) -> tuple[bool, float, str, float]:
        """Generate a synthetic flag and score independent of model label."""
        del response, model  # Labels and local text do not affect this toy detector.
        started_at = time.perf_counter()
        method = random.choice(self.detection_methods)
        is_trigger_prompt = any(
            phrase in prompt.lower() for phrase in self.trigger_phrases
        )
        score = (
            random.uniform(0.65, 0.95)
            if is_trigger_prompt
            else random.uniform(0.05, 0.45)
        )
        detector_runtime = time.perf_counter() - started_at
        return score > 0.5, score, method, detector_runtime


detector = HallucinationDetector()


def simulate_llm_response(prompt: str, model: str) -> str:
    """Create local sample text; this function never calls an AI API."""
    if any(phrase in prompt.lower() for phrase in detector.trigger_phrases):
        samples = (
            "[Synthetic example] The Eiffel Tower was built on the Moon.",
            "[Synthetic example] Python was invented as a spreadsheet formula.",
            "[Synthetic example] Ocean tides are controlled by traffic lights.",
        )
    else:
        samples = (
            "[Synthetic example] This local response represents a routine answer.",
            "[Synthetic example] The application generated this text without an AI API.",
            "[Synthetic example] This response exists only to create monitoring traffic.",
        )

    return f"{random.choice(samples)} Model label: {model}."


@app.get("/")
async def root():
    return {
        "message": "Chapter 4 synthetic LLM monitoring application",
        "status": "running",
        "simulation_notice": SIMULATION_NOTICE,
    }


@app.get("/metrics")
async def metrics_endpoint():
    """Expose OpenTelemetry measurements in Prometheus text format."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/chat", response_model=LLMResponse)
async def chat_completion(request: LLMRequest):
    """Generate a local response and an application-side simulated score."""
    started_at = time.perf_counter()
    session_id = request.user_id or str(uuid.uuid4())
    attributes = {"model": request.model, "traffic": "synthetic"}

    llm_requests_total.add(1, attributes)
    model_usage.add(1, attributes)

    # This short delay represents local application work, not provider latency.
    time.sleep(random.uniform(0.05, 0.20))
    response_text = simulate_llm_response(request.prompt, request.model)
    detected, score, method, detector_runtime = detector.detect_hallucination(
        request.prompt,
        response_text,
        request.model,
    )
    total_time = time.perf_counter() - started_at

    llm_response_time.record(total_time, attributes)
    hallucination_score.record(score, {**attributes, "method": method})
    detection_latency.record(
        detector_runtime,
        {"method": method, "traffic": "synthetic"},
    )

    if detected:
        hallucination_counter.add(
            1,
            {
                **attributes,
                "severity": "high" if score > 0.8 else "medium",
                "method": method,
            },
        )

    logger.info(
        "Synthetic request processed - label=%s session=%s flag=%s score=%.2f",
        request.model,
        session_id,
        detected,
        score,
    )
    return LLMResponse(
        response=response_text,
        model=request.model,
        response_time=total_time,
        hallucination_detected=detected,
        hallucination_score=score,
        timestamp=datetime.now(timezone.utc).isoformat(),
        simulation_notice=SIMULATION_NOTICE,
    )


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/stats")
async def get_stats():
    return {
        "metrics": "Available from /metrics and Prometheus",
        "detection_methods": detector.detection_methods,
        "simulation_notice": SIMULATION_NOTICE,
    }


@app.post("/simulate/false-positive")
async def simulate_false_positive():
    false_positive_counter.add(1, {"traffic": "synthetic"})
    return {"message": "Synthetic false-positive event recorded"}


@app.post("/simulate/false-negative")
async def simulate_false_negative():
    false_negative_counter.add(1, {"traffic": "synthetic"})
    return {"message": "Synthetic false-negative event recorded"}


@app.post("/simulate/session/start")
async def start_session():
    active_sessions_gauge.add(1, {"traffic": "synthetic"})
    return {"message": "Synthetic session started"}


@app.post("/simulate/session/end")
async def end_session():
    active_sessions_gauge.add(-1, {"traffic": "synthetic"})
    return {"message": "Synthetic session ended"}


if __name__ == "__main__":
    logger.info("Starting synthetic monitoring app on http://localhost:8000")
    logger.info("Metrics available at http://localhost:8000/metrics")
    uvicorn.run(app, host="0.0.0.0", port=8000)
