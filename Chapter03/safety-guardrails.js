import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import OpenAI from "openai";

const PRIMARY_MODEL = process.env.OPENAI_PRIMARY_MODEL || "gpt-5.6-terra";
const FALLBACK_MODEL = process.env.OPENAI_FALLBACK_MODEL || "gpt-5.6-luna";
const SAFE_DEFAULT =
  "We're reviewing this with a team member. Please try again shortly.";

const SYSTEM_PROMPT = `You are a careful assistant that separates sourced facts from uncertainty.
Cite only sources explicitly provided in the context.
Never invent a citation or source.
If no source is available, say that the claim cannot be sourced.
Do not present medical or legal information as professional advice.`;

const securityEvents = [];
const reviewQueue = [];

// These regexes are illustrative teaching aids. They cannot identify every kind
// of sensitive data and must not be treated as a complete security control.
const illustrativeSensitivePatterns = [
  { label: "email address", pattern: /\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/i },
  { label: "possible API key", pattern: /\bsk-[A-Za-z0-9_-]{20,}\b/ },
  { label: "possible payment-card number", pattern: /\b(?:\d[ -]*?){13,19}\b/ },
];

function logSecurityEvent(type, details = {}) {
  const event = {
    timestamp: new Date().toISOString(),
    type,
    ...details,
  };

  securityEvents.push(event);
  return event;
}

function addToReviewQueue(item) {
  const reviewItem = {
    id: reviewQueue.length + 1,
    createdAt: new Date().toISOString(),
    ...item,
  };

  reviewQueue.push(reviewItem);
  return reviewItem;
}

function sendToHumanReview(item) {
  // Educational stand-in only: no message or external service is contacted.
  return addToReviewQueue(item);
}

function handleFallback(reason = "The request could not be completed safely.") {
  return {
    status: "fallback",
    answer: SAFE_DEFAULT,
    reason,
  };
}

function validateInput(input) {
  const reasons = [];

  if (typeof input !== "string" || input.trim().length === 0) {
    reasons.push("Input must be a non-empty string");
  } else {
    for (const { label, pattern } of illustrativeSensitivePatterns) {
      if (pattern.test(input)) {
        reasons.push(`Illustrative pattern matched: ${label}`);
      }
    }

    if (input.length > 4_000) {
      reasons.push("Input exceeds the educational example's length limit");
    }
  }

  return { isValid: reasons.length === 0, reasons };
}

function safeQuery(input) {
  const validation = validateInput(input);

  if (!validation.isValid) {
    logSecurityEvent("input_rejected", { reasons: validation.reasons });
    sendToHumanReview({ stage: "input", reasons: validation.reasons });
    return { ok: false, reason: validation.reasons.join("; ") };
  }

  return { ok: true, value: input.trim() };
}

function validateOutput(output) {
  const reasons = [];

  if (typeof output !== "string" || output.trim().length === 0) {
    return { isValid: false, reasons: ["Output must be a non-empty string"] };
  }

  const hasCitation = /\[[^\]]{2,}\]|https?:\/\/|\bsource:/i.test(output);
  const expressesUncertainty =
    /\b(may|might|uncertain|cannot verify|cannot be sourced|based on|according to)\b/i.test(
      output,
    );
  const looksLikeProfessionalAdvice =
    /\b(you should|you must|i recommend)\b.{0,80}\b(dosage|diagnos|medication|lawsuit|legal action|plead)\b/i.test(
      output,
    );

  if (looksLikeProfessionalAdvice) {
    reasons.push("Medical/legal advice detected");
  }

  if (output.length > 300 && !hasCitation) {
    reasons.push("Long claim without sources");
  }

  if (output.length > 300 && !hasCitation && !expressesUncertainty) {
    reasons.push("No uncertainty or citations");
  }

  // This is a deliberately simple demonstration. A passing result does not
  // prove that the response is factual, safe, private, or properly sourced.
  return { isValid: reasons.length === 0, reasons };
}

function createOpenAIClient() {
  if (!process.env.OPENAI_API_KEY) {
    throw new Error("OPENAI_API_KEY is required for the live API example");
  }

  return new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
}

async function requestModel(input, { context = "", model, client } = {}) {
  const activeClient = client || createOpenAIClient();
  const sourceContext = context.trim()
    ? `SOURCE CONTEXT:\n${context.trim()}`
    : "SOURCE CONTEXT:\nNo source was supplied.";

  const response = await activeClient.responses.create({
    model,
    instructions: SYSTEM_PROMPT,
    input: `${sourceContext}\n\nUSER QUESTION:\n${input}`,
  });

  if (!response.output_text) {
    throw new Error("The API response did not contain output_text");
  }

  return response.output_text;
}

async function callAIWithGuardrails(
  input,
  { context = "", model = PRIMARY_MODEL, client } = {},
) {
  const checkedInput = safeQuery(input);
  if (!checkedInput.ok) {
    return handleFallback(checkedInput.reason);
  }

  const answer = await requestModel(checkedInput.value, {
    context,
    model,
    client,
  });
  const checkedOutput = validateOutput(answer);

  if (!checkedOutput.isValid) {
    logSecurityEvent("output_rejected", {
      model,
      reasons: checkedOutput.reasons,
    });
    sendToHumanReview({
      stage: "output",
      model,
      reasons: checkedOutput.reasons,
    });
    return handleFallback(checkedOutput.reasons.join("; "));
  }

  return { status: "ok", answer, model };
}

async function callAIWithLogging(input, options = {}) {
  const model = options.model || PRIMARY_MODEL;
  logSecurityEvent("model_call_started", { model });

  try {
    const result = await callAIWithGuardrails(input, { ...options, model });
    logSecurityEvent("model_call_completed", { model, status: result.status });
    return result;
  } catch (error) {
    logSecurityEvent("model_call_failed", { model, error: error.message });
    throw error;
  }
}

function analyzeFailures(events = securityEvents) {
  return events.reduce((summary, event) => {
    if (
      event.type.includes("failed") ||
      event.type.includes("rejected") ||
      event.status === "fallback"
    ) {
      summary.total += 1;
      summary.byType[event.type] = (summary.byType[event.type] || 0) + 1;
    }
    return summary;
  }, { total: 0, byType: {} });
}

function tryMainModel(input, options = {}) {
  return callAIWithLogging(input, { ...options, model: PRIMARY_MODEL });
}

function tryFallbackModel(input, options = {}) {
  return callAIWithLogging(input, { ...options, model: FALLBACK_MODEL });
}

function trySafeDefault(reason) {
  logSecurityEvent("safe_default_used", { reason });
  return handleFallback(reason);
}

async function queryWithFallbacks(input, options = {}) {
  try {
    return await tryMainModel(input, options);
  } catch (mainError) {
    try {
      return await tryFallbackModel(input, options);
    } catch (fallbackError) {
      return trySafeDefault(
        `Both model attempts failed: ${mainError.message}; ${fallbackError.message}`,
      );
    }
  }
}

async function runLocalDemo() {
  let calls = 0;
  const mockClient = {
    responses: {
      create: async ({ model }) => {
        calls += 1;
        if (model === PRIMARY_MODEL) {
          throw new Error("Simulated primary model outage");
        }
        return {
          output_text:
            "According to [DataMind Support Information], Pro Plan email support responds within 24 hours.",
        };
      },
    },
  };

  const result = await queryWithFallbacks("How quickly does Pro Plan support respond?", {
    context:
      "[DataMind Support Information] Pro Plan: Email support within 24 hours.",
    client: mockClient,
  });
  safeQuery("Please contact reader@example.com with the result.");
  const safeDefault = trySafeDefault("Demonstrating the final local fallback");

  console.log("Local mocked guardrails demo (no API calls)");
  console.log({ result, safeDefault, calls, reviewItems: reviewQueue.length });
  console.log("Failure summary:", analyzeFailures());
}

async function runLiveDemo() {
  const context = await readFile(
    new URL("./data/additional_info.txt", import.meta.url),
    "utf8",
  );
  const result = await queryWithFallbacks("How quickly does Pro Plan support respond?", {
    context: `[DataMind Support Information]\n${context}`,
  });

  console.log(result);
}

const isMainModule =
  process.argv[1] &&
  import.meta.url === pathToFileURL(resolve(process.argv[1])).href;

if (isMainModule) {
  const run = process.argv.includes("--live") ? runLiveDemo : runLocalDemo;
  run().catch((error) => {
    console.error(error.message);
    process.exitCode = 1;
  });
}

export {
  SYSTEM_PROMPT,
  addToReviewQueue,
  analyzeFailures,
  callAIWithGuardrails,
  callAIWithLogging,
  handleFallback,
  logSecurityEvent,
  queryWithFallbacks,
  safeQuery,
  sendToHumanReview,
  tryFallbackModel,
  tryMainModel,
  trySafeDefault,
  validateInput,
  validateOutput,
};
