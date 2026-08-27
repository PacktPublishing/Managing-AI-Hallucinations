# Chapter 3 - Detecting and Verifying AI Output

## Contents

### Safety guardrails example

`safety-guardrails.js` is the complete runnable version of the smaller safety, logging, validation, and fallback snippets presented in Chapter 3. It includes the named functions from the chapter, uses the OpenAI Responses API, and keeps the human-review queue and security log in memory so the educational helpers do not contact external services.

The default command uses a mock client. It deliberately simulates a primary-model failure, exercises the fallback model path, demonstrates input review and a safe default, and makes no paid API request. The explicit live command makes one model request in the normal case and at most two if the primary attempt fails and the fallback is tried.

The repository version uses the intended model identifiers `gpt-5.6-terra` and `gpt-5.6-luna`; either can be overridden with the optional variables in `.env.example`. It uses the current SDK's `client.responses.create()` interface and reads `response.output_text`. The manuscript's `temperature` option is omitted because it is not required for the current Responses API example and model-specific support was not established.

The system instruction is also tightened for hallucination safety: it asks the model to cite only sources supplied in the context, never invent a citation, and say when a claim cannot be sourced.

### Knowledge-based agent supporting files

- `data/DataMind_FAQ_EN.pdf`
- `data/additional_info.txt`

These unchanged files support the chapter's file-search and knowledge-grounding example. They are included here so readers do not need to retrieve them from another repository.

### Codex code review example

The Codex code-review walkthrough uses the reader's own GitHub repository and therefore does not require a sample repository in this directory. The author's private demonstration repository is intentionally not included.

Other conceptual sections in the chapter do not need standalone files merely to repeat prose or short snippets.

## Requirements

- Node.js 22 or newer
- npm
- An OpenAI API key for the explicit live API demonstration
- GitHub and Codex access for the separate code-review walkthrough

## Setup

Run these commands from the `Chapter03` directory:

```powershell
npm install
Copy-Item .env.example .env
```

On macOS or Linux, use `cp .env.example .env`. Replace the placeholder value in `.env` with your OpenAI API key. Do not commit `.env`.

Run the local mocked demonstration and syntax check:

```powershell
npm test
```

Run the example without the extra syntax-check step:

```powershell
npm start
```

To opt in to the small live API demonstration:

```powershell
npm run start:live
```

The live command uses Node.js's `--env-file` option to load `OPENAI_API_KEY` from `.env`. Model availability depends on the models enabled for the reader's API account; use the optional `OPENAI_PRIMARY_MODEL` and `OPENAI_FALLBACK_MODEL` variables if different model IDs are required.

## Important note

The regular-expression input patterns are illustrative and will not detect every kind of sensitive information. `validateOutput()` is a simple demonstration, not a factual verification engine. Passing either validation function does not prove that content is secure, private, factually correct, properly sourced, or safe. A production system needs threat modeling, policy-specific controls, structured monitoring, and appropriate human review.

## Book sections

This directory supports Chapter 3, **Detecting and Verifying AI Output**, particularly the sections on safety guardrails, logging, fallbacks, moderation concepts, and file-backed knowledge grounding. Multi-step routing is illustrated by `queryWithFallbacks()` without turning the example into a production framework.
