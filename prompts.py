"""Prompt sets for the DiffusionGemma vs AR benchmark.

Three sanity prompts (coherence gate) + a matrix spanning the adaptive-compute
claim: short structured (code), medium chat, long-form. The architecture claims
*fewer* denoising steps for simpler prompts; the matrix lets us check whether
denoising-steps actually track prompt complexity on Apple Silicon.
"""

# --- PHASE 1: coherence gate. Plain, verifiable, hard to fake. ---
SANITY = [
    {
        "id": "sanity_factual",
        "prompt": "What is the capital of Japan? Answer in one sentence.",
        "max_tokens": 64,
    },
    {
        "id": "sanity_reasoning",
        "prompt": "If a train travels 60 km in 45 minutes, what is its average speed in km/h? Show your reasoning.",
        "max_tokens": 256,
    },
    {
        "id": "sanity_code",
        "prompt": "Write a Python function `is_palindrome(s)` that returns True if the string is a palindrome, ignoring case and spaces.",
        "max_tokens": 256,
    },
]

# --- PHASE 3: benchmark matrix. Spans the adaptive-compute claim. ---
MATRIX = [
    # short structured (code) — claim: should need few denoising steps
    {
        "id": "short_code",
        "category": "short_structured",
        "prompt": "Write a Python one-liner that returns the sum of all even numbers in a list `xs`.",
        "max_tokens": 128,
    },
    {
        "id": "short_json",
        "category": "short_structured",
        "prompt": 'Return a JSON object with keys "name", "age", "city" for a fictional 30-year-old named Aria living in Lyon. Output only the JSON.',
        "max_tokens": 128,
    },
    # medium chat
    {
        "id": "medium_chat",
        "category": "medium_chat",
        "prompt": "Explain the difference between a process and a thread to a junior developer, with a concrete analogy.",
        "max_tokens": 512,
    },
    {
        "id": "medium_advice",
        "category": "medium_chat",
        "prompt": "I'm choosing between renting and buying an apartment. Give me a balanced list of the main trade-offs.",
        "max_tokens": 512,
    },
    # long-form — claim: more denoising steps / larger canvas
    {
        "id": "long_essay",
        "category": "long_form",
        "prompt": "Write a detailed, well-structured essay (multiple paragraphs) on how unified-memory architectures like Apple Silicon change the trade-offs for running large language models locally.",
        "max_tokens": 1024,
    },
    {
        "id": "long_story",
        "category": "long_form",
        "prompt": "Write a complete short story (several paragraphs) about a lighthouse keeper who discovers the light is talking back.",
        "max_tokens": 1024,
    },
]
