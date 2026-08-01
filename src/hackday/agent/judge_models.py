"""Central model ids for every judge / monitor call.

Routed through OpenRouter rather than the Anthropic API directly: the cluster
jobs only have OPENROUTER_API_KEY in their environment (see k8s/README.md).
Same underlying Claude models, so scores stay comparable with earlier
Anthropic-keyed runs. To go back to the direct API, change these two strings
to "anthropic/claude-haiku-4-5-20251001" / "anthropic/claude-sonnet-4-5-20250929".
"""

JUDGE_HAIKU = "openrouter/anthropic/claude-haiku-4.5"
JUDGE_SONNET = "openrouter/anthropic/claude-sonnet-4.5"
