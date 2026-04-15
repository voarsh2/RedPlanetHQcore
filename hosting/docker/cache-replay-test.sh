#!/bin/bash
# Replay a saved wire body multiple times to test cache stability
# Usage: ./cache-replay-test.sh <body-file.json> <iterations>

BODY_FILE="${1:-/tmp/chat-replay-body.json}"
ITERATIONS="${2:-5}"

if [ ! -f "$BODY_FILE" ]; then
  echo "Body file not found: $BODY_FILE"
  exit 1
fi

BASE_URL="https://api.theclawbay.com/v1"
API_KEY="ca_v1.aHR0cDovL2dhdGV3YXktZXUtaGV0em5lci50aGVjbGF3YmF5LmNvbQ.zTvHR0B8KRDffZcUfIOnt_IHlMd6XsZXaY1Fjvxd6fg"
SESSION_ID="cache-replay-test-$(date +%s)"

echo "=== Cache Replay Test ==="
echo "Body file: $BODY_FILE ($(wc -c < "$BODY_FILE") bytes)"
echo "Iterations: $ITERATIONS"
echo "Session ID: $SESSION_ID"
echo ""

for i in $(seq 1 $ITERATIONS); do
  echo "--- Iteration $i/$ITERATIONS ---"

  TMPFILE=$(mktemp)

  curl -s \
    -X POST "$BASE_URL/responses" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $API_KEY" \
    -H "session_id: $SESSION_ID" \
    -H "x-client-request-id: $SESSION_ID" \
    -d @"$BODY_FILE" \
    --max-time 120 \
    -o "$TMPFILE"

  # Extract usage from SSE stream - response.completed event has usage object
  # Format: data: {"type":"response.completed","response":{"usage":{"input_tokens":...,"cached_input_tokens":...}}}
  USAGE_LINE=$(grep -o '"usage":{[^}]*}' "$TMPFILE" | tail -1)

  if [ -n "$USAGE_LINE" ]; then
    TOTAL=$(echo "$USAGE_LINE" | jq -r '.total_tokens // "N/A"')
    PROMPT=$(echo "$USAGE_LINE" | jq -r '.input_tokens // .prompt_tokens // "N/A"')
    COMPLETION=$(echo "$USAGE_LINE" | jq -r '.output_tokens // .completion_tokens // "N/A"')
    CACHED=$(echo "$USAGE_LINE" | jq -r '.cached_input_tokens // .cached_tokens // "N/A"')

    echo "  Tokens: prompt=$PROMPT, completion=$COMPLETION, cached=$CACHED, total=$TOTAL"

    if [ "$CACHED" = "0" ] || [ "$CACHED" = "N/A" ] || [ "$CACHED" = "null" ]; then
      echo "  Status: CACHE MISS"
    else
      echo "  Status: CACHE HIT (${CACHED}/${PROMPT})"
    fi
  else
    echo "  Could not parse usage from response"
    echo "  Raw tail: $(tail -c 500 "$TMPFILE")"
  fi

  rm -f "$TMPFILE"
  echo ""
done
