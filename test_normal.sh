#!/bin/bash
echo "=== Testing Normal Traffic (No Drift) ==="
echo "Generating 100 normal predictions..."

for i in {1..100}; do
  curl -s -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"features": [0.5, -0.3, 0.8, 0.2, -0.1]}' | jq -r '.prediction'
  
  if [ $((i % 10)) -eq 0 ]; then
    echo "Completed $i predictions"
  fi
done

echo "=== Checking Drift Status ==="
curl -s http://localhost:8000/drift/status | jq '.'
