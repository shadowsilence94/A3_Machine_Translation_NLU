#!/bin/bash

# Start App Background - Wait for it to start
python app/app.py > app_log.txt 2>&1 &
PID=$!
echo "Application started with PID: $PID"
echo "Waiting for 20 seconds for model loading..."
sleep 20

echo "--- Testing NLLB English -> Kazakh ---"
curl -X POST http://localhost:5001/api/translate \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world", "model":"nllb", "lang":"kk", "direction":"e2f"}'
echo ""
echo "--- Testing NLLB English -> Bengali ---"
curl -X POST http://localhost:5001/api/translate \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world", "model":"nllb", "lang":"bn", "direction":"e2f"}'
echo ""
echo "--- Testing NLLB English -> Burmese ---"
curl -X POST http://localhost:5001/api/translate \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world", "model":"nllb", "lang":"my", "direction":"e2f"}'
echo ""

echo "--- Testing Scratch English -> Burmese (Should Fail) ---"
curl -X POST http://localhost:5001/api/translate \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world", "model":"scratch", "lang":"my", "direction":"e2f"}'
echo ""

# Kill App
kill $PID
echo "Test Complete"
