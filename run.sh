nohup python -m uvicorn api_v3:APP --host 0.0.0.0 --port 7080 --workers 4 >> /workspace/res/out.txt 2>&1 &
