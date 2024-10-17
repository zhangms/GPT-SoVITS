import os
import signal
import traceback

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

APP = FastAPI()


class TTSRequest(BaseModel):
    text: str = None
    speaker: str = ""


@APP.get("/api/check-health")
async def check_health():
    return "OK"


@APP.post("/tts")
async def tts(request: TTSRequest):
    req = request.dict()
    return await tts_handle(req)


if __name__ == "__main__":
    try:
        uvicorn.run(app=APP, host="0.0.0.0", port=9880, workers=1)
    except Exception as e:
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)
