import os
import signal
import traceback

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from inference import Inference

APP = FastAPI()

tts_pipline = Inference()


class TTSRequest(BaseModel):
    text: str = None
    speaker_id: str = ""
    trace_id: str = ""


async def tts_handle(trace_id, text, speaker_id):
    return tts_pipline.inference(trace_id, text, speaker_id)


@APP.get("/api/check-health")
async def check_health():
    return "OK"


@APP.get("/tts")
async def tts(request: TTSRequest):
    return await tts_handle(request, request.text, request.speaker_id)


@APP.post("/audgeneratebase64")
async def tts(request: TTSRequest):
    return await tts_handle(request, request.text, request.speaker_id)


if __name__ == "__main__":
    try:
        uvicorn.run(app=APP, host="0.0.0.0", port=9880, workers=1)
    except Exception as e:
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)
