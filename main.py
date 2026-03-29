from fastapi import FastAPI
import socketio
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from router import router
from llm import llm
import json

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

sio = socketio.AsyncServer(cors_allowed_origins="*", async_mode='asgi')
sio_app = socketio.ASGIApp(sio, other_asgi_app=app)

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@sio.event
async def connect(sid, environ):
    print(f"Client connected: {sid}")
    
@sio.event
async def disconnect(sid):
    print(f"Client disconnected: {sid}")
    
@sio.event
async def message(sid, data):
    print(f"Message from {sid}: {data}")
    MAX_CHARS = 4000  # adjust based on model

    if len(data) > MAX_CHARS:
        data = data[:MAX_CHARS]
    response = llm.invoke(data)
    print(f"Response to {sid}: {response}")
    await sio.emit('response', {'data': json.dumps(response.content)}, to=sid)


if __name__ == "__main__":
    print("Server is running on http://localhost:8000")
    uvicorn.run(sio_app, host="0.0.0.0", port=8000, log_level="warning")
