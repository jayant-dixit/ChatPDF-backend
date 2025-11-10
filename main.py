from fastapi import FastAPI
import socketio
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from router import router

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
    await sio.emit('response', {'data': 'Message received: ' + data}, to=sid)


if __name__ == "__main__":
    uvicorn.run(sio_app, host="0.0.0.0", port=8000, log_level="warning")
