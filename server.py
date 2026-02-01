from fastapi import FastAPI, WebSocket
import uvicorn

app = FastAPI()

# 这是一个简单的 WebSocket 接口，之后 Unity 就连这个地址
@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    print("等待连接...")
    await websocket.accept()
    print("✅ 客户端已连接！")
    try:
        while True:
            # 1. 接收消息 (可以是文字，以后改成接收音频 bytes)
            data = await websocket.receive_text()
            print(f"收到消息: {data}")
            
            # 2. 模拟 AI 回复
            await websocket.send_text(f"服务端收到: {data}")
    except Exception as e:
        print(f"❌ 连接断开: {e}")

if __name__ == "__main__":
    # 启动服务，监听 8000 端口
    uvicorn.run(app, host="0.0.0.0", port=8000)