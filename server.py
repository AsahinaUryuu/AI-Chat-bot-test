from fastapi import FastAPI, WebSocket
import uvicorn
import google.generativeai as genai
import asyncio

# ================= 配置区域 =================
# ⚠️ 请把这里换成你刚才复制的 API Key
MY_API_KEY = "XXXXXXXXXXXX"

# 配置 Gemini
genai.configure(api_key=MY_API_KEY)
# 使用 Gemini 3 Flash，因为它最快，适合实时聊天
model = genai.GenerativeModel('gemini-3-flash-preview')

# 初始化对话历史（为了让它有记忆）
chat_session = model.start_chat(history=[
    {"role": "user", "parts": ["你现在是一个名为'艾莉'的二次元虚拟助手，性格活泼可爱，喜欢用颜文字。请用简短的口语回答我。"]}
])
# ===========================================

app = FastAPI()

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ 客户端已连接 (流式模式)")
    
    try:
        while True:
            # 1. 接收用户消息
            user_text = await websocket.receive_text()
            print(f"👂 收到: {user_text}")
            
            # 2. 调用 Gemini (开启 stream=True)
            response = await chat_session.send_message_async(user_text, stream=True)
            
            # 3. 循环把生成的碎片推给前端
            async for chunk in response:
                if chunk.text:
                    print(f"碎片: {chunk.text}") # 调试用，嫌吵可以注释掉
                    await websocket.send_text(chunk.text)
            
            # 4. (可选) 发送一个特殊标记，告诉前端“我说完了”
            # 比如: await websocket.send_text("[END]") 
            # 但目前为了简单，我们先不加，靠前端逻辑判断
            
            print("✅ 回复完成")
            
    except Exception as e:
        print(f"❌ 连接断开: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)