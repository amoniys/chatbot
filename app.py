import os
import json
import re
from flask import Flask, request, Response, session, after_this_request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch
import threading

app = Flask(__name__)
app.secret_key = os.urandom(24)

# 使用本地模型路径
model_name = "./local_models/deepseek_r1_distill_qwen_1.5b"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16
).eval()

# 最大对话历史消息数
MAX_HISTORY_LENGTH = 5

@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    chat_id = request.headers.get('X-Chat-ID')
    if not chat_id:
        return jsonify({"error": "No chat ID received"}), 400

    if 'chats' not in session:
        session['chats'] = {}

    if chat_id not in session['chats']:
        session['chats'][chat_id] = []

    data = request.get_json()
    user_message = data.get("message")
    if user_message:
        # 将用户消息添加到对话历史
        session['chats'][chat_id].append({"role": "user", "content": user_message})

        # 限制对话历史长度
        if len(session['chats'][chat_id]) > MAX_HISTORY_LENGTH:
            session['chats'][chat_id] = session['chats'][chat_id][-MAX_HISTORY_LENGTH:]

        # 构建包含对话历史的输入文本
        input_text = ""
        for message in session['chats'][chat_id]:
            input_text += f"{message['role']}: {message['content']}\n"

        def generate():
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, timeout=30.0)
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
            generation_kwargs = dict(


                inputs,
                streamer=streamer,

                max_length=2048,  # 增加最大生成长度
                num_return_sequences=1,
                temperature=0.8,
                top_p=0.95,

                do_sample=True
            )

            thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()

            for new_text in streamer:
                if new_text:
                    # 清理输出内容
                    cleaned_text = re.sub(r'(user:\s*.*?)\1+', r'\1', new_text)
                    # 使用正则表达式移除 <｜end of sentence｜> 标签，忽略大小写
                    cleaned_text = re.sub(r'<\｜end\s*of\s*sentence\｜>', '', cleaned_text, flags=re.IGNORECASE)
                    cleaned_text = cleaned_text.replace('</think>', '')
                    cleaned_text = re.sub(r'user:\s*', '', cleaned_text, flags=re.IGNORECASE)
                    yield f"data: {json.dumps({'response': cleaned_text})}\n\n"
            yield "data: [DONE]\n\n"

        return Response(generate(), mimetype="text/event-stream")
    return jsonify({"error": "No message received"}), 400


if __name__ == "__main__":
    app.run(debug=True)
