import os
from flask import Flask, request, jsonify
import gradio as gr

# 创建 Flask 应用
app = Flask(__name__)

# 创建上传目录用于保存录制的音频文件
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 处理上传的音频文件
@app.route("/upload_audio", methods=["POST"])
def upload_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    file_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
    audio_file.save(file_path)

    print(f"Saved audio to {file_path}")

    return jsonify({"status": "success", "file_path": file_path})

# Gradio 前端 UI，包括录音控制和上传逻辑
def gradio_ui():
    html_code = """
    <script>
    let mediaRecorder;
    let recordedChunks = [];

    // 开始录音函数
    function startRecording() {
        navigator.mediaDevices.getUserMedia({ audio: true })
        .then(function(stream) {
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.ondataavailable = function(event) {
                if (event.data.size > 0) {
                    recordedChunks.push(event.data);
                }
            };
            mediaRecorder.start();
            console.log('Recording started');
        })
        .catch(function(err) {
            console.log('Error accessing microphone: ' + err);
        });
    }

    // 停止录音并上传音频
    function stopRecording() {
        mediaRecorder.stop();
        mediaRecorder.onstop = function() {
            const blob = new Blob(recordedChunks, { type: 'audio/webm' });
            const formData = new FormData();
            formData.append("audio", blob, "recorded_audio.webm");

            // 上传录制的音频文件
            fetch("/upload_audio", {
                method: "POST",
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                console.log('Audio uploaded: ', data);
                alert('Audio uploaded successfully!');
            })
            .catch(error => console.error('Error uploading audio:', error));
        };
        console.log('Recording stopped');
    }
    </script>

    <button onclick="startRecording()">Start Recording</button>
    <button onclick="stopRecording()">Stop and Upload</button>
    """
    return html_code

# Gradio 接口
demo = gr.Blocks()

with demo:
    gr.HTML(gradio_ui())

# 运行 Flask 和 Gradio 应用
if __name__ == "__main__":
    demo.launch()  # 启动 Gradio 界面
    app.run(port=7860, debug=True)  # 启动 Flask 应用
