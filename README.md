# Medical-ASR-System

- dependencies
  requirments.txt
- Pipeline

  - Default model: faster-whisper large-v3
- running

1. 启动的时候使用: python app.py --share

## Debug List

1. Could not load library libcudnn_ops_infer.so.8

   - https://github.com/SYSTRAN/faster-whisper/issues/516

     1. cuda 11.8
     2. edit LD_LIBRARY_PATH

     ```
     1. nano ~/.bashrc

     2.  export LD_LIBRARY_PATH=/home/is/kaifan-l/anaconda3/envs/whisper-demo/lib/python3.10/site-packages/nvidia/cudnn/lib:/home/is/kaifan-l/anaconda3/envs/whisper-demo/lib/python3.10/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH

     3. source ~/.bashrc

     4. echo $LD_LIBRARY_PATH

     ```
2. OOM ERROR: 本地运行的原因, 还是需要时刻注意
3. whisper_type:

   1. whisper: TODO: bug
   2. faster-whisper: 正常运行
   3. insanely-faster-whisper: TODO: bug
4. diarization:

   - FIXME:  默认使用和whisper模型同一个gpu, 会产生冲突, 目前默认放在cpu上
   - ffmpeg库依赖: conda install ffmpeg

- OpenAI:
  - tools_call:
    - Error code 400: 可能原因是在tools的strict字段设置为True，导致模型强制检查，当模型判断某个字段会出现的时候，则要求一定要出现在'required'列表中。
      `Error code: 400 - {'error': {'message': "Invalid schema for function 'read_conversation_and_summarize_into_electronic_medical_record': In context=(), 'required' is required to be supplied and to be an array including every key in properties. Missing 'physical_examination_findings'.", 'type': 'invalid_request_error', 'param': 'tools[0].function.parameters', 'code': 'invalid_function_parameters'}}`

## TODO List

- whisper

  - **调整每个片段的文本长度**
  - 提供电子病历表格的下载
  - Online:
    - 目前online的duration测量，一律基于采样率16000Hz
    - 调整输出
      - 添加一个窗口可以实时的查看terminal中的输出结果
      - 输出结果目前是独立的单词或者连续的句子，需要分割
- Diarization

  - ~~提供UI接口来调整speaker-diarization模型所能识别的最小和最大人数~~
    - ~~UI界面添加slider~~
    - ~~模型初始化添加字段~~
- Translation

  - 现在只支持 其他语言-> EN
  - 
- Gradio

  - 重载模式: https://www.gradio.app/guides/developing-faster-with-reload-mode
- Open AI

  - 添加电子病历生成功能
    - 添加一个从文本中生成电子病历摘要的功能
    - 电子病历文本标准化
  - 基于相似度的电子病历查询对话匹配功能
  - 是否应该当作NER任务呢
- SIP3 API

  - 现在norm方法存在问题
- UI

  - 自动将SPEAKER_N转化为患者或医生
  - 对患者和医生的对话框进行区分
  - 提供EMR table的可编辑功能
- Mic
- QR Code
