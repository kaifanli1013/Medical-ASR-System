# Medical-ASR-System

- dependencies
    requirments.txt

- Pipeline
    - Default model: faster-whisper large-v2

- running
1. 启动的时候使用: python demo.py --share 

- debug
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
    

## TODO List
- Diarization
    - 提供UI接口来调整speaker-diarization模型所能识别的最小和最大人数
        - ~~UI界面添加slider~~
        - ~~模型初始化添加字段~~

- 生成的字幕文件先转换为pandas的dataframe文件, 然后转换为excel文档

- Gradio
    - 重载模式: https://www.gradio.app/guides/developing-faster-with-reload-mode