# Medical-ASR-System

- dependencies
    requirments.txt

- running
1. 启动的时候使用: python demo.py --share 

- debug
1. Could not load library libcudnn_ops_infer.so.8 
    > https://github.com/SYSTRAN/faster-whisper/issues/516
    > 1. cuda 11.8
    > 2. edit LD_LIBRARY_PATH

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
