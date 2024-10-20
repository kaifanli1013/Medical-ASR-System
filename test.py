from faster_whisper import WhisperModel

model = WhisperModel("large-v3")

segments, info = model.transcribe("test.wav", word_timestamps=True)
results = []
for _ in segments:
    results.append(_)
    
print(results)
