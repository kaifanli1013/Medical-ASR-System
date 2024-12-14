# from faster_whisper import WhisperModel

# model = WhisperModel("large-v3")

# segments, info = model.transcribe("test.wav", word_timestamps=True)
# results = []
# for _ in segments:
#     results.append(_)
    
# print(results)

# import librosa

# def load_audio(audio_path):
#     audio, sr = librosa.load(audio_path, sr=None)
#     return audio, sr

# audio, sr = load_audio("test.wav")
# print(audio, sr)

from faster_whisper import WhisperModel

model = WhisperModel("large-v3")

segments, info = model.transcribe("test01.wav")
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))