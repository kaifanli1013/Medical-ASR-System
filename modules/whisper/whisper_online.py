import sys
import time 
import numpy as np
from functools import lru_cache
import librosa
from dataclasses import dataclass

# TODO: we need a better way to handle the default args
@dataclass
class default_args:
    start_at: float = 0.0 # Start processing at this time in seconds. Default is 0.
    min_chunk_size: float = 1.0 # Minimum audio chunk size in seconds. It waits up to this time to do processing. If the processing takes shorter time, it waits, otherwise it processes the whole segment that was received by this time.
    buffer_trimming_sec: float = 15.0 # help='Buffer trimming threshold in seconds. If the buffer is longer than this, it is trimmed. Default is 15 seconds.'
    sampling_rate: int = 16000
    
default_args_instance = default_args()

def segments_end_ts(res):
    return [s.end for s in res]

def ts_words(segments):
    o = []
    for segment in segments:
        for word in segment.words:
            if segment.no_speech_prob > 0.9:
                continue
            # not stripping the spaces -- should not be merged with them!
            w = word.word
            t = (word.start, word.end, w)
            o.append(t)
    return o
    
def output_transcript(o, start, now=None):
    # output format in stdout is like:
    # 4186.3606 0 1720 Takhle to je
    # - the first three words are:
    #    - emission time from startinning of processing, in milliseconds
    #    - start and end timestamp of the text segment, as estimated by Whisper model. The timestamps are not accurate, but they're useful anyway
    # - the next words: segment transcript
    if now is None:
        now = time.time()-start
    if o[0] is not None: 
        print("%1.4f %1.0f %1.0f %s" % (now*1000, o[0]*1000,o[1]*1000,o[2]),flush=True)
    else:
        # No text, so no output
        pass
        

def online_inference(audio_file, online_model, args):

    # sampling rate for the audio file
    SAMPLING_RATE = default_args_instance.sampling_rate
    
    # duration of the audio file
    try:
        duration = len(load_audio(audio_file)) / SAMPLING_RATE
        print("Audio duration is %2.2f seconds" % duration)
    except Exception as e:
        print(f"Error in loading audio file: {e}")
        return None
    
    min_chunk = default_args_instance.min_chunk_size # Minimum audio chunk size in seconds. It waits up to this time to do processing. If the processing takes shorter time, it waits, otherwise it processes the whole segment that was received by this time.
    
    start = default_args_instance.start_at # start time of the processing in audio file
    start_time = time.time() - start # start time of the processing in real time
    end = 0
    
    while True:        
        now = time.time() - start_time
        if now < end + min_chunk: # wait for the min_chunk time
            time.sleep(min_chunk + end - now)

        # loading the audio chunk
        try:
            end = time.time() - start_time
            audio_chunk = load_audio_chunk(audio_file, start, end)
            print(f"Loaded audio chunk from {start:.2f} to {end:.2f}.")
            start = end
        except Exception as e:
            print(f"Error in loading audio chunk: {e}")
            return None

        # inserting the audio chunk into the model
        online_model.insert_audio_chunk(audio_chunk)

        # processing the iteration
        try:
            output = online_model.process_iter(args)
        except Exception as e:
            print(f"Error in processing iteration: {e}")
            return None
        else:
            try:
                print('Passed the assertion.')
                output_transcript(output, start_time)
                print('Transcript output successfully.')
            except Exception as e:
                print(f"Error in outputting transcript: {e}")
                return None

        now = time.time() - start_time
        print(f"## Last processed {end:.2f} s, now is {now:.2f}, the latency is {now - end:.2f}")

        if end >= duration:
            break

    o = online_model.finish()
    output_transcript(o, start_time, now)

    result = online_model.commited.copy() 
    result.extend(online_model.transcript_buffer.complete())
    
    return result, duration

@lru_cache
def load_audio(fname):
    try:
        a, _ = librosa.load(fname, sr=16000, dtype=np.float32)
    except Exception as e:
        print(f"Error in loading audio file: {e}")
    return a

def load_audio_chunk(fname, start, end):
    audio = load_audio(fname)
    start_s = int(start*16000)
    end_s = int(end*16000)
    return audio[start_s:end_s]

class HypothesisBuffer:

    def __init__(self):
        self.commited_in_buffer = [] # 代表已确认并稳定的文本片段，图中黄色高亮部分
        self.buffer = [] # 对应Update N-1中黑框的内容 
        self.new = [] # 对应Update N中的黑框内容 

        self.last_commited_time = 0 # 对应于图中蓝色垂直线
        self.last_commited_word = None # 对应于图中的绿色下划线的最后一个确认单词

    # 将新的片段插入到缓冲区中，并检测和去除在已确认的输出与新的片段之间的重复部分。
    def insert(self, new, offset):
        # compare self.commited_in_buffer and new. 
        # It inserts only the words in new that extend the commited_in_buffer, 
        #   it means they are roughly behind last_commited_time and new in content
        # the new tail is added to self.new
        
        new = [(a+offset,b+offset,t) for a,b,t in new]
        self.new = [(a,b,t) for a,b,t in new if a > self.last_commited_time-0.1]

        if len(self.new) >= 1:
            a,b,t = self.new[0]
            if abs(a - self.last_commited_time) < 1:
                if self.commited_in_buffer:
                    # it's going to search for 1, 2, ..., 5 consecutive words (n-grams) that are identical in commited and new. If they are, they're dropped.
                    cn = len(self.commited_in_buffer)
                    nn = len(self.new)
                    for i in range(1,min(min(cn,nn),5)+1):  # 5 is the maximum 
                        c = " ".join([self.commited_in_buffer[-j][2] for j in range(1,i+1)][::-1])
                        tail = " ".join(self.new[j-1][2] for j in range(1,i+1))
                        if c == tail:
                            words = []
                            for j in range(i):
                                words.append(repr(self.new.pop(0)))
                            words_msg = " ".join(words)
                            print(f"removing last {i} words: {words_msg}")
                            break

    # 找到连续插入中稳定且一致的部分，将这些部分标记为已确认的输出。
    def flush(self):
        # returns commited chunk = the longest common prefix of 2 last inserts. 

        commit = []
        while self.new:
            na, nb, nt = self.new[0]

            if len(self.buffer) == 0:
                break
            
            # 比较 self.new 的第一个单词与 self.buffer 的第一个单词
            if nt == self.buffer[0][2]:
                commit.append((na,nb,nt))
                self.last_commited_word = nt
                self.last_commited_time = nb
                self.buffer.pop(0)
                self.new.pop(0)
            else:
                break
        self.buffer = self.new
        self.new = []
        self.commited_in_buffer.extend(commit)
        return commit

    def pop_commited(self, time):
        while self.commited_in_buffer and self.commited_in_buffer[0][1] <= time:
            self.commited_in_buffer.pop(0)

    def complete(self):
        return self.buffer

class OnlineASRProcessor:

    SAMPLING_RATE = 16000

    def __init__(self, asr, buffer_trimming_sec=15):
        """asr: WhisperASR object
        tokenizer: sentence tokenizer object for the target language. Must have a method *split* that behaves like the one of MosesTokenizer. It can be None, if "segment" buffer trimming option is used, then tokenizer is not used at all.
        ("segment", 15)
        buffer_trimming: a pair of (option, seconds), where option is either "sentence" or "segment", and seconds is a number. Buffer is trimmed if it is longer than "seconds" threshold. Default is the most recommended option.
        logfile: where to store the log. 
        """
        self.asr = asr
        self.init()
        self.buffer_trimming_sec = buffer_trimming_sec

    def init(self, offset=None):
        """run this when starting or restarting processing"""
        self.audio_buffer = np.array([],dtype=np.float32)
        self.transcript_buffer = HypothesisBuffer()
        self.buffer_time_offset = 0
        if offset is not None:
            self.buffer_time_offset = offset
        self.transcript_buffer.last_commited_time = self.buffer_time_offset
        self.commited = []

    def insert_audio_chunk(self, audio):
        self.audio_buffer = np.append(self.audio_buffer, audio)

    def prompt(self):
        """
        prompt: 已经确认的文本，图中黄色高亮部分
        non_prompt: 已经确认但是还在buffer里的文本, 图中黑框部分
        
        Returns a tuple: (prompt, context), where "prompt" is a 200-character suffix of commited text that is inside of the scrolled away part of audio buffer. 
        "context" is the commited text that is inside the audio buffer. It is transcribed again and skipped. It is returned only for debugging and logging reasons.
        """
        k = max(0,len(self.commited)-1)
        while k > 0 and self.commited[k-1][1] > self.buffer_time_offset:
            k -= 1

        p = self.commited[:k]
        p = [t for _,_,t in p]
        prompt = []
        l = 0
        while p and l < 200:  # 200 characters prompt size
            x = p.pop(-1)
            l += len(x)+1
            prompt.append(x)
        non_prompt = self.commited[k:]
        return self.asr.sep.join(prompt[::-1]), self.asr.sep.join(t for _,_,t in non_prompt)

    def process_iter(self, args):
        """Runs on the current audio buffer.
        Returns: a tuple (start_timestamp, end_timestamp, "text"), or (None, None, ""). 
        The non-emty text is confirmed (committed) partial transcript.
        """
        
        try:
            prompt, non_prompt = self.prompt()
            print(f"PROMPT: {prompt}")
            print(f"CONTEXT: {non_prompt}")
            print(f"transcribing {len(self.audio_buffer)/self.SAMPLING_RATE:2.2f} seconds from {self.buffer_time_offset:2.2f}")
        except Exception as e:
            print(f"Error generating prompt and context: {e}")
            return (None, None, "")

        try:
            res, info = self.asr.transcribe(self.audio_buffer, initial_prompt=prompt, **args)
            res = list(res)

        except Exception as e:
            print(f"Error during transcription: {e}")
            return (None, None, "")

        try:
            tsw = ts_words(res)
            self.transcript_buffer.insert(tsw, self.buffer_time_offset)
            o = self.transcript_buffer.flush() # return commited chunk
            self.commited.extend(o)
            print(f"Inserted and flushed transcript: {o}")
        except Exception as e:
            print(f"Error processing transcript buffer: {e}")
            return (None, None, "")

        try:
            completed = self.to_flush(o)
            print(f">>>>COMPLETE NOW: {completed}")
            the_rest = self.to_flush(self.transcript_buffer.complete())
            print(f"INCOMPLETE: {the_rest}")
        except Exception as e:
            print(f"Error finalizing flush: {e}")
            return (None, None, "")


        if len(self.audio_buffer)/self.SAMPLING_RATE > self.buffer_trimming_sec:
            self.chunk_completed_segment(res)
            print("Chunked completed segment.")

        try:
            return self.to_flush(o)
        except Exception as e:
            print(f"Error returning final output: {e}")
            return (None, None, "")
    
    def chunk_completed_segment(self, res):
        try:
            if self.commited == []: 
                return

            try:
                ends = segments_end_ts(res)
            except Exception as e:
                print(f"Error in calling segments_end_ts: {e}")
                return

            try:
                t = self.commited[-1][1]
            except Exception as e:
                print(f"Error accessing last commited timestamp: {e}")
                return

            if len(ends) > 1:
                try:
                    e = ends[-2] + self.buffer_time_offset
                    while len(ends) > 2 and e > t:
                        try:
                            ends.pop(-1)
                            e = ends[-2] + self.buffer_time_offset
                        except Exception as e:
                            print(f"Error popping and updating ends: {e}")
                            return

                    if e <= t:
                        try:
                            print(f"--- segment chunked at {e:2.2f}")
                            self.chunk_at(e)
                        except Exception as e:
                            print(f"Error calling chunk_at: {e}")
                            return
                    else:
                        print(f"--- last segment not within commited area")
                except Exception as e:
                    print(f"Error in segment chunking logic: {e}")
                    return
            else:
                print(f"--- not enough segments to chunk")
        except Exception as e:
            print(f"Error in chunk_completed_segment: {e}")

    def chunk_at(self, time):
        """trims the hypothesis and audio buffer at "time"
        """
        self.transcript_buffer.pop_commited(time)
        cut_seconds = time - self.buffer_time_offset
        self.audio_buffer = self.audio_buffer[int(cut_seconds*self.SAMPLING_RATE):]
        self.buffer_time_offset = time

    def words_to_sentences(self, words):
        """Uses self.tokenizer for sentence segmentation of words.
        Returns: [(start,end,"sentence 1"),...]
        """
        
        cwords = [w for w in words]
        t = " ".join(o[2] for o in cwords)
        s = self.tokenizer.split(t)
        out = []
        while s:
            start = None
            end = None
            sent = s.pop(0).strip()
            fsent = sent
            while cwords:
                b,e,w = cwords.pop(0)
                w = w.strip()
                if start is None and sent.startswith(w):
                    start = b
                elif end is None and sent == w:
                    end = e
                    out.append((start,end,fsent))
                    break
                sent = sent[len(w):].strip()
        return out

    def finish(self):
        """Flush the incomplete text when the whole processing ends.
        Returns: the same format as self.process_iter()
        """
        o = self.transcript_buffer.complete()
        f = self.to_flush(o)
        print(f"last, noncommited: {f}")
        self.buffer_time_offset += len(self.audio_buffer)/16000
        return f


    def to_flush(self, sents, sep=None, offset=0, ):
        # concatenates the timestamped words or sentences into one sequence that is flushed in one line
        # sents: [(start1, end1, "sentence1"), ...] or [] if empty
        # return: (start1,end-of-last-sentence,"concatenation of sentences") or (None, None, "") if empty
        if sep is None:
            sep = self.asr.sep
        t = sep.join(s[2] for s in sents)
        if len(sents) == 0:
            b = None
            e = None
        else:
            b = offset + sents[0][0]
            e = offset + sents[-1][1]
        return (b,e,t)