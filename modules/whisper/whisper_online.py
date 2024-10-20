import sys
import time 
import numpy as np
import logging
from functools import lru_cache
import librosa
from dataclasses import dataclass

logger = logging.getLogger(__name__)
logfile = sys.stderr

# TODO: we need a better way to handle the default args
@dataclass
class default_args:
    start_at: int = 0 # 从第几秒开始
    min_chunk_size: float = 1.0 # Minimum audio chunk size in seconds. It waits up to this time to do processing. If the processing takes shorter time, it waits, otherwise it processes the whole segment that was received by this time.
    buffer_trimming: str = "segment" # choices=["sentence", "segment"],help='Buffer trimming strategy -- trim completed sentences marked with punctuation mark and detected by sentence segmenter, or the completed segments returned by Whisper. Sentence segmenter must be installed for "sentence" option.')
    buffer_trimming_sec: float = 15.0 # help='Buffer trimming threshold in seconds. If the buffer is longer than this, it is trimmed. Default is 15 seconds.'

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
    #    - emission time from beginning of processing, in milliseconds
    #    - beg and end timestamp of the text segment, as estimated by Whisper model. The timestamps are not accurate, but they're useful anyway
    # - the next words: segment transcript
    if now is None:
        now = time.time()-start
    if o[0] is not None:
        print('error here?') 
        print("%1.4f %1.0f %1.0f %s" % (now*1000, o[0]*1000,o[1]*1000,o[2]),file=logfile,flush=True)
        print("%1.4f %1.0f %1.0f %s" % (now*1000, o[0]*1000,o[1]*1000,o[2]),flush=True)
    else:
        # No text, so no output
        pass
        
# def online_inference(audio_file, online_model, args):
#     SAMPLING_RATE = 16000 # FIXME: 
#     duration = len(load_audio(audio_file))/SAMPLING_RATE
#     beg = default_args_instance.start_at
#     start = time.time() - beg # FIXME: 改成一个可控参数？
#     end = 0
#     min_chunk = default_args_instance.min_chunk_size
#     while True:
#         now = time.time() - start
#         if now < end+min_chunk:
#             time.sleep(min_chunk+end-now)
#         end = time.time() - start
#         audio_chunk = load_audio_chunk(audio_file, beg, end)
#         beg = end
#         online_model.insert_audio_chunk(audio_chunk)
        
#         try:
#             output= online_model.process_iter(args)
#             print(f"output: {output}")
            
#         except AssertionError as e:
#             logger.error(f"AssertionError: {e}")
#             pass
#         else:
#             print('pass the assertion')
#             output_transcript(output, start)
#             print('error1?')           
#         now = time.time() - start
#         logger.debug(f"## last processed {end:.2f} s, now is {now:.2f}, the latency is {now-end:.2f}")
        
#         if end >= duration:
#             break
        
#     now = None
    
#     print("Finishing the processing time 1", file=logfile, flush=True)
#     output = online_model.finish()
#     print("Finishing the processing time 2", file=logfile, flush=True)
#     output_transcript(output, now=now)
#     print("Finishing the processing time 3", file=logfile, flush=True)
#     return output
def online_inference(audio_file, online_model, args):

    SAMPLING_RATE = 16000  # FIXME: 
    duration = len(load_audio(audio_file)) / SAMPLING_RATE
    print("Loaded audio file successfully.")

    try:
        beg = default_args_instance.start_at
        start = time.time() - beg
        # start = time.time() - beg  # FIXME: 改成一个可控参数？
        end = 0
        min_chunk = default_args_instance.min_chunk_size
        print(f"Initialized parameters: beg={beg}, start={start}, min_chunk={min_chunk}")
    except Exception as e:
        logger.error(f"Error in initializing parameters: {e}")
        return None

    results = []
    while True:
        try:
            now = time.time() - start
            if now < end + min_chunk:
                time.sleep(min_chunk + end - now)
        except Exception as e:
            logger.error(f"Error in controlling sleep timing: {e}")
            return None

        try:
            end = time.time() - start
            audio_chunk = load_audio_chunk(audio_file, beg, end)
            print(f"Loaded audio chunk from {beg} to {end}.")
            beg = end
        except Exception as e:
            logger.error(f"Error in loading audio chunk: {e}")
            return None

        try:
            online_model.insert_audio_chunk(audio_chunk)
            print("Inserted audio chunk successfully.")
        except Exception as e:
            logger.error(f"Error in inserting audio chunk into the model: {e}")
            return None

        try:
            output = online_model.process_iter(args)
            results.append(output)
            print(f"Output: {output}")
        # except AssertionError as e:
        #     logger.error(f"AssertionError in processing iteration: {e}")
        #     pass
        except Exception as e:
            logger.error(f"Error in processing iteration: {e}")
            return None
        else:
            try:
                print('Passed the assertion.')
                output_transcript(output, start)
                print('Transcript output successfully.')
            except Exception as e:
                logger.error(f"Error in outputting transcript: {e}")
                return None

        try:
            now = time.time() - start
            logger.debug(f"## Last processed {end:.2f} s, now is {now:.2f}, the latency is {now - end:.2f}")
        except Exception as e:
            logger.error(f"Error in logging debug information: {e}")
            return None

        if end >= duration:
            break

    try:
        now = None
        print("Finishing the processing time 1", file=logfile, flush=True)
        output = online_model.finish()
        print("Finishing the processing time 2", file=logfile, flush=True)
        results.append(output)
    except Exception as e:
        logger.error(f"Error in finishing the model processing: {e}")
        return None

    # try:
    #     output_transcript(output, start=start, now=now)
    #     print("Finishing the processing time 3", file=logfile, flush=True)
    # except Exception as e:
    #     logger.error(f"Error in final outputting transcript: {e}")
    #     return None

    print('Results:', results)
    return online_model.to_flush(results)

def online_model_init(asr, language, logfile=logfile):
    target_language = language
    print(f'Initializing online model for {target_language}', file=logfile, flush=True)
    # Create the tokenizer
    if default_args_instance.buffer_trimming == "sentence":
        tokenizer = create_tokenizer(target_language)
    else:
        tokenizer = None
    
    asr.sep = ""
    online_model = OnlineASRProcessor(asr,tokenizer,logfile=logfile,buffer_trimming=(default_args_instance.buffer_trimming, default_args_instance.buffer_trimming_sec))

    return online_model

@lru_cache
def load_audio(fname):
    a, _ = librosa.load(fname, sr=16000, dtype=np.float32)
    return a

def load_audio_chunk(fname, beg, end):
    audio = load_audio(fname)
    beg_s = int(beg*16000)
    end_s = int(end*16000)
    return audio[beg_s:end_s]

WHISPER_LANG_CODES = "af,am,ar,as,az,ba,be,bg,bn,bo,br,bs,ca,cs,cy,da,de,el,en,es,et,eu,fa,fi,fo,fr,gl,gu,ha,haw,he,hi,hr,ht,hu,hy,id,is,it,ja,jw,ka,kk,km,kn,ko,la,lb,ln,lo,lt,lv,mg,mi,mk,ml,mn,mr,ms,mt,my,ne,nl,nn,no,oc,pa,pl,ps,pt,ro,ru,sa,sd,si,sk,sl,sn,so,sq,sr,su,sv,sw,ta,te,tg,th,tk,tl,tr,tt,uk,ur,uz,vi,yi,yo,zh".split(",")
def create_tokenizer(lan):
    """returns an object that has split function that works like the one of MosesTokenizer"""

    assert lan in WHISPER_LANG_CODES, "language must be Whisper's supported lang code: " + " ".join(WHISPER_LANG_CODES)

    if lan == "uk":
        import tokenize_uk
        class UkrainianTokenizer:
            def split(self, text):
                return tokenize_uk.tokenize_sents(text)
        return UkrainianTokenizer()

    # supported by fast-mosestokenizer
    if lan in "as bn ca cs de el en es et fi fr ga gu hi hu is it kn lt lv ml mni mr nl or pa pl pt ro ru sk sl sv ta te yue zh".split():
        from mosestokenizer import MosesTokenizer
        return MosesTokenizer(lan)

    # the following languages are in Whisper, but not in wtpsplit:
    if lan in "as ba bo br bs fo haw hr ht jw lb ln lo mi nn oc sa sd sn so su sw tk tl tt".split():
        logger.debug(f"{lan} code is not supported by wtpsplit. Going to use None lang_code option.")
        lan = None

    from wtpsplit import WtP
    # downloads the model from huggingface on the first use
    wtp = WtP("wtp-canine-s-12l-no-adapters")
    class WtPtok:
        def split(self, sent):
            return wtp.split(sent, lang_code=lan)
    return WtPtok()

class HypothesisBuffer:

    def __init__(self, logfile=sys.stderr):
        self.commited_in_buffer = [] # 代表已确认并稳定的文本片段，图中黄色高亮部分
        self.buffer = [] # 对应Update N-1中黑框的内容 
        self.new = [] # 对应Update N中的黑框内容 

        self.last_commited_time = 0 # 对应于图中蓝色垂直线
        self.last_commited_word = None # 对应于图中的绿色下划线的最后一个确认单词

        self.logfile = logfile 

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
                            logger.debug(f"removing last {i} words: {words_msg}")
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

    def __init__(self, asr, tokenizer=None, buffer_trimming=("segment", 15), logfile=sys.stderr):
        """asr: WhisperASR object
        tokenizer: sentence tokenizer object for the target language. Must have a method *split* that behaves like the one of MosesTokenizer. It can be None, if "segment" buffer trimming option is used, then tokenizer is not used at all.
        ("segment", 15)
        buffer_trimming: a pair of (option, seconds), where option is either "sentence" or "segment", and seconds is a number. Buffer is trimmed if it is longer than "seconds" threshold. Default is the most recommended option.
        logfile: where to store the log. 
        """
        self.asr = asr
        self.tokenizer = tokenizer
        self.logfile = logfile

        self.init()

        self.buffer_trimming_way, self.buffer_trimming_sec = buffer_trimming

    def init(self, offset=None):
        """run this when starting or restarting processing"""
        self.audio_buffer = np.array([],dtype=np.float32)
        self.transcript_buffer = HypothesisBuffer(logfile=self.logfile)
        self.buffer_time_offset = 0
        if offset is not None:
            self.buffer_time_offset = offset
        self.transcript_buffer.last_commited_time = self.buffer_time_offset
        self.commited = []

    def insert_audio_chunk(self, audio):
        self.audio_buffer = np.append(self.audio_buffer, audio)

    def prompt(self):
        """Returns a tuple: (prompt, context), where "prompt" is a 200-character suffix of commited text that is inside of the scrolled away part of audio buffer. 
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

    # def process_iter(self, args):
    #     """Runs on the current audio buffer.
    #     Returns: a tuple (beg_timestamp, end_timestamp, "text"), or (None, None, ""). 
    #     The non-emty text is confirmed (committed) partial transcript.
    #     """

    #     prompt, non_prompt = self.prompt()
    #     logger.debug(f"PROMPT: {prompt}")
    #     logger.debug(f"CONTEXT: {non_prompt}")
    #     logger.debug(f"transcribing {len(self.audio_buffer)/self.SAMPLING_RATE:2.2f} seconds from {self.buffer_time_offset:2.2f}")
    #     res, info = self.asr.transcribe(self.audio_buffer, initial_prompt=prompt, **args)

    #     res = list(res)
    #     # print(f"res: {res}")
        
        
    #     # transform to [(beg,end,"word1"), ...]
    #     tsw = ts_words(res)
        
    #     self.transcript_buffer.insert(tsw, self.buffer_time_offset)
    #     o = self.transcript_buffer.flush()
    #     self.commited.extend(o)
    #     completed = self.to_flush(o)
    #     logger.debug(f">>>>COMPLETE NOW: {completed}")
    #     the_rest = self.to_flush(self.transcript_buffer.complete())
    #     logger.debug(f"INCOMPLETE: {the_rest}")

    #     # there is a newly confirmed text

    #     if o and self.buffer_trimming_way == "sentence":  # trim the completed sentences
    #         if len(self.audio_buffer)/self.SAMPLING_RATE > self.buffer_trimming_sec:  # longer than this
    #             self.chunk_completed_sentence()

        
    #     if self.buffer_trimming_way == "segment":
    #         s = self.buffer_trimming_sec  # trim the completed segments longer than s,
    #     else:
    #         s = 30 # if the audio buffer is longer than 30s, trim it
        
    #     if len(self.audio_buffer)/self.SAMPLING_RATE > s:
    #         self.chunk_completed_segment(res)

    #         # alternative: on any word
    #         #l = self.buffer_time_offset + len(self.audio_buffer)/self.SAMPLING_RATE - 10
    #         # let's find commited word that is less
    #         #k = len(self.commited)-1
    #         #while k>0 and self.commited[k][1] > l:
    #         #    k -= 1
    #         #t = self.commited[k][1] 
    #         logger.debug("chunking segment")
    #         #self.chunk_at(t)

    #     logger.debug(f"len of buffer now: {len(self.audio_buffer)/self.SAMPLING_RATE:2.2f}")
    #     return self.to_flush(o)
    def process_iter(self, args):
        """Runs on the current audio buffer.
        Returns: a tuple (beg_timestamp, end_timestamp, "text"), or (None, None, ""). 
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
            # print(f"res: {res}")
            # print(f"Transcription result: {res}")
        except Exception as e:
            print(f"Error during transcription: {e}")
            return (None, None, "")

        try:
            tsw = ts_words(res)
            self.transcript_buffer.insert(tsw, self.buffer_time_offset)
            o = self.transcript_buffer.flush() # return commited chunk
            self.commited.extend(o)
            logger.debug(f"Inserted and flushed transcript: {o}")
        except Exception as e:
            logger.error(f"Error processing transcript buffer: {e}")
            return (None, None, "")

        try:
            completed = self.to_flush(o)
            print(f">>>>COMPLETE NOW: {completed}")
            the_rest = self.to_flush(self.transcript_buffer.complete())
            print(f"INCOMPLETE: {the_rest}")
        except Exception as e:
            print(f"Error finalizing flush: {e}")
            return (None, None, "")

        try:
            if o and self.buffer_trimming_way == "sentence":  # trim the completed sentences
                if len(self.audio_buffer)/self.SAMPLING_RATE > self.buffer_trimming_sec:  # longer than this
                    self.chunk_completed_sentence()
                    print("Trimmed completed sentences.")
        except Exception as e:
            print(f"Error trimming completed sentences: {e}")
            return (None, None, "")

        try:
            if self.buffer_trimming_way == "segment":
                s = self.buffer_trimming_sec  # trim the completed segments longer than s,
            else:
                s = 30  # if the audio buffer is longer than 30s, trim it

            if len(self.audio_buffer)/self.SAMPLING_RATE > s:
                self.chunk_completed_segment(res)
                print("Chunked completed segment.")
        except Exception as e:
            logger.error(f"Error trimming or chunking segment: {e}")
            return (None, None, "")

        try:
            return self.to_flush(o)
        except Exception as e:
            print(f"Error returning final output: {e}")
            return (None, None, "")


    # def chunk_completed_sentence(self):
    #     if self.commited == []: return
    #     logger.debug(self.commited)
    #     sents = self.words_to_sentences(self.commited)
    #     for s in sents:
    #         logger.debug(f"\t\tSENT: {s}")
    #     if len(sents) < 2:
    #         return
    #     while len(sents) > 2:
    #         sents.pop(0)
    #     # we will continue with audio processing at this timestamp
    #     chunk_at = sents[-2][1]

    #     logger.debug(f"--- sentence chunked at {chunk_at:2.2f}")
    #     self.chunk_at(chunk_at)

    def chunk_completed_sentence(self):
        try:
            if self.commited == []: 
                return
            logger.debug(self.commited)

            # 尝试将 words 转换为 sentences
            try:
                sents = self.words_to_sentences(self.commited)
            except Exception as e:
                print(f"Error in words_to_sentences: {e}")
                return

            # 打印生成的 sentences
            try:
                for s in sents:
                    logger.debug(f"\t\tSENT: {s}")
            except Exception as e:
                print(f"Error logging sentences: {e}")
                return

            # 如果生成的 sentences 少于两个，则返回
            if len(sents) < 2:
                return

            # 尝试修剪 sentences 列表
            try:
                while len(sents) > 2:
                    sents.pop(0)
            except Exception as e:
                print(f"Error trimming sentences: {e}")
                return

            # 继续处理音频
            try:
                chunk_at = sents[-2][1]
                print(f"--- sentence chunked at {chunk_at:2.2f}")
                self.chunk_at(chunk_at)
            except Exception as e:
                print(f"Error chunking at sentence: {e}")
                return

        except Exception as e:
            logger.error(f"Error in chunk_completed_sentence: {e}")

    # def chunk_completed_segment(self, res):
    #     if self.commited == []: return

    #     ends = segments_end_ts(res)

    #     t = self.commited[-1][1]

    #     if len(ends) > 1:

    #         e = ends[-2]+self.buffer_time_offset
    #         while len(ends) > 2 and e > t:
    #             ends.pop(-1)
    #             e = ends[-2]+self.buffer_time_offset
    #         if e <= t:
    #             logger.debug(f"--- segment chunked at {e:2.2f}")
    #             self.chunk_at(e)
    #         else:
    #             logger.debug(f"--- last segment not within commited area")
    #     else:
    #         logger.debug(f"--- not enough segments to chunk")
    
    def chunk_completed_segment(self, res):
        try:
            if self.commited == []: 
                return

            try:
                ends = segments_end_ts(res)
            except Exception as e:
                logger.error(f"Error in calling segments_end_ts: {e}")
                return

            try:
                t = self.commited[-1][1]
            except Exception as e:
                logger.error(f"Error accessing last commited timestamp: {e}")
                return

            if len(ends) > 1:
                try:
                    e = ends[-2] + self.buffer_time_offset
                    while len(ends) > 2 and e > t:
                        try:
                            ends.pop(-1)
                            e = ends[-2] + self.buffer_time_offset
                        except Exception as e:
                            logger.error(f"Error popping and updating ends: {e}")
                            return

                    if e <= t:
                        try:
                            logger.debug(f"--- segment chunked at {e:2.2f}")
                            self.chunk_at(e)
                        except Exception as e:
                            logger.error(f"Error calling chunk_at: {e}")
                            return
                    else:
                        logger.debug(f"--- last segment not within commited area")
                except Exception as e:
                    logger.error(f"Error in segment chunking logic: {e}")
                    return
            else:
                logger.debug(f"--- not enough segments to chunk")
        except Exception as e:
            logger.error(f"Error in chunk_completed_segment: {e}")

    
    def chunk_completed_sentence(self):
        try:
            if self.commited == []: 
                return
            logger.debug(self.commited)

            # 尝试将 words 转换为 sentences
            try:
                sents = self.words_to_sentences(self.commited)
            except Exception as e:
                logger.error(f"Error in words_to_sentences: {e}")
                return

            # 打印生成的 sentences
            try:
                for s in sents:
                    logger.debug(f"\t\tSENT: {s}")
            except Exception as e:
                logger.error(f"Error logging sentences: {e}")
                return

            # 如果生成的 sentences 少于两个，则返回
            if len(sents) < 2:
                return

            # 尝试修剪 sentences 列表
            try:
                while len(sents) > 2:
                    sents.pop(0)
            except Exception as e:
                logger.error(f"Error trimming sentences: {e}")
                return

            # 继续处理音频
            try:
                chunk_at = sents[-2][1]
                logger.debug(f"--- sentence chunked at {chunk_at:2.2f}")
                self.chunk_at(chunk_at)
            except Exception as e:
                logger.error(f"Error chunking at sentence: {e}")
                return

        except Exception as e:
            logger.error(f"Error in chunk_completed_sentence: {e}")

    def chunk_at(self, time):
        """trims the hypothesis and audio buffer at "time"
        """
        self.transcript_buffer.pop_commited(time)
        cut_seconds = time - self.buffer_time_offset
        self.audio_buffer = self.audio_buffer[int(cut_seconds*self.SAMPLING_RATE):]
        self.buffer_time_offset = time

    def words_to_sentences(self, words):
        """Uses self.tokenizer for sentence segmentation of words.
        Returns: [(beg,end,"sentence 1"),...]
        """
        
        cwords = [w for w in words]
        t = " ".join(o[2] for o in cwords)
        s = self.tokenizer.split(t)
        out = []
        while s:
            beg = None
            end = None
            sent = s.pop(0).strip()
            fsent = sent
            while cwords:
                b,e,w = cwords.pop(0)
                w = w.strip()
                if beg is None and sent.startswith(w):
                    beg = b
                elif end is None and sent == w:
                    end = e
                    out.append((beg,end,fsent))
                    break
                sent = sent[len(w):].strip()
        return out

    def finish(self):
        """Flush the incomplete text when the whole processing ends.
        Returns: the same format as self.process_iter()
        """
        o = self.transcript_buffer.complete()
        f = self.to_flush(o)
        logger.debug(f"last, noncommited: {f}")
        self.buffer_time_offset += len(self.audio_buffer)/16000
        return o


    def to_flush(self, sents, sep=None, offset=0, ):
        # concatenates the timestamped words or sentences into one sequence that is flushed in one line
        # sents: [(beg1, end1, "sentence1"), ...] or [] if empty
        # return: (beg1,end-of-last-sentence,"concatenation of sentences") or (None, None, "") if empty
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