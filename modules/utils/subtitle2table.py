import pandas as pd
import re

def parse_srt_to_table(files: list):
    
    with open(files[0].name, 'r', encoding='utf-8') as subtitle_file:
        content = subtitle_file.read()

    pattern = r"(SPEAKER_\d+\|.+)"
    matches = re.findall(pattern, content)

    # generate table data
    data = []
    for match in matches:
        # split '|' to get speaker and dialogue
        speaker, dialogue = match.split('|', 1)
        data.append([speaker.strip(), dialogue.strip(), ""])  # EMR列为空
        
    df = pd.DataFrame(data, columns=["Speaker", "Dialogue", "EMR"])

    return df