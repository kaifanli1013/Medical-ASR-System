import gradio as gr 
from transformers import pipeline
 
pipe = pipeline("text-classification")
 
def clf(text):
    result = pipe(text)
    label = result[0]['label'] # result[0]
    score = result[0]['score']
    res = {label:score,'POSITIVE' if label=='NEGATIVE' else 'NEGATIVE': 1-score}
    return res 
 
demo = gr.Interface(fn=clf, inputs="text", outputs="label")
gr.close_all()
demo.launch(share=True)
