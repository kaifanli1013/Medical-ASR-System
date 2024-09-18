import gradio as gr 
import pandas as pd 
from ultralytics import YOLO
from skimage import data
from PIL import Image
 
model = YOLO('yolov8n-cls.pt')
def predict(img):
    result = model.predict(source=img)
    df = pd.Series(result[0].names).to_frame()
    df.columns = ['names']
    df['probs'] = result[0].probs
    df = df.sort_values('probs',ascending=False)
    res = dict(zip(df['names'],df['probs']))
    return res
gr.close_all() 
demo = gr.Interface(fn = predict,inputs = gr.Image(type='pil'), outputs = gr.Label(num_top_classes=5), 
                    examples = ['cat.jpeg','people.jpeg','coffee.jpeg'])
demo.launch()