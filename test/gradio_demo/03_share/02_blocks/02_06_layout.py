# # Horizontal layout
# import gradio as gr

# with gr.Blocks() as demo:
#     with gr.Row():
#         img1 = gr.Image() 
#         text1 = gr.Text()
        
#     btn1 = gr.Button("Button")
    
# demo.launch()

## Vertical layout
# import gradio as gr
# with gr.Blocks() as demo:
#     with gr.Row():
#         text1 = gr.Textbox(label="t1")
#         slider2 = gr.Textbox(label="s2")
#         drop3 = gr.Dropdown(["a", "b", "c"], label="d3")
#     with gr.Row():
#         # scale与相邻列相比的相对宽度。例如，如果列A的比例为2，列B的比例为1，则A的宽度将是B的两倍。
#         # min_width设置最小宽度，防止列太窄
#         with gr.Column(scale=2, min_width=600):
#             text1 = gr.Textbox(label="prompt 1")
#             text2 = gr.Textbox(label="prompt 2")
#             inbtw = gr.Button("Between")
#             text4 = gr.Textbox(label="prompt 1")
#             text5 = gr.Textbox(label="prompt 2")
#         with gr.Column(scale=1, min_width=600):
#             img1 = gr.Image("test.jpg")
#             btn = gr.Button("Go")
# demo.launch()


# module visiable
import gradio as gr

with gr.Blocks() as demo:
    # 错误提示框
    error_box = gr.Textbox(label="Error", visible=False)
    
    # 输入框
    name_box = gr.Textbox(label="Name")
    age_box = gr.Number(label="Age")   
    symptoms_box = gr.CheckboxGroup(["Cough", "Fever", "Runny Nose"])
    submit_btn = gr.Button("Submit")
    
    # 输出不可见
    with gr.Column(visible=False) as output_col:
        diagnosis_box = gr.Textbox(label="Diagnosis")
        patient_summary_box = gr.Textbox(label="Patient Summary")
    
    def submit(name, age, symptoms):
        if len(name) == 0:
            return {
                error_box: gr.update(value="Enter name", visible=True),
                output_col: gr.update(visible=False),  # 隐藏输出
                diagnosis_box: gr.update(value=""),  # 清空诊断信息
                patient_summary_box: gr.update(value="")  # 清空摘要
            }
        if age < 0 or age > 200:
            return {
                error_box: gr.update(value="Invalid age", visible=True),
                output_col: gr.update(visible=False),  # 隐藏输出
                diagnosis_box: gr.update(value=""),  # 清空诊断信息
                patient_summary_box: gr.update(value="")  # 清空摘要
            }
        
        # 如果输入正确，显示诊断信息并隐藏错误框
        return {
            error_box: gr.update(visible=False),  # 隐藏错误框
            output_col: gr.update(visible=True),  # 显示输出
            diagnosis_box: "covid" if "Cough" in symptoms else "flu",
            patient_summary_box: f"{name}, {age} y/o"
        }
    submit_btn.click(
        submit,
        [name_box, age_box, symptoms_box],
        [error_box, diagnosis_box, patient_summary_box, output_col],
    )        
demo.launch()        
    
