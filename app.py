import os
import glob
import pandas as pd
from datetime import datetime
import gradio as gr
import inference
import utils
import config
from PIL import Image
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

# Prepare Examples List
sample_folder = config.SAMPLE_FOLDER
all_folders = sorted([d for d in os.listdir(sample_folder) if os.path.isdir(os.path.join(sample_folder, d))]) if os.path.exists(sample_folder) else []

def load_images_from_folder(folder_name):
    path = os.path.join(config.SAMPLE_FOLDER, folder_name)
    images = glob.glob(os.path.join(path, "*.[jJ][pP][gG]")) + glob.glob(os.path.join(path, "*.[pP][nN][gG]"))
    return images

def process_ui(input_img, model_choice, do_flip, bright, cont, erase, mode):
    if input_img is None:
        return None, "WARNING: Please upload an image first.", gr.update(visible=False), None

    m_type = 'swin' if model_choice == "SwinTransformer" else 'inception'
    
    if mode == "NORMAL":
        aug_img = input_img 
        header = "### NORMAL TEST\n"
    else:
        aug_img = utils.apply_custom_augmentation(
            input_img,
            do_flip=do_flip,
            brightness=bright,
            contrast=cont,
            erase_prob=erase)
        header = "### STRESS TEST (GENERALIZATION)\n"
    
    result = inference.predict_logic(aug_img, model_type=m_type)
    display_img = result.get('face_img') if result.get('face_img') is not None else aug_img
    prediction_info = inference.format_prediction_text(result)
    
    return display_img, header + "```\n" + prediction_info + "```", gr.update(visible=True), result

def load_selected_img(evt: gr.SelectData):
        img_path = evt.value['image']['path']
        return Image.open(img_path).convert("RGB")

def toggle_actual_class(choice):
        if choice == "WRONG":
            return gr.update(visible=True, value=None)
        else:
            return gr.update(visible=False, value=None)

def save_feedback(res_state, is_corr, act_class, comment, flip, bright, cont, erase):
    if res_state is None:
        return gr.update(value="Error: Run a test first.", visible=True)
    
    if not is_corr:
        return gr.update(value="Please select CORRECT or WRONG first!", visible=True)
    
    log_file = "prediction_logs.csv"
    
    top3_text = ""
    if res_state.get('probs') is not None:
        import torch
        top3_conf, top3_idx = torch.topk(res_state['probs'], 3)
        for i in range(3):
            name = config.CLASS_NAMES[top3_idx[i].item()]
            perc = top3_conf[i].item() * 100
            top3_text += f"[{name}: {perc:.2f}%] "

    new_data = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Model": res_state.get('model_type'),
        "Prediction": res_state.get('label'),
        "Confidence": f"{res_state.get('confidence', 0)*100:.2f}%",
        "Status": res_state.get('status'),
        "Top 3 Analysis": top3_text,
        "User Feedback": is_corr,
        "Actual Class": act_class if is_corr == "WRONG" else res_state.get('label'),
        "User Comment": comment,
        "Flip": flip, "Brightness": bright, "Contrast": cont, "Random Erase": erase
    }
    
    df = pd.DataFrame([new_data])
    df.to_csv(log_file, mode='a', header=not os.path.exists(log_file), index=False)
    
    return "Log saved to prediction_logs.csv", None, None, ""

# GRADIO USER INTERFACE
with gr.Blocks(theme=gr.themes.Soft(), title="Face Recognition System") as demo:
    last_result = gr.State(None)

    gr.Markdown("## Face Identification Demo with Swin Transformer & Inception Resnet V1")
    gr.Markdown("**Take a photo from the sample gallery or upload your own image.**")
    
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                folder_select = gr.Dropdown(choices=all_folders, label="Choose Folder (Class)")
                sample_gallery = gr.Gallery(show_label=False, columns=5, height=150)

            model_dropdown = gr.Dropdown(choices=["SwinTransformer", "InceptionResNetV1"], value="SwinTransformer", label="Architecture Selection")

            with gr.Accordion("Stress Test Config", open=False):
                flip_check = gr.Checkbox(label="Horizontal Flip")
                bright_slider = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="Brightness")
                cont_slider = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="Contrast")
                erase_slider = gr.Slider(0, 0.2, value=0, step=0.02, label="Random Erasing Area")

            with gr.Row():
                btn_normal = gr.Button("NORMAL TEST", variant="secondary")
                btn_stress = gr.Button("STRESS TEST", variant="primary")

        with gr.Column(scale=1):
            with gr.Row():
                input_image = gr.Image(type="pil", label="Input Image")
                output_img = gr.Image(label="Processed Image Result")
            
            output_info = gr.Markdown("Waiting for input...")

            # FEEDBACK AREA
            with gr.Group(visible=False) as feedback_area:
                gr.Markdown("### Feedback & Correction")
                with gr.Row():
                    is_correct = gr.Radio(["CORRECT", "WRONG"], label="How was my prediction?")
                    actual_class = gr.Dropdown(
                        choices=all_folders, 
                        label="Actual Class (If WRONG)", 
                        visible=False, 
                        value=None
                    )
                
                user_comment = gr.Textbox(label="User Comments", placeholder="e.g. 'Poor lighting but still identified'")
                btn_save = gr.Button("Send Feedback", variant="primary")
                save_status = gr.Markdown("")

    # Interactions
    folder_select.change(fn=load_images_from_folder, inputs=folder_select, outputs=sample_gallery)
    sample_gallery.select(fn=load_selected_img, outputs=input_image)
    is_correct.change(
        fn=toggle_actual_class, 
        inputs=[is_correct], 
        outputs=[actual_class]
    )

    # Event Handlers
    predict_inputs = [input_image, model_dropdown, flip_check, bright_slider, cont_slider, erase_slider]
    predict_outputs = [output_img, output_info, feedback_area, last_result]
    
    btn_normal.click(
        fn=lambda i, m, f, b, c, e: process_ui(i, m, f, b, c, e, "NORMAL"),
        inputs=[input_image, model_dropdown, flip_check, bright_slider, cont_slider, erase_slider],
        outputs=predict_outputs
    )
    
    btn_stress.click(
        fn=lambda i, m, f, b, c, e: process_ui(i, m, f, b, c, e, "STRESS"),
        inputs=[input_image, model_dropdown, flip_check, bright_slider, cont_slider, erase_slider],
        outputs=predict_outputs
    )

    btn_save.click(
        fn=save_feedback,
        inputs=[last_result, is_correct, actual_class, user_comment, flip_check, bright_slider, cont_slider, erase_slider],
        outputs=[save_status, is_correct, actual_class, user_comment]
    )

if __name__ == "__main__":
    demo.queue().launch(
        # server_name="0.0.0.0",
        # server_port=7860,
        # show_api=False
    )