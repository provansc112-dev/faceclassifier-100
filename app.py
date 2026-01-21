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
import json

warnings.filterwarnings("ignore", category=FutureWarning)

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

print("\n" + "="*50)
print("DEBUGGING SYSTEM PATHS & FILES")
current_dir = os.getcwd()
print(f"Current Working Directory: {current_dir}")
print(f"User ID: {os.getuid()} Group ID: {os.getgid()}")

base_path = os.getcwd()
contents = os.listdir(base_path)
print(f"Root Folder Contents: {contents}")

config_sample_path = getattr(config, 'SAMPLE_FOLDER', None)

candidates = ["samples", "faceclassifier-100", os.path.join(current_dir, "faceclassifier-100")]
if config_sample_path:
    candidates.insert(0, config_sample_path)

SAMPLE_PATH = None

print(f"DEBUG: Searching for sample folder in: {candidates}")

for path in candidates:
    if os.path.exists(path) and os.path.isdir(path):
        sub = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        if sub:
            SAMPLE_PATH = path
            print(f"DEBUG: FOUND VALID SAMPLE FOLDER AT: {SAMPLE_PATH}")
            break

if SAMPLE_PATH is None:
    SAMPLE_PATH = "samples" 
    print(f"CRITICAL: No valid sample folder found. Defaulting to '{SAMPLE_PATH}'")

models_path = "models" 
print(f"Checking Models Path: {os.path.abspath(models_path)}")
if os.path.exists(models_path):
    print(f"Models Folder Exists. Contents: {os.listdir(models_path)}")
else:
    print(f"WARNING: Models folder not found at {models_path}")

print("="*50 + "\n")

def get_folders():
    folders = []
    if os.path.exists(SAMPLE_PATH):
        folders = sorted([d for d in os.listdir(SAMPLE_PATH) 
                       if os.path.isdir(os.path.join(SAMPLE_PATH, d))])
    # Add 'None' option for out-of-distribution cases
    return ["None"] + folders

all_folders = get_folders()

def load_images_from_folder(folder_name):
    if not folder_name or folder_name == "None":
        return []
    path = os.path.join(SAMPLE_PATH, folder_name)
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(path, ext)))
    return sorted(images)

def process_ui(input_img, model_choice, do_flip, bright, cont, erase, mode, face_state, preview_img):
    if input_img is None:
        return None, "WARNING: Please upload an image first.", gr.update(visible=False), "{}"

    m_type = 'swin' if model_choice == "SwinTransformer" else 'inception'
    
    if mode == "NORMAL":
        aug_img = input_img
        header = "### NORMAL TEST\n"
    else:
        if preview_img is not None:
            aug_img = preview_img
        else:
            if face_state is not None:
                face_tensor = face_state
                try:
                    face_img_np = face_tensor.permute(1, 2, 0).byte().cpu().numpy()
                    face_img = Image.fromarray(face_img_np)
                except Exception:
                    face_tensor = inference.get_mtcnn()(input_img)
                    if face_tensor is None:
                         return None, "WARNING: Face state error. Please re-upload.", gr.update(visible=False), "{}"
                    face_img_np = face_tensor.permute(1, 2, 0).byte().cpu().numpy()
                    face_img = Image.fromarray(face_img_np)
            else:
                face_tensor = inference.get_mtcnn()(input_img)
                if face_tensor is None:
                    return None, "WARNING: No face detected. Please upload a clear face image.", gr.update(visible=False), "{}"
                face_img_np = face_tensor.permute(1, 2, 0).byte().cpu().numpy()
                face_img = Image.fromarray(face_img_np)
            
            aug_img = utils.apply_custom_augmentation(
                face_img,
                do_flip=do_flip,
                brightness=bright,
                contrast=cont,
                erase_prob=erase)
            
        header = "### STRESS TEST (GENERALIZATION)\n"
    
    try:
        result = inference.predict_logic(aug_img, model_type=m_type)
        display_img = result.get('face_img') if result.get('face_img') is not None else aug_img
        prediction_info = inference.format_prediction_text(result)
        
        result_clean = {k: v for k, v in result.items() if k != 'face_img'}
        result_json = json.dumps(result_clean, default=str)
        
        return display_img, header + "```\n" + prediction_info + "```", gr.update(visible=True), result_json
    except Exception as e:
        print(f"Inference Error: {e}")
        return None, f"Inference Error: {str(e)}", gr.update(visible=False), "{}"

def load_selected_img(evt: gr.SelectData):
    img_path = evt.value['image']['path']
    return Image.open(img_path).convert("RGB")

def preview_augmentation(input_img, do_flip, bright, cont, erase):
    if input_img is None:
        return None, None
    
    try:
        face_tensor = inference.get_mtcnn()(input_img)
        if face_tensor is None:
            return None, None
        
        face_img_np = face_tensor.permute(1, 2, 0).byte().cpu().numpy()
        face_img = Image.fromarray(face_img_np)
        
        aug_img = utils.apply_custom_augmentation(
            face_img,
            do_flip=do_flip,
            brightness=bright,
            contrast=cont,
            erase_prob=erase)
        
        return aug_img, face_tensor
    except Exception as e:
        print(f"Preview Error: {e}")
        return None, None

def toggle_actual_class(choice):
    if choice == "WRONG":
        return gr.update(visible=True, value=None)
    else:
        return gr.update(visible=False, value=None)

def save_feedback(res_state_json, is_corr, act_class, comment, flip, bright, cont, erase):
    if not res_state_json:
        return gr.update(value="Error: Run a test first.", visible=True)
    
    try:
        res_state = json.loads(res_state_json)
    except:
        res_state = {}
    
    if not is_corr:
        return gr.update(value="Please select CORRECT or WRONG first!", visible=True)
    
    log_file = "prediction_logs.csv"
    
    new_data = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Model": res_state.get('model_type'),
        "Prediction": res_state.get('label'),
        "Confidence": f"{res_state.get('confidence', 0)*100:.2f}%",
        "Status": res_state.get('status'),
        "User Feedback": is_corr,
        "Actual Class": act_class if is_corr == "WRONG" else res_state.get('label'),
        "User Comment": comment,
        "Flip": flip, "Brightness": bright, "Contrast": cont, "Random Erase": erase
    }
    
    try:
        df = pd.DataFrame([new_data])
        df.to_csv(log_file, mode='a', header=not os.path.exists(log_file), index=False)
        return "Log saved to prediction_logs.csv", None, None, ""
    except Exception as e:
        return f"Error saving log: {e}", None, None, ""

with gr.Blocks(theme=gr.themes.Soft(), title="Face Recognition System") as demo:
    last_result = gr.State("")
    detected_face = gr.State(None)

    gr.Markdown("## Face Identification Demo with Swin Transformer & Inception Resnet V1")
    gr.Markdown("**Take a photo from the sample gallery or upload your own image.**")
    
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                folder_select = gr.Dropdown(choices=all_folders, label="Choose Folder (Class)", interactive=True)
                sample_gallery = gr.Gallery(show_label=False, columns=5, height=150, allow_preview=False)

            model_dropdown = gr.Dropdown(choices=["SwinTransformer", "InceptionResNetV1"], value="SwinTransformer", label="Architecture Selection")

            with gr.Accordion("Stress Test Config", open=False):
                flip_check = gr.Checkbox(label="Horizontal Flip")
                bright_slider = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Brightness")
                cont_slider = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Contrast")
                erase_slider = gr.Slider(0, 0.2, value=0, step=0.01, label="Random Erasing Area")

            with gr.Row():
                btn_normal = gr.Button("NORMAL TEST", variant="secondary")
                btn_stress = gr.Button("STRESS TEST", variant="primary")

        with gr.Column(scale=1):
            with gr.Row():
                input_image = gr.Image(type="pil", label="Input Image", scale=1)
                preview_output = gr.Image(label="Augmentation Preview", type="pil", scale=1, visible=False)
            
            with gr.Row():
                output_img = gr.Image(label="Processed Image Result", scale=1)
            
            output_info = gr.Markdown("Waiting for input...")

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
                
                user_comment = gr.Textbox(label="User Comments", placeholder="Comments")
                btn_save = gr.Button("Send Feedback", variant="primary")
                save_status = gr.Markdown("")

    folder_select.change(fn=load_images_from_folder, inputs=folder_select, outputs=sample_gallery, api_name=False)
    sample_gallery.select(fn=load_selected_img, outputs=input_image, api_name=False)
    is_correct.change(
        fn=toggle_actual_class, 
        inputs=[is_correct], 
        outputs=[actual_class],
        api_name=False
    )
    
    preview_params = [input_image, flip_check, bright_slider, cont_slider, erase_slider]
    for param in preview_params:
        param.change(
            fn=preview_augmentation,
            inputs=preview_params,
            outputs=[preview_output, detected_face],
            api_name=False
        ).then(fn=lambda img: gr.update(visible=True) if img is not None else gr.update(), inputs=preview_output, outputs=preview_output, api_name=False)

    predict_inputs_stress = [input_image, model_dropdown, flip_check, bright_slider, cont_slider, erase_slider, detected_face, preview_output]
    predict_inputs_normal = [input_image, model_dropdown, flip_check, bright_slider, cont_slider, erase_slider, detected_face, gr.State(None)]
    
    predict_outputs = [output_img, output_info, feedback_area, last_result]
    
    btn_normal.click(
        fn=lambda i, m, f, b, c, e, d, p: process_ui(i, m, f, b, c, e, "NORMAL", d, p),
        inputs=predict_inputs_normal,
        outputs=predict_outputs,
        api_name=False
    )
    
    btn_stress.click(
        fn=lambda i, m, f, b, c, e, d, p: process_ui(i, m, f, b, c, e, "STRESS", d, p),
        inputs=predict_inputs_stress,
        outputs=predict_outputs,
        api_name=False
    )

    btn_save.click(
        fn=save_feedback,
        inputs=[last_result, is_correct, actual_class, user_comment, flip_check, bright_slider, cont_slider, erase_slider],
        outputs=[save_status, is_correct, actual_class, user_comment],
        api_name=False
    )

if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False
    )