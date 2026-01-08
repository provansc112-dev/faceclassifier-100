import torch
from facenet_pytorch import MTCNN
from PIL import Image
import config
import utils
import model

mtcnn = MTCNN(image_size=224, margin=20, device=config.DEVICE, post_process=False)

def format_prediction_text(result):
    if result['probs'] is None:
        return (
            f"STATUS         : {result['status']}\n"
            f"ERROR          : {result['label']}\n"
            f"To resolve, please try another image with a clear frontal face."
        )
    
    text = (
        f"MODEL          : {result['model_type'].upper()}\n"
        f"THRESHOLD      : {result['threshold']}\n"
        f"PREDICTION     : {result['label']}\n"
        f"CONFIDENCE     : {result['confidence']*100:.2f}%\n"
        f"STATUS         : {result['status']}\n\n"
        f"Top 3 Candidates Analysis:\n"
    )
    
    top3_conf, top3_idx = torch.topk(result['probs'], 3)
    for i in range(3):
        name = config.CLASS_NAMES[top3_idx[i].item()]
        percentage = top3_conf[i].item() * 100
        text += f" {i+1}. {name:15} | {percentage:.2f}%\n"
        
    return text

def predict_logic(img_obj, model_type='swin'):
    """
    Main inference logic for a single image object and return the prediction result.
    Web or local usage can call this function.   
    """

    # Face Detection
    face = mtcnn(img_obj)
    if face is None:
        return {
            "label": "No Face Detected",
            "confidence": 0,
            "status": "UNRECOGNIZED",
            "probs": None,
            "threshold": 0,
            "model_type": model_type,
            "face_img": None
        }
    face_img_np = face.permute(1, 2, 0).byte().cpu().numpy()
    face_img = Image.fromarray(face_img_np)
    
    # Load Model
    net = model.load_trained_model(model_type)

    # Transform
    transform = utils.get_transforms(model_type, is_train=False)
    
    # Preprocess
    img_input = transform(face_img).unsqueeze(0).to(config.DEVICE)
    
    # Threshold
    target_threshold = config.THRESHOLD_SWIN if model_type == 'swin' else config.THRESHOLD_INCEPTION

    # Inference
    net.eval()
    with torch.no_grad():
        res = net(img_input)
        outputs = res[2] if isinstance(res, (tuple, list)) else res
        probs = torch.nn.functional.softmax(outputs.squeeze(), dim=0)
        conf, pred_idx = torch.max(probs, 0)

    confidence_score = conf.item()
    class_name = config.CLASS_NAMES[pred_idx.item()] if confidence_score >= target_threshold else "Unknown"
    status = "VERIFIED" if confidence_score >= target_threshold else "UNRECOGNIZED"

    return {
        "label": class_name,
        "confidence": confidence_score,
        "status": status,
        "probs": probs,
        "threshold": target_threshold,
        "model_type": model_type,
        "face_img": face_img
    }

def predict_image_file(image_path, model_type='swin', do_stress_test=False, **kwargs):
    """
    This function is for local/terminal usage (based on file path).
    """
    print(f"Starts prediction using {model_type.upper()} ---")
    img = Image.open(image_path).convert('RGB')
    
    # Optional Stress Test with Custom Augmentation
    if do_stress_test:
        img = utils.apply_custom_augmentation(img, **kwargs)
        img.show()

    result = predict_logic(img, model_type)
    
    print("-" * 30)
    print(format_prediction_text(result))
    print("-" * 30)
    
    return result

if __name__ == "__main__":
    # EXAMPLE 1: Normal Prediction
    predict_image_file('089_61.jpg', model_type='swin')

    # EXAMPLE 2: Stress Test Prediction with Custom Augmentation
    # predict_image_file('089_61.jpg', model_type='swin', 
    #                    do_stress_test=True, do_flip=True, 
    #                    brightness=1.5, contrast=1.2, erase_prob=0.1)