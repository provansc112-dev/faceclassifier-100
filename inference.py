import torch
from facenet_pytorch import MTCNN
from PIL import Image
import config
import utils
import model

mtcnn = None
MODELS_CACHE = {}

def get_mtcnn():
    global mtcnn
    if mtcnn is None:
        mtcnn = MTCNN(
            image_size=224, 
            margin=20, 
            device=config.DEVICE, 
            post_process=False,
            thresholds=config.MTCNN_THRESHOLDS
        )
    return mtcnn

def _resize_if_huge(img, max_size=1024):
    if img is None:
        return None
    w, h = img.size
    if w > max_size or h > max_size:
        ratio = min(max_size/w, max_size/h)
        new_size = (int(w*ratio), int(h*ratio))
        return img.resize(new_size, Image.Resampling.LANCZOS)
    return img

def format_prediction_text(result):
    if result['probs'] is None:
        return (
            f"STATUS          : {result['status']}\n"
            f"ERROR           : {result['label']}\n"
            f"To resolve, please try another image with a clear frontal face."
        )
    
    text = (
        f"MODEL           : {result['model_type'].upper()}\n"
        f"THRESHOLD       : {result['threshold']}\n"
        f"PREDICTION      : {result['label']}\n"
        f"CONFIDENCE      : {result['confidence']*100:.2f}%\n"
        f"STATUS          : {result['status']}\n\n"
        f"Top Candidates Analysis:\n"
    )
    
    num_candidates = min(3, len(config.CLASS_NAMES))
    top_conf, top_idx = torch.topk(result['probs'], num_candidates)
    
    for i in range(num_candidates):
        name = config.CLASS_NAMES[top_idx[i].item()]
        percentage = top_conf[i].item() * 100
        text += f" {i+1}. {name:15} | {percentage:.2f}%\n"
        
    return text

def predict_logic(img_obj, model_type='swin'):
    if not isinstance(img_obj, Image.Image):
        try:
            img_obj = Image.fromarray(img_obj.astype('uint8')).convert('RGB')
        except Exception as e:
            print(f"Conversion Error: {e}")
    
    if img_obj.size != (224, 224):
        img_obj = _resize_if_huge(img_obj)
    
    if img_obj.size == (224, 224):
        face_img = img_obj
    else:
        face = get_mtcnn()(img_obj)
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
    
    global MODELS_CACHE
    if model_type not in MODELS_CACHE:
        MODELS_CACHE[model_type] = model.load_trained_model(model_type)
    
    net = MODELS_CACHE[model_type]
    transform = utils.get_transforms(model_type, is_train=False)
    img_input = transform(face_img).unsqueeze(0).to(config.DEVICE)
    
    target_threshold = config.THRESHOLD_SWIN if model_type == 'swin' else config.THRESHOLD_INCEPTION

    net.eval()
    with torch.no_grad():
        res = net(img_input)
        outputs = res[2] if isinstance(res, (tuple, list)) else res
        probs = torch.nn.functional.softmax(outputs.squeeze(), dim=0).cpu()
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
    img = Image.open(image_path).convert('RGB')
    
    if do_stress_test:
        img = utils.apply_custom_augmentation(img, **kwargs)

    result = predict_logic(img, model_type)
    print(format_prediction_text(result))
    return result

if __name__ == "__main__":
    print("Use predict_logic() or predict_image_file() functions to run inference.")