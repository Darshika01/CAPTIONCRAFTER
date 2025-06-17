from flask import Flask, render_template, request, send_file, session, redirect, url_for
from werkzeug.utils import secure_filename
import os
import torch
import io
from PIL import Image, ImageStat
import numpy as np
from collections import Counter
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
import json
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoImageProcessor, AutoModelForImageClassification

app = Flask(__name__)
app.secret_key = "supersecretkey"

UPLOAD_FOLDER = "static/uploads"
FEATURE_FOLDER = "static/features"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FEATURE_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

app.json_encoder = NumpyEncoder

# Load models
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

processor_ai_real = AutoImageProcessor.from_pretrained("dima806/ai_vs_real_image_detection")
model_ai_real = AutoModelForImageClassification.from_pretrained("dima806/ai_vs_real_image_detection").to(device)

def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

def save_feature_maps(image_path, save_folder=FEATURE_FOLDER, num_maps=16):
    model_resnet = models.resnet18(pretrained=True)
    model_resnet.eval()
    activation = {}

    def hook_fn(module, input, output):
        activation['feature_maps'] = output.detach()

    hook = model_resnet.conv1.register_forward_hook(hook_fn)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        model_resnet(img_tensor)

    hook.remove()
    feature_maps = activation['feature_maps'].squeeze()
    saved_paths = []

    for i in range(min(num_maps, feature_maps.shape[0])):
        fmap = feature_maps[i].cpu()
        plt.imshow(fmap, cmap='viridis')
        plt.axis('off')
        file_path = os.path.join(save_folder, f"fmap_{i}.png")
        plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        saved_paths.append("/" + file_path.replace("\\", "/"))

    return saved_paths

def predict_ai_real(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor_ai_real(images=image, return_tensors="pt").to(device)

    model_ai_real.eval()
    with torch.no_grad():
        outputs = model_ai_real(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()

    label_map = model_ai_real.config.id2label
    label = label_map.get(predicted_class, "Unknown").lower()

    # Only return label
    if any(word in label for word in ["ai", "fake", "generated", "synth"]):
        return "AI Generated"
    else:
        return "Real Image"

def analyze_image(image_path):
    image = Image.open(image_path)
    image_array = np.array(image)

    results = {
        "filename": os.path.basename(image_path),
        "dimensions": f"{image.width} x {image.height}",
        "color_type": image.mode,
        "channels": len(image.getbands()),
        "file_size": f"{os.path.getsize(image_path) / 1024:.2f} KB",
    }

    stat = ImageStat.Stat(image.convert("L"))
    results["brightness"] = round(stat.mean[0], 2)
    results["contrast"] = round(stat.stddev[0], 2)
    gray = image.convert("L")
    gray_array = np.array(gray)
    results["sharpness"] = round(np.var(gray_array), 2)

    pixels = image_array.reshape(-1, image_array.shape[-1])
    pixel_list = [tuple(int(value) for value in pixel) for pixel in pixels]
    most_common_colors = Counter(pixel_list).most_common(5)
    results["dominant_colors"] = [color[0] for color in most_common_colors]

    try:
        results["caption"] = generate_caption(image_path)
    except Exception as e:
        results["caption"] = f"Caption generation failed: {str(e)}"

    try:
        results["ai_detection"] = predict_ai_real(image_path)
    except Exception as e:
        results["ai_detection"] = f"Detection failed: {str(e)}"

    results["feature_maps"] = save_feature_maps(image_path)
    return results

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"

        file = request.files["file"]
        if file.filename == "":
            return "No selected file"

        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        results = analyze_image(file_path)
        session["results"] = results
        session["file_path"] = file_path

        return redirect(url_for("results"))

    return render_template("index.html", results=None)

@app.route("/results")
def results():
    results = session.get("results")
    file_path = session.get("file_path")
    if not results or not file_path:
        return redirect(url_for("home"))

    relative_path = "/" + file_path if not file_path.startswith("/") else file_path
    return render_template("result.html", results=results, file_path=relative_path)

@app.route("/download_report")
def download_report():
    results = session.get("results")
    if not results:
        return "No results to download"

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)

    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(100, 750, "AI Image Analysis Report")

    pdf.setFont("Helvetica", 12)
    pdf.drawString(100, 730, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    pdf.line(100, 720, 500, 720)

    y = 700
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(100, y, "AI Caption:")
    y -= 20
    pdf.setFont("Helvetica", 12)
    caption = results.get("caption", "")
    lines = caption.split()
    line_str = ""
    for word in lines:
        if pdf.stringWidth(line_str + word + " ", "Helvetica", 12) < 400:
            line_str += word + " "
        else:
            pdf.drawString(120, y, line_str)
            y -= 20
            line_str = word + " "
    if line_str:
        pdf.drawString(120, y, line_str)
        y -= 20

    y -= 10
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(100, y, "Image Properties:")
    y -= 20

    for label, val in [
        ("Filename", results["filename"]),
        ("Dimensions", results["dimensions"]),
        ("Color Type", results["color_type"]),
        ("Channels", results["channels"]),
        ("File Size", results["file_size"]),
        ("Brightness", results["brightness"]),
        ("Contrast", results["contrast"]),
        ("Sharpness", results["sharpness"]),
        ("AI Detection", results["ai_detection"]),
    ]:
        pdf.setFont("Helvetica", 12)
        pdf.drawString(120, y, f"{label}: {val}")
        y -= 20

    y -= 10
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(100, y, "Top 5 Dominant Colors:")
    y -= 20
    pdf.setFont("Helvetica", 12)
    for color in results["dominant_colors"]:
        pdf.drawString(120, y, f"RGB: {color}")
        y -= 20

    pdf.setFont("Helvetica", 10)
    pdf.setFillColor(colors.grey)
    pdf.drawString(100, 50, "Generated by AI Image Analyzer")

    pdf.save()
    buffer.seek(0)

    return send_file(buffer, as_attachment=True, download_name="AI_Image_Report.pdf", mimetype="application/pdf")

if __name__ == "__main__":
    app.run(debug=True, port=10000)
