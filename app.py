import gradio as gr
import pandas as pd
from ultralytics import YOLO

model = YOLO("best.pt")



ACCENT = "#F97316"      
BG     = "#09090B"  
CARD   = "#18181B"      
BORDER = "#27272A"      

theme = gr.themes.Base(
    primary_hue="orange",
    neutral_hue="zinc",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
    radius_size="lg",
).set(
    body_background_fill_dark=BG,
    block_background_fill_dark=CARD,
    block_border_color_dark=BORDER,
    block_label_background_fill_dark=CARD,
    input_background_fill_dark="#27272A",
    button_primary_background_fill_dark=ACCENT,
    button_primary_background_fill_hover_dark="#FB923C",
    button_primary_text_color_dark="#FFFFFF",
    slider_color_dark=ACCENT,
)


css = f"""
.gradio-container {{ max-width: 1100px !important; margin: auto; }}
footer {{ display: none !important; }}

#title {{ text-align: center; padding: 28px 0 8px; }}
#title h1 {{ font-size: 2.6rem; font-weight: 800; margin: 0; color: #FAFAFA; }}
#title p  {{ color: #A1A1AA; margin-top: 8px; font-size: 1.05rem; }}


.gr-block, .block {{ border-radius: 16px !important; }}


#run-btn {{ height: 52px; font-size: 1.1rem; font-weight: 700; }}


.summary {{
    background: {CARD}; border: 1px solid {BORDER}; border-radius: 16px;
    padding: 20px; text-align: center;
}}
.summary .num  {{ font-size: 3.2rem; font-weight: 800; color: {ACCENT}; line-height: 1; }}
.summary .text {{ color: #A1A1AA; margin-top: 6px; }}
.summary.ok .num {{ color: #22C55E; }}
"""

HEADER = """
<div id="title">
  <h1>🛣️ Road Damage Detector</h1>
  <p>Upload a road photo and the model will find cracks, potholes and other damage.</p>
</div>
"""

def summary_html(num, text, ok=False):
    cls = "summary ok" if ok else "summary"
    return f'<div class="{cls}"><div class="num">{num}</div><div class="text">{text}</div></div>'



def detect(image, conf):
    if image is None:
        return None, summary_html("–", "Upload an image first"), pd.DataFrame(columns=["Damage type", "Confidence"])

    result = model.predict(image[..., ::-1], conf=conf, verbose=False)[0]
    annotated = result.plot()[..., ::-1]

    rows = [{"Damage type": model.names[int(c)], "Confidence": round(float(s), 3)}
            for c, s in zip(result.boxes.cls, result.boxes.conf)]
    table = pd.DataFrame(rows, columns=["Damage type", "Confidence"])

    if table.empty:
        summary = summary_html("0", "No road damage detected", ok=True)
    else:
        summary = summary_html(len(table), "damage(s) detected")
    return annotated, summary, table



with gr.Blocks(theme=theme, css=css, title="Road Damage Detector",
               js="() => document.body.classList.add('dark')") as demo:

    gr.HTML(HEADER)

    with gr.Row():
        image_in  = gr.Image(type="numpy", label="Road image", height=380)
        image_out = gr.Image(label="Result", height=380)

    with gr.Row():
        with gr.Column(scale=2):
            conf = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Confidence threshold")
            btn  = gr.Button("Detect damage", variant="primary", elem_id="run-btn")
        with gr.Column(scale=1):
            summary = gr.HTML(summary_html("–", "Waiting for an image"))

    table = gr.Dataframe(label="Detections")

    btn.click(detect, [image_in, conf], [image_out, summary, table])

demo.launch()
