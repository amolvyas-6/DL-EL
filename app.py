"""Run with: uv run python app.py (train checkpoints first)."""

import time
from contextlib import contextmanager
from pathlib import Path
from statistics import mean

import cv2
import gradio as gr
from PIL import Image, ImageDraw

from pothole.compression import compress_image
from pothole.models import CustomClassifier, Prediction, YoloClassifier
from pothole.severity import SeverityClassifier, SeverityPrediction

YOLO_WEIGHTS = Path("artifacts/yolo/best.pt")
CUSTOM_WEIGHTS = Path("artifacts/custom/best.pt")

THEME = gr.themes.Base(
    primary_hue="orange", neutral_hue="stone",
    font=[gr.themes.GoogleFont("Archivo"), "system-ui", "sans-serif"],
)
CSS = """
.app-title { margin-bottom: 8px; }
.app-title h1 { font-size: 2rem; font-weight: 700; letter-spacing: -0.02em; margin: 0; }
.app-title p { margin: 4px 0 0; color: var(--body-text-color-subdued); }

.readings { display: grid; grid-template-columns: 1fr 1fr; border-top: 2px solid var(--body-text-color); }
.reading { padding: 18px 24px 22px 0; }
.reading + .reading { padding-left: 24px; border-left: 1px solid var(--border-color-primary); }
.reading header { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; }
.reading h3 { margin: 0; font-size: 1rem; font-weight: 600; }
.reading header span { font-size: 0.85rem; color: var(--body-text-color-subdued); }

.verdict { margin: 22px 0 24px; font-size: clamp(2.2rem, 4.5vw, 3.2rem); font-weight: 700;
           line-height: 1; letter-spacing: -0.03em; }
.verdict.pothole { color: #c2410c; }
.verdict.not-pothole { color: #2f6f4f; }
.verdict.non-road { color: #6b7280; }

.figures { display: grid; grid-template-columns: 1fr 1fr; gap: 12px 20px; margin: 0 0 18px; }
.figures dt { font-size: 0.8rem; color: var(--body-text-color-subdued); }
.figures dd { margin: 2px 0 0; font-size: 1.35rem; font-weight: 600; font-variant-numeric: tabular-nums; }
.figures dd small { font-size: 0.8rem; font-weight: 400; color: var(--body-text-color-subdued); }

.meter { height: 4px; border-radius: 2px; background: var(--border-color-primary); overflow: hidden; }
.meter span { display: block; height: 100%; background: var(--color-accent); transition: width 0.4s ease; }

.severity { margin-bottom: 18px; }
.severity p { margin: 0 0 8px; font-size: 0.8rem; color: var(--body-text-color-subdued); }
.steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: 4px; }
.step { padding: 9px 0; text-align: center; font-size: 0.9rem; border-radius: 3px;
        background: var(--background-fill-secondary); color: var(--body-text-color-subdued); }
.step.on.low { background: #ca8a04; color: #fff; font-weight: 600; }
.step.on.medium { background: #ea580c; color: #fff; font-weight: 600; }
.step.on.high { background: #b91c1c; color: #fff; font-weight: 600; }

.summary { display: flex; flex-wrap: wrap; gap: 36px; margin: 22px 0 8px; }
.summary dt { font-size: 0.8rem; color: var(--body-text-color-subdued); }
.summary dd { margin: 2px 0 0; font-size: 1.5rem; font-weight: 600; font-variant-numeric: tabular-nums; }

.caption { display: flex; justify-content: space-between; align-items: baseline; gap: 16px; margin: 4px 0 6px; }
.caption h2 { margin: 0; font-size: 1.1rem; font-weight: 600; word-break: break-all; }
.counter { font-size: 0.9rem; color: var(--body-text-color-subdued); font-variant-numeric: tabular-nums; }
.nav { align-self: center; height: 48px !important; min-width: 48px !important; padding: 0 !important;
       font-size: 1.4rem !important; border-radius: 50% !important; }

.status { display: flex; align-items: center; gap: 14px; margin: 6px 0; font-size: 0.9rem;
          color: var(--body-text-color-subdued); }
.status:empty { display: none; }
.bar { flex: 1; max-width: 320px; height: 4px; border-radius: 2px; overflow: hidden;
       background: var(--border-color-primary); }
.bar div { height: 100%; background: var(--color-accent); transition: width 0.3s ease; }
.bar.busy div { width: 35%; animation: sweep 1.1s ease-in-out infinite; }
@keyframes sweep { from { transform: translateX(-100%); } to { transform: translateX(300%); } }

button:focus-visible { outline: 2px solid var(--color-accent); outline-offset: 2px; }
@media (max-width: 720px) {
  .readings { grid-template-columns: 1fr; }
  .reading + .reading { padding-left: 0; border-left: 0; border-top: 1px solid var(--border-color-primary); }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
}
"""

VERDICTS = {"pothole": "Pothole", "not-pothole": "Not pothole", "non-road": "Non-road"}


# ---------------------------------------------------------------- HTML rendering

def percent(value: float | None) -> str:
    return "-" if value is None else f"{value:.1%}"


def reading(name: str, place: str, label: str, confidence: float | None,
            ms: float, slowest: float, extra: str = "") -> str:
    return (f'<article class="reading"><header><h3>{name}</h3><span>{place}</span></header>'
            f'<p class="verdict {label}">{VERDICTS[label]}</p>'
            f'<dl class="figures"><div><dt>Confidence</dt><dd>{percent(confidence)}</dd></div>'
            f'<div><dt>Latency</dt><dd>{ms:.0f} <small>ms</small></dd></div></dl>'
            f'<div class="meter" title="Latency relative to the slower model">'
            f'<span style="width: {ms / slowest:.0%}"></span></div>{extra}</article>')


def comparison_html(yolo: Prediction, custom: Prediction) -> str:
    slowest = max(yolo.milliseconds, custom.milliseconds) or 1.0
    return ('<div class="readings">'
            + reading("YOLO11s", "", yolo.label, yolo.confidence, yolo.milliseconds, slowest)
            + reading("Custom", "", custom.label, custom.confidence, custom.milliseconds, slowest)
            + "</div>")


def severity_html(result: SeverityPrediction) -> str:
    steps = "".join(f'<span class="step {level} {"on" if level == result.severity else ""}">'
                    f'{level.title()}</span>' for level in ("low", "medium", "high"))
    scale = (f'<div class="severity"><p>Severity · {percent(result.severity_confidence)}</p>'
             f'<div class="steps">{steps}</div></div>'
             if result.severity else '<div class="severity"><p>Severity</p><b>None</b></div>')
    return ('<div class="readings">'
            + reading("Custom + severity", "", result.label, result.confidence,
                      result.milliseconds, result.milliseconds, extra=scale)
            + "</div>")


def summary_html(count: int, latencies: dict[str, list[float]]) -> str:
    figures = [("Images", str(count))] + [(f"{name} mean", f"{mean(values):.0f} ms")
                                          for name, values in latencies.items() if values]
    return '<dl class="summary">' + "".join(f"<div><dt>{k}</dt><dd>{v}</dd></div>" for k, v in figures) + "</dl>"


def status(text: str, fraction: float | None = None) -> str:
    """The app's single loader: a label and a bar that is determinate when `fraction` is given."""
    if fraction is None:
        bar = '<div class="bar busy"><div></div></div>'
    else:
        bar = f'<div class="bar"><div style="width: {fraction:.0%}"></div></div>'
    return f'<div class="status"><span>{text}</span>{bar}</div>'


def draw_boxes(image: Image.Image, boxes) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for x1, y1, x2, y2, confidence in boxes:
        draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
        draw.text((x1, max(y1 - 15, 0)), f"pothole {confidence:.2f}", fill="red")
    return annotated


# ---------------------------------------------------------------- video and errors

@contextmanager
def friendly_errors():
    """Show any failure as a Gradio error message instead of a bare 'Error'."""
    try:
        yield
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def realtime_frames(path: str):
    """Yield (frame number, RGB frame) at the video's own pace, skipping frames when inference can't keep up."""
    capture = cv2.VideoCapture(path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not capture.isOpened() or fps <= 0:
        capture.release()
        raise gr.Error("Could not read the uploaded video.")
    start, index = time.perf_counter(), -1
    try:
        while True:
            due = int((time.perf_counter() - start) * fps)
            ok = capture.grab()
            index += 1
            while ok and index < due:  # behind schedule: drop frames
                ok = capture.grab()
                index += 1
            if not ok:
                return
            ok, frame = capture.retrieve()
            if not ok:
                return
            time.sleep(max(0.0, index / fps - (time.perf_counter() - start)))  # ahead: wait
            yield index, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()


# ---------------------------------------------------------------- batch tab

def show(batch: list, index: int):
    """Return (index, picture, html) for item `index`, wrapping around at both ends."""
    if not batch:
        return 0, None, ""
    index %= len(batch)
    picture, title, html = batch[index]
    return index, picture, (f'<div class="caption"><h2>{title}</h2>'
                            f'<span class="counter">{index + 1} / {len(batch)}</span></div>{html}')


def batch_tab(names: list[str], classify):
    """Upload images, classify each with `classify(image, name) -> (picture, title, html, latencies)`, and browse.

    Each uploaded image is compressed first and `classify` receives the compressed image.

    `latencies` holds one millisecond value per name in `names`, used for the summary.
    """
    files = gr.File(file_count="multiple", file_types=["image"], label="Images")
    button = gr.Button("Classify", variant="primary")
    loader = gr.HTML()
    summary_view = gr.HTML()
    batch, index = gr.State([]), gr.State(0)
    with gr.Row(equal_height=True):
        back = gr.Button("‹", scale=0, min_width=48, elem_classes="nav")
        picture = gr.Image(show_label=False, interactive=False, height=440, scale=12)
        forward = gr.Button("›", scale=0, min_width=48, elem_classes="nav")
    details = gr.HTML()
    view = [index, picture, details]

    def run(paths):
        if not paths:
            raise gr.Error("Upload at least one image first.")
        items, latencies = [], {name: [] for name in names}
        with friendly_errors():
            for number, path in enumerate(paths, start=1):
                yield (status(f"Classifying image {number} of {len(paths)}", (number - 1) / len(paths)),
                       gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip())
                picture_out, title, html, times = classify(compress_image(path), Path(path).name)
                items.append((picture_out, title, html))
                for name, ms in zip(names, times):
                    latencies[name].append(ms)
        yield "", items, summary_html(len(items), latencies), *show(items, 0)

    button.click(run, files, [loader, batch, summary_view, *view], show_progress="hidden")
    back.click(lambda i, items: show(items, i - 1), [index, batch], view, show_progress="hidden")
    forward.click(lambda i, items: show(items, i + 1), [index, batch], view, show_progress="hidden")


# ---------------------------------------------------------------- application

class PotholeApp:
    def __init__(self):
        self.yolo = None
        self.custom = None
        self.severity = SeverityClassifier()

    def _comparison_models(self):
        if self.yolo is None:
            yolo, custom = YoloClassifier(YOLO_WEIGHTS), CustomClassifier(CUSTOM_WEIGHTS)
            custom.warm_up() 
            yolo.predict(Image.new("RGB", (320, 320)))
            self.yolo, self.custom = yolo, custom
        return self.yolo, self.custom

    def compare_image(self, image: Image.Image, name: str):
        yolo, custom = self._comparison_models()
        a, b = yolo.predict(image), custom.predict(image)
        return (draw_boxes(image, a.boxes), name, comparison_html(a, b),
                [a.milliseconds, b.milliseconds])

    def severity_image(self, image: Image.Image, name: str):
        result = self.severity.predict(image)
        return image, name, severity_html(result), [result.milliseconds]

    def compare_video(self, path: str):
        if not path:
            raise gr.Error("Upload a video first.")
        yield None, "", status("Loading models…")
        with friendly_errors():
            yolo, custom = self._comparison_models()
            for index, frame in realtime_frames(path):
                image = Image.fromarray(frame)
                image.thumbnail((640, 640))
                a, b = yolo.predict(image), custom.predict(image)
                yield (draw_boxes(image, a.boxes), comparison_html(a, b),
                       status(f"Live · frame {index} · YOLO {a.milliseconds:.0f} ms · "
                              f"Custom {b.milliseconds:.0f} ms"))
        yield None, "", ""

    def build(self):
        self.severity.warm_up()  # connect to Cloudflare now, not on the first uploaded image
        with gr.Blocks(title="Pothole inspection") as demo:
            gr.Markdown("# Pothole inspection\nA YOLO model and Custom Classifier, side by side on road photos.",
                        elem_classes="app-title")
            with gr.Tab("Images · YOLO vs Custom"):
                batch_tab(["YOLO", "Custom"], self.compare_image)
            with gr.Tab("Video · YOLO vs Custom"):
                video = gr.Video(label="Video")
                button = gr.Button("Classify", variant="primary")
                loader = gr.HTML()
                with gr.Row(equal_height=True):
                    frame = gr.Image(label="Current frame", interactive=False, scale=3)
                    current = gr.HTML(scale=2)
                button.click(self.compare_video, video, [frame, current, loader], show_progress="hidden")
            with gr.Tab("Images · Custom with severity"):
                batch_tab(["Severity"], self.severity_image)
        return demo


if __name__ == "__main__":
    PotholeApp().build().launch(theme=THEME, css=CSS)
