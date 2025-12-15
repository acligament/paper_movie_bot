import os
import re
import json
import textwrap
from datetime import datetime
from urllib.parse import quote_plus

import requests
import feedparser
import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont
from gtts import gTTS

# moviepy は 1.0.3 固定前提（run.ymlで固定）
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips

# -----------------------------
# Settings
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(SAVE_DIR, exist_ok=True)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_API_BASE = os.environ.get("GEMINI_API_BASE", "https://generativelanguage.googleapis.com").strip()
GEMINI_API_VERSION = os.environ.get("GEMINI_API_VERSION", "v1beta").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()

MAX_PAPERS = int(os.environ.get("MAX_PAPERS", "3"))
SLIDE_SECONDS = float(os.environ.get("SLIDE_SECONDS", "4"))

# IPAフォント（Ubuntu + fonts-ipafont-gothic）
# 環境によってパスが違うことがあるので複数候補
FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
    "/usr/share/fonts/truetype/ipafont-gothic/ipag.ttf",
]

# -----------------------------
# Utils
# -----------------------------
def safe_filename(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\r\n]', "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")[:120]

def pick_font(size: int = 54) -> ImageFont.FreeTypeFont:
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()

def clamp_text(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…"

def keep_only_bullets(text: str) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    bullets = []
    for l in lines:
        if l.startswith(("-", "・", "*")):
            l = l.lstrip("*").strip()
            if l.startswith("・"):
                l = "- " + l[1:].strip()
            elif l.startswith("-"):
                l = "- " + l[1:].strip()
            bullets.append(l)
    return "\n".join(bullets[:3])

# -----------------------------
# 1) arXiv fetch
# -----------------------------
def fetch_arxiv_papers():
    raw_query = "cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:cs.CV OR cat:stat.ML"
    encoded_query = quote_plus(raw_query)

    url = (
        "http://export.arxiv.org/api/query?"
        f"search_query={encoded_query}"
        "&start=0"
        f"&max_results={MAX_PAPERS}"
        "&sortBy=submittedDate"
        "&sortOrder=descending"
    )
    feed = feedparser.parse(url)
    return feed.entries

# -----------------------------
# 2) PDF download
# -----------------------------
def download_pdf(pdf_url: str, filename: str):
    try:
        r = requests.get(pdf_url, timeout=30)
        r.raise_for_status()
    except Exception as e:
        print(f"PDF download failed: {pdf_url}  error={e}")
        return None

    path = os.path.join(SAVE_DIR, filename)
    with open(path, "wb") as f:
        f.write(r.content)
    return path

# -----------------------------
# 3) PDF text extract
# -----------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    if not pdf_path:
        return ""
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"PDF open failed: {pdf_path}  error={e}")
        return ""

    chunks = []
    for page in doc:
        chunks.append(page.get_text())
    return "\n".join(chunks)

def extract_pdf_thumbnail(pdf_path: str, out_png: str) -> str:
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)  # 1ページ目
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 高解像度
    out_path = os.path.join(SAVE_DIR, out_png)
    pix.save(out_path)
    return out_path

# -----------------------------
# 4) Gemini REST (ListModels + fallback)
# -----------------------------

def gemini_list_models():
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is empty. Set it via GitHub Actions Secret.")
    url = f"{GEMINI_API_BASE}/{GEMINI_API_VERSION}/models?key={GEMINI_API_KEY}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get("models", [])

def normalize_model_name(model: str) -> str:
    # accepts: "gemini-xxx" or "models/gemini-xxx"
    model = model.strip()
    if model.startswith("models/"):
        return model
    return f"models/{model}"

def pick_working_model(preferred: str) -> str:
    models = gemini_list_models()
    preferred_norm = normalize_model_name(preferred)

    # まず希望モデルが存在&generateContent対応か
    for m in models:
        if m.get("name") == preferred_norm:
            methods = m.get("supportedGenerationMethods", [])
            if "generateContent" in methods:
                return preferred_norm

    # フォールバック：generateContentできるモデルを上から探す
    for m in models:
        methods = m.get("supportedGenerationMethods", [])
        if "generateContent" in methods:
            return m.get("name")

    raise RuntimeError("No model supports generateContent. Check your API key / project access.")

def gemini_generate_content(prompt: str, model_name: str) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is empty. Set it via GitHub Actions Secret.")

    model_norm = normalize_model_name(model_name)
    url = f"{GEMINI_API_BASE}/{GEMINI_API_VERSION}/{model_norm}:generateContent?key={GEMINI_API_KEY}"

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
        },
    }

    r = requests.post(url, json=payload, timeout=60)
    if r.status_code != 200:
        body = ""
        try:
            body = r.text[:800]
        except Exception:
            body = "<no body>"
        raise RuntimeError(f"Gemini REST error {r.status_code}: {body}")

    data = r.json()
    # candidates[0].content.parts[0].text が基本
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return json.dumps(data)[:800]

def gemini_summarize_ja(paper_text: str, title: str) -> str:
    # 長文対策：先頭だけ
    paper_text = paper_text[:12000]

    prompt = f"""
あなたは日本語が得意なAI研究者です。
次の論文の内容を、日本語で短く要約してください。

出力ルール（厳守）:
- 出力は箇条書きのみ
- 箇条書きは「- 」で始める
- 3点以内
- 各点は最大35文字
- 前置き・挨拶・説明文は禁止

タイトル:
{title}

本文（抜粋）:
{paper_text}
""".strip()

    chosen = pick_working_model(GEMINI_MODEL)
    print(f"🧠 Summarizing by Gemini (REST) using model: {chosen}")

    raw = gemini_generate_content(prompt, chosen).strip()
    clean = keep_only_bullets(raw)

    return clean if clean else raw

# -----------------------------
# 5) TTS (gTTS) + speed up by ffmpeg
# -----------------------------
def generate_tts_mp3(text_ja: str, out_mp3: str) -> str:
    path = os.path.join(SAVE_DIR, out_mp3)
    tts = gTTS(text=text_ja, lang="ja")
    tts.save(path)
    return path

def speedup_audio_ffmpeg(in_path: str, out_path: str, speed: float = 1.15) -> str:
    # ffmpeg atempo は 0.5〜2.0
    speed = max(0.5, min(2.0, speed))
    cmd = f'ffmpeg -y -i "{in_path}" -filter:a "atempo={speed}" "{out_path}"'
    rc = os.system(cmd)
    if rc != 0:
        print("⚠ ffmpeg speed-up failed, keep original audio.")
        return in_path
    return out_path

def generate_slide_audios(slide_scripts):
    audio_files = []
    for i, text in enumerate(slide_scripts, 1):
        mp3 = generate_tts_mp3(text, f"slide_audio_{i:02d}.mp3")
        audio_files.append(mp3)
    return audio_files

# -----------------------------
# 6) Slide text & image
# -----------------------------

def extract_pdf_thumbnail(pdf_path: str, out_png: str) -> str:
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    path = os.path.join(SAVE_DIR, out_png)
    pix.save(path)
    return path


def create_cover_slide(title: str, thumbnail_path: str, out_png: str) -> str:
    W, H = 1920, 1080
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)

    title_font = pick_font(64)

    thumb = Image.open(thumbnail_path).convert("RGB")
    thumb.thumbnail((900, 900))
    img.paste(thumb, (100, 150))

    title_wrapped = "\n".join(textwrap.wrap(title, 28))
    draw.multiline_text(
        (1050, 300),
        title_wrapped,
        fill="black",
        font=title_font,
        spacing=12,
    )

    path = os.path.join(SAVE_DIR, out_png)
    img.save(path)
    return path


def create_slide_image(lines: str, out_png: str) -> str:
    W, H = 1920, 1080
    img = Image.new("RGB", (W, H), color="white")
    draw = ImageDraw.Draw(img)

    title_font = pick_font(64)
    body_font = pick_font(50)

    parts = lines.split("\n", 1)
    title = parts[0].strip()
    body = parts[1].strip() if len(parts) > 1 else ""

    title_wrapped = "\n".join(textwrap.wrap(title, width=22))
    body_wrapped = "\n".join(textwrap.wrap(body, width=34))

    draw.multiline_text((120, 120), title_wrapped, fill="black", font=title_font, spacing=12)
    draw.multiline_text((120, 340), body_wrapped, fill="black", font=body_font, spacing=18)

    path = os.path.join(SAVE_DIR, out_png)
    img.save(path)
    return path

def build_slide_scripts(title: str, summary: str):
    bullets = [s.strip(" -・") for s in summary.split("\n") if s.strip()]
    while len(bullets) < 3:
        bullets.append("")

    return [
        f"{title}。",
        f"ポイント1。{bullets[0]}。",
        f"ポイント2。{bullets[1]}。",
        f"ポイント3。{bullets[2]}。",
        "以上で紹介を終わります。"
    ]

def build_slides(title: str, summary: str, pdf_path: str):
    bullets = [s.strip(" -・") for s in summary.split("\n") if s.strip()]
    while len(bullets) < 3:
        bullets.append("（要点なし）")

    thumb = extract_pdf_thumbnail(pdf_path, "thumbnail.png")

    slides = [
        create_cover_slide(title, thumb, "slide_00_cover.png"),
        create_slide_image(f"POINT 1\n{bullets[0]}", "slide_01.png"),
        create_slide_image(f"POINT 2\n{bullets[1]}", "slide_02.png"),
        create_slide_image(f"POINT 3\n{bullets[2]}", "slide_03.png"),
        create_slide_image("END\nありがとうございました", "slide_04.png"),
    ]

    return slides

# -----------------------------
# 7) Video
# -----------------------------
def generate_video(slide_files, audio_files, out_mp4):
    clips = []

    for img, audio_path in zip(slide_files, audio_files):
        audio = AudioFileClip(audio_path)
        duration = audio.duration + 0.2  # 少し余白

        clip = (
            ImageClip(img)
            .set_duration(duration)
            .set_audio(audio)
        )
        clips.append(clip)

    final = concatenate_videoclips(clips, method="compose")
    out_path = os.path.join(SAVE_DIR, out_mp4)
    final.write_videofile(out_path, fps=24, codec="libx264", audio_codec="aac")
    return out_path

# -----------------------------
# MAIN
# -----------------------------
def main():
    print("📥 Fetching AI papers...")
    papers = fetch_arxiv_papers()
    if not papers:
        print("No papers found.")
        return

    # 先頭の1本を動画化（まず確実に動く最小）
    entry = papers[0]
    title = entry.title.strip()
    safe = safe_filename(title.replace(" ", "_"))
    print(f"\n▶ Processing: {title}")

    pdf_url = entry.id.replace("abs", "pdf") + ".pdf"
    pdf_path = download_pdf(pdf_url, f"{safe}.pdf")
    text = extract_text_from_pdf(pdf_path)
    if not text:
        print("⚠ PDFからテキスト抽出できませんでした")
        return

    summary = gemini_summarize_ja(text, title)
    print("\n✅ Summary (JA):")
    print(summary)

    # slides + scripts（← 新）
    slide_files = build_slides(title, summary, pdf_path)
    slide_scripts = build_slide_scripts(title, summary)

    # narration per slide（← 新）
    audio_files = generate_slide_audios(slide_scripts)

    
    # video（← 新：音声長に完全同期）
    today = datetime.utcnow().strftime("%Y%m%d")
    out = generate_video(
        slide_files,
        audio_files,
        f"paper_video_{today}.mp4"
    )

    
if __name__ == "__main__":
    main()
