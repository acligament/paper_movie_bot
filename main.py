import os
import re
import json
import time
import textwrap
from datetime import datetime
from urllib.parse import quote_plus

import requests
import feedparser
import fitz  # PyMuPDF

from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont

# moviepy 2.x 対応：moviepy.editor は使わない
from moviepy import ImageClip, AudioFileClip, concatenate_videoclips

SAVE_DIR = "outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------------
# 設定（環境変数で上書き可）
# -----------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")  # 例：gemini-1.5-flash
MAX_PAPERS = int(os.environ.get("MAX_PAPERS", "3"))
SLIDE_SECONDS = float(os.environ.get("SLIDE_SECONDS", "4"))
TTS_LANG = os.environ.get("TTS_LANG", "ja")  # gTTS: "ja"
FONT_PATH = os.environ.get("FONT_PATH", "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf")

# -----------------------------
# Utility
# -----------------------------
def safe_filename(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\r\n]', "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")

def http_get(url: str, timeout=30):
    return requests.get(url, timeout=timeout, headers={"User-Agent": "paper-ai-bot/1.0"})

def http_post(url: str, json_data: dict, timeout=60):
    return requests.post(url, json=json_data, timeout=timeout, headers={"Content-Type": "application/json"})

# -----------------------------
# ① arXiv 取得
# -----------------------------
def fetch_arxiv_papers(max_results: int):
    raw_query = "cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:cs.CV OR cat:stat.ML"
    encoded_query = quote_plus(raw_query)

    url = (
        "http://export.arxiv.org/api/query?"
        f"search_query={encoded_query}"
        "&start=0"
        f"&max_results={max_results}"
        "&sortBy=submittedDate"
        "&sortOrder=descending"
    )
    feed = feedparser.parse(url)
    return feed.entries

# -----------------------------
# ② PDF ダウンロード
# -----------------------------
def download_pdf(pdf_url: str, filename: str):
    try:
        res = http_get(pdf_url, timeout=40)
        res.raise_for_status()
    except Exception as e:
        print(f"PDF download failed: {pdf_url} error={e}")
        return None

    path = os.path.join(SAVE_DIR, filename)
    with open(path, "wb") as f:
        f.write(res.content)
    return path

# -----------------------------
# ③ PDF → テキスト抽出
# -----------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    if not pdf_path:
        return ""
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"PDF open failed: {pdf_path} error={e}")
        return ""

    texts = []
    for page in doc:
        texts.append(page.get_text())
    return "\n".join(texts)

def truncate_for_llm(text: str, max_chars=12000) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]

# -----------------------------
# ④ Gemini（REST）で日本語要約
#    - SDK不使用で Illegal metadata 回避
# -----------------------------
def gemini_summarize_ja(text: str) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY が空です。GitHub Secrets に設定してください。")

    prompt = f"""あなたは日本語が得意なAI研究者です。
以下の論文本文を、日本語で短く要約してください。

条件:
- 箇条書き3点以内
- 各点は最大40文字
- 核心だけ
- 専門用語はできるだけ平易に

本文:
{text}
"""

    endpoint = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
        f"?key={GEMINI_API_KEY}"
    )  # generateContent REST  [oai_citation:2‡Google AI for Developers](https://ai.google.dev/api?utm_source=chatgpt.com)

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 300,
        },
    }

    r = http_post(endpoint, payload, timeout=90)
    if r.status_code != 200:
        raise RuntimeError(f"Gemini REST error {r.status_code}: {r.text[:500]}")

    data = r.json()
    # candidates[0].content.parts[0].text が基本
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        return json.dumps(data, ensure_ascii=False)[:500]

# -----------------------------
# ⑤ gTTS で音声生成（無料枠で簡易に）
# -----------------------------
def tts_gtts(text: str, out_mp3: str) -> str:
    path = os.path.join(SAVE_DIR, out_mp3)
    # 読み上げに邪魔な記号を除去
    cleaned = text.replace("**", "").replace("_", "").strip()
    gTTS(text=cleaned, lang=TTS_LANG).save(path)
    return path

# -----------------------------
# ⑥ スライド用テキスト構成
# -----------------------------
def build_slides(title: str, summary: str):
    # summary を箇条書きとして扱う
    slides = []
    slides.append(("TITLE", title))
    slides.append(("SUMMARY", summary))
    slides.append(("OUTRO", "以上です。良い一日を！"))
    return slides

# -----------------------------
# ⑦ Pillowでスライド画像生成（日本語フォント）
# -----------------------------
def create_slide_image(header: str, body: str, out_png: str) -> str:
    W, H = 1280, 720
    img = Image.new("RGB", (W, H), color="white")
    draw = ImageDraw.Draw(img)

    # フォント（Actionsでは fonts-ipafont-gothic を入れる）
    try:
        font_h = ImageFont.truetype(FONT_PATH, 44)
        font_b = ImageFont.truetype(FONT_PATH, 34)
    except Exception:
        font_h = ImageFont.load_default()
        font_b = ImageFont.load_default()

    x, y = 70, 60
    draw.text((x, y), header, fill="black", font=font_h)

    wrapped = "\n".join(textwrap.wrap(body, width=28))
    draw.multiline_text((x, y + 90), wrapped, fill="black", font=font_b, spacing=18)

    path = os.path.join(SAVE_DIR, out_png)
    img.save(path)
    return path

# -----------------------------
# ⑧ 動画生成（mp3 + png）
# -----------------------------
def make_video(slide_pngs, audio_mp3, out_mp4) -> str:
    clips = [ImageClip(p).with_duration(SLIDE_SECONDS) for p in slide_pngs]
    video = concatenate_videoclips(clips, method="compose")

    audio = AudioFileClip(audio_mp3)
    final = video.with_audio(audio)

    out_path = os.path.join(SAVE_DIR, out_mp4)
    final.write_videofile(out_path, fps=24, codec="libx264", audio_codec="aac")
    return out_path

# -----------------------------
# MAIN
# -----------------------------
def main():
    print("📥 Fetching AI papers...")
    papers = fetch_arxiv_papers(MAX_PAPERS)

    if not papers:
        print("No papers found.")
        return

    # 先頭1本だけ動画化（安定優先）
    entry = papers[0]
    raw_title = entry.title.strip()
    print(f"\n▶ Processing: {raw_title}")

    pdf_url = entry.id.replace("abs", "pdf") + ".pdf"
    pdf_path = download_pdf(pdf_url, safe_filename(raw_title) + ".pdf")

    text = extract_text_from_pdf(pdf_path)
    if not text:
        print("⚠ PDF text extract failed.")
        return

    text = truncate_for_llm(text, max_chars=12000)

    print("🧠 Summarizing by Gemini (REST)...")
    summary_ja = gemini_summarize_ja(text)
    print("✅ Summary done.")

    # スライド
    slides = build_slides(raw_title, summary_ja)
    slide_pngs = []
    for i, (h, b) in enumerate(slides, start=1):
        slide_pngs.append(create_slide_image(h, b, f"slide_{i:02d}.png"))

    # 音声（要約読み上げ）
    today = datetime.utcnow().strftime("%Y%m%d")
    audio_mp3 = tts_gtts(summary_ja, f"narration_{today}.mp3")

    # 動画
    print("🎬 Creating video...")
    mp4 = make_video(slide_pngs, audio_mp3, f"paper_video_{today}.mp4")
    print(f"🎉 Done: {mp4}")

if __name__ == "__main__":
    main()
