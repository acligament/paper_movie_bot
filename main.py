import requests
import feedparser
import fitz  # PyMuPDF
import os
import re
from datetime import datetime
from urllib.parse import quote_plus

# -----------------------------
# 追加：画像 & 動画生成モジュール
# -----------------------------
from PIL import Image, ImageDraw
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips

# -----------------------------
#  Gemini API（無料要約用）
# -----------------------------
import google.generativeai as genai
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

SAVE_DIR = "outputs"
os.makedirs(SAVE_DIR, exist_ok=True)


# -------------------------------------------------------
# ファイル名を安全に整形
# -------------------------------------------------------
def safe_filename(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\r\n]', '_', name)
    name = re.sub(r'_+', '_', name)
    return name.strip('_')


# -------------------------------------------------------
# ① arXiv 最新 AI 論文取得
# -------------------------------------------------------
def fetch_arxiv_papers():
    raw_query = "cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:cs.CV OR cat:stat.ML"
    encoded_query = quote_plus(raw_query)

    url = (
        "http://export.arxiv.org/api/query?"
        f"search_query={encoded_query}"
        "&start=0"
        "&max_results=5"
        "&sortBy=submittedDate"
        "&sortOrder=descending"
    )

    feed = feedparser.parse(url)
    return feed.entries


# -------------------------------------------------------
# ② PDF ダウンロード
# -------------------------------------------------------
def download_pdf(pdf_url, filename):
    try:
        res = requests.get(pdf_url, timeout=20)
        res.raise_for_status()
    except Exception as e:
        print(f"PDF download failed: {pdf_url}, error={e}")
        return None

    filepath = os.path.join(SAVE_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(res.content)
    return filepath


# -------------------------------------------------------
# ③ PDF → テキスト抽出
# -------------------------------------------------------
def extract_text_from_pdf(pdf_path):
    if not pdf_path:
        return ""

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"PDF open failed: {pdf_path}, {e}")
        return ""

    text = ""
    for page in doc:
        text += page.get_text()

    return text


# -------------------------------------------------------
# ④ 日本語要約（Google Gemini・無料）
# -------------------------------------------------------
def summarize_text_ja(text):

    if len(text) > 10000:
        text = text[:10000]

    prompt = f"""
あなたは日本語が得意なAI研究者です。
以下の論文本文を、簡潔でわかりやすい日本語で要約してください。

条件:
- 箇条書き 3点以内
- 各点は最大 30文字以内
- 専門用語は簡単に

本文:
{text}
"""

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)

    return response.text


# -------------------------------------------------------
# ★ VOICEVOX：speaker ID を名前で自動取得
# -------------------------------------------------------
def get_speaker_id(target_name="四国めたん", target_style="ノーマル"):
    speakers = requests.get("http://localhost:50021/speakers").json()

    for sp in speakers:
        if sp["name"] == target_name:
            for st in sp["styles"]:
                if st["name"] == target_style:
                    return st["id"]

    return None


# -------------------------------------------------------
# ⑤ VOICEVOX 音声生成（四国めたん × speed=1.1）
# -------------------------------------------------------
def generate_voice_voicevox(
    text,
    filename,
    speaker_name="四国めたん",
    style="ノーマル",
    speed=1.1
):
    audio_path = os.path.join(SAVE_DIR, filename)

    speaker_id = get_speaker_id(speaker_name, style)
    if speaker_id is None:
        raise ValueError(f"Speaker {speaker_name}/{style} が見つかりません")

    cleaned = text.replace("**", "").replace("_", "")

    query = requests.post(
        "http://localhost:50021/audio_query",
        params={"text": cleaned, "speaker": speaker_id}
    ).json()

    query["speedScale"] = speed

    synthesis = requests.post(
        "http://localhost:50021/synthesis",
        params={"speaker": speaker_id},
        json=query
    )

    with open(audio_path, "wb") as f:
        f.write(synthesis.content)

    return audio_path


# -------------------------------------------------------
# ⑥ スライド構成（Gemini で生成）
# -------------------------------------------------------
def slide_structure_from_summary(title, summary):

    prompt = f"""
次の論文を、動画用に以下の5スライドに分けてください。

1. TITLE: タイトル
2. PURPOSE: 研究の目的
3. METHOD: 手法
4. RESULT: 結果
5. CONCLUSION: 結論

形式：
TITLE: ...
PURPOSE: ...
METHOD: ...
RESULT: ...
CONCLUSION: ...

論文タイトル:
{title}

要約:
{summary}
"""

    model = genai.GenerativeModel("gemini-1.5-flash")
    res = model.generate_content(prompt).text

    slides = {}
    for line in res.split("\n"):
        if ":" in line:
            key, val = line.split(":", 1)
            slides[key.strip()] = val.strip()

    return slides


# -------------------------------------------------------
# ⑦ スライド画像生成
# -------------------------------------------------------
def create_slide_image(text, filename):
    W, H = 1920, 1080
    img = Image.new("RGB", (W, H), color="white")
    draw = ImageDraw.Draw(img)

    x, y = 120, 180
    draw.multiline_text((x, y), text, fill="black", spacing=30)

    img.save(filename)
    return filename


# -------------------------------------------------------
# ⑧ 動画生成
# -------------------------------------------------------
def generate_video(slide_files, audio_path, output_path):

    clips = [ImageClip(slide).set_duration(4) for slide in slide_files]

    video = concatenate_videoclips(clips, method="compose")
    audio = AudioFileClip(audio_path)

    final = video.set_audio(audio)
    final.write_videofile(output_path, fps=24)

    return output_path


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
def main():
    print("📥 Fetching new AI papers...")
    papers = fetch_arxiv_papers()

    summaries = []

    for entry in papers:
        raw_title = entry.title
        filename = safe_filename(raw_title.replace(" ", "_"))

        print(f"\n▶ Processing: {raw_title}")

        pdf_url = entry.id.replace("abs", "pdf") + ".pdf"
        pdf_path = download_pdf(pdf_url, f"{filename}.pdf")

        text = extract_text_from_pdf(pdf_path)
        if not text:
            continue

        summary_ja = summarize_text_ja(text)

        summaries.append({"title": raw_title, "summary": summary_ja})

    if not summaries:
        print("No papers found.")
        return

    # 1つ目の論文を動画化
    first = summaries[0]
    title = first["title"]
    summary = first["summary"]

    print("\n📝 Creating slide structure...")
    slide_data = slide_structure_from_summary(title, summary)

    # スライド生成
    slide_files = []
    for key in ["TITLE", "PURPOSE", "METHOD", "RESULT", "CONCLUSION"]:
        text = f"{key}\n\n{slide_data.get(key, '')}"
        path = os.path.join(SAVE_DIR, f"slide_{key.lower()}.png")
        create_slide_image(text, path)
        slide_files.append(path)

    # 音声生成
    today_str = datetime.utcnow().strftime("%Y%m%d")
    audio_file = generate_voice_voicevox(
        summary,
        f"narration_{today_str}.wav",
        speaker_name="四国めたん",
        style="ノーマル",
        speed=1.1
    )

    # 動画生成
    video_output = os.path.join(SAVE_DIR, f"paper_video_{today_str}.mp4")
    print("\n🎬 Generating video...")
    generate_video(slide_files, audio_file, video_output)

    print(f"\n🎉 完成！動画ファイル → {video_output}")


if __name__ == "__main__":
    main()
