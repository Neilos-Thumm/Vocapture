#!/usr/bin/env python3
"""
Vocabulary Capture Appliance — runs on a Raspberry Pi 5 with Camera Module 3.

Flow: set session (book/chapter) -> add <word> (photo->OCR->sentence->SQLite pending)
      -> enrich (pending -> Anthropic -> enriched) -> review (approve -> CSV + Anki -> synced).

Usage:
    python3 main.py session --book "1Q84" --chapter 7
    python3 main.py add <word>
    python3 main.py enrich
    python3 main.py review
    python3 main.py status
    python3 main.py capture-test
"""

import argparse
import csv
import json
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# --- Config -----------------------------------------------------------------

BASE = Path(__file__).resolve().parent
DB_PATH = BASE / "vocab.db"
CSV_PATH = BASE / "vocab_log.csv"
IMG_DIR = BASE / "captures"
IMG_DIR.mkdir(exist_ok=True)

ANKI_CONNECT_URL = os.environ.get("ANKI_CONNECT_URL", "http://localhost:8765")
ANKI_DECK = os.environ.get("ANKI_DECK", "Vocabulary")
ANKI_MODEL = "Basic (and reversed card)"  # ships with Anki; fields Front / Back

ANTHROPIC_MODEL = "claude-sonnet-4-6"

# --- Database ---------------------------------------------------------------

def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS session (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            book TEXT, chapter TEXT
        );
        CREATE TABLE IF NOT EXISTS words (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            word TEXT NOT NULL,
            sentence TEXT,
            book TEXT,
            chapter TEXT,
            phonetic TEXT,
            part_of_speech TEXT,
            definition TEXT,
            status TEXT NOT NULL DEFAULT 'pending',  -- pending|enriched|synced
            created_at TEXT NOT NULL
        );
        """)

def now_iso():
    return datetime.now(timezone.utc).isoformat()

# --- Session ----------------------------------------------------------------

def get_session():
    with db() as conn:
        row = conn.execute("SELECT book, chapter FROM session WHERE id = 1").fetchone()
        return (row["book"], row["chapter"]) if row else (None, None)

def set_session(book, chapter):
    with db() as conn:
        conn.execute(
            "INSERT INTO session (id, book, chapter) VALUES (1, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET book=excluded.book, chapter=excluded.chapter",
            (book, chapter),
        )
    print(f"Session set: book={book!r} chapter={chapter!r}")

# --- Capture: camera + OpenCV preprocess + Tesseract OCR --------------------

def capture_page_image():
    """Photograph the current page with the Pi camera (Camera Module 3, IMX708).
    Returns path to the raw image. Imported lazily so non-Pi tooling can load this file.

    IMPORTANT: The Camera Module 3 has autofocus, but picamera2 does NOT enable it by
    default. A book page is a close (~20-30cm), flat subject, so without driving focus
    the still comes out blurry and OCR fails. We trigger a single autofocus sweep and
    wait for lock before capturing.

    For a FIXED reading rig (camera clamped at a constant distance from the page),
    manual fixed focus is more reliable — it avoids autofocus 'hunting' on flat pages.
    See the commented alternative below; tune LensPosition once for your setup.
    LensPosition is dioptres (1/metres): 0 = infinity, ~3-5 ≈ 20-33cm, 10 ≈ 10cm.
    The standard Module 3 focuses down to ~10cm minimum.
    """
    from picamera2 import Picamera2
    from libcamera import controls

    path = IMG_DIR / f"page_{int(time.time())}.jpg"
    for i in range(4, 0, -1):
        print(f"Capturing in {i}...", flush=True)
        time.sleep(1.0)
    cam = Picamera2()
    config = cam.create_still_configuration()
    cam.configure(config)
    cam.start()
    time.sleep(1.0)  # let exposure/AE settle

    # --- Autofocus: single sweep, then lock before capture (robust default) ---
    cam.set_controls({"AfMode": controls.AfModeEnum.Auto})
    cam.autofocus_cycle()  # blocks until the sweep completes / focus locks

    # --- Manual fixed-focus alternative for a fixed rig (comment out the two
    #     lines above and uncomment the line below; tune the value to your distance):
    # cam.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": 4.0})

    cam.capture_file(str(path))
    cam.stop()
    cam.close()
    return path

def preprocess(image_path):
    """OpenCV cleanup so Tesseract reads cleanly: grayscale -> crop -> threshold -> deskew.
    Returns a processed image (numpy array)."""
    import cv2
    import numpy as np

    img = cv2.imread(str(image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Crop to left page only — thick books can't lie flat so the right page
    # curls up and confuses OCR. Tune CROP_RIGHT (0.0-1.0) for your setup.
    CROP_LEFT, CROP_RIGHT = 0.02, 0.72
    h, w = gray.shape
    gray = gray[:, int(w * CROP_LEFT):int(w * CROP_RIGHT)]
    # Adaptive threshold handles uneven page lighting better than a global one.
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )
    # Deskew: estimate text angle from non-zero pixels and rotate flat.
    coords = np.column_stack(np.where(thresh < 255))
    if coords.size:
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        (h, w) = thresh.shape
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        thresh = cv2.warpAffine(
            thresh, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
    return thresh

def ocr(processed_image):
    import pytesseract
    # --psm 6: assume a single uniform block of text (best for book pages)
    # --oem 3: use the LSTM engine
    return pytesseract.image_to_string(
        processed_image, config="--psm 6 --oem 3 -l eng"
    )

# --- Sentence extraction with the three edge cases --------------------------

def split_sentences(text):
    # Normalize OCR whitespace/newlines, then naive sentence split.
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]

def find_sentence(word, ocr_text):
    """Return the sentence containing `word`.
    EDGE CASES (first-class):
      - exactly one match  -> return it automatically
      - multiple matches   -> let the user pick from a numbered list
      - zero matches       -> let the user paste/type the sentence manually
    """
    sentences = split_sentences(ocr_text)
    pat = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
    matches = [s for s in sentences if pat.search(s)]

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        print(f"\n'{word}' appears in {len(matches)} sentences. Pick one:")
        for i, s in enumerate(matches, 1):
            print(f"  [{i}] {s}")
        choice = input("Number (or 'p' to paste manually): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(matches):
            return matches[int(choice) - 1]
        # fall through to manual paste

    # zero matches, or user chose to paste
    print(f"\n'{word}' not found cleanly in the OCR text (OCR miss or not on page).")
    print("Paste the sentence it appeared in:")
    manual = input("> ").strip()
    return manual or None

# --- Commands ---------------------------------------------------------------

def cmd_session(args):
    book = args.book
    chapter = args.chapter
    cur_book, cur_chapter = get_session()
    set_session(book or cur_book, chapter or cur_chapter)

def save_processed(img_path, processed):
    """Save the OpenCV-processed image beside the raw capture for inspection."""
    import cv2
    out = img_path.with_stem(img_path.stem + "_processed")
    cv2.imwrite(str(out), processed)
    return out

def cmd_add(args):
    word = args.word.strip()
    book, chapter = get_session()
    if not book:
        print("No session set. Run: vocab session --book ... --chapter ...")
        sys.exit(1)

    print(f"Capturing page for '{word}' (book={book}, chapter={chapter})...")
    img = capture_page_image()
    processed = preprocess(img)
    processed_path = save_processed(img, processed)
    text = ocr(processed)

    preview = text[:160].replace("\n", " ")
    print(f"OCR ({len(text)} chars): {preview!r}")
    print(f"Images: {img}  |  processed: {processed_path}")

    sentence = find_sentence(word, text)

    with db() as conn:
        conn.execute(
            "INSERT INTO words (word, sentence, book, chapter, status, created_at) "
            "VALUES (?, ?, ?, ?, 'pending', ?)",
            (word, sentence, book, chapter, now_iso()),
        )
    print(f"Staged '{word}' (pending). Sentence: {sentence!r}")

def cmd_capture_test(args):
    """Fire the camera, preprocess, OCR, print everything. No DB write. Tuning tool.

    Run this to dial in lighting and adaptiveThreshold before a real session.
    Saves both raw and processed images to captures/ for inspection.
    """
    print("Firing camera (no DB write — tuning mode)...")
    img = capture_page_image()
    processed = preprocess(img)
    processed_path = save_processed(img, processed)

    text = ocr(processed)
    sentences = split_sentences(text)

    print(f"\nRaw image:       {img}")
    print(f"Processed image: {processed_path}")
    print(f"\n--- OCR OUTPUT ({len(text)} chars, {len(sentences)} sentences) ---")
    print(text)
    print("--- END OCR ---")

def enrich_one(word, sentence):
    """One Anthropic call -> {phonetic, part_of_speech, definition} for THIS context."""
    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    prompt = (
        f"The word is: {word}\n"
        f"It appeared in this sentence: {sentence}\n\n"
        "Give the meaning that fits THIS sentence's context only. "
        "Respond with ONLY a JSON object, no prose, no markdown fences, with keys:\n"
        '  "phonetic": IPA in slashes, e.g. /məˈtɪkjələs/\n'
        '  "part_of_speech": abbreviated — one of n., v., adj., adv., prep., conj., pron., interj.\n'
        '  "definition": a concise definition of the word as used in this sentence\n'
    )
    msg = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = "".join(b.text for b in msg.content if b.type == "text").strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    return json.loads(raw)

def cmd_enrich(args):
    with db() as conn:
        rows = conn.execute(
            "SELECT id, word, sentence FROM words WHERE status = 'pending'"
        ).fetchall()
    if not rows:
        print("Nothing pending to enrich.")
        return
    print(f"Enriching {len(rows)} word(s)...")
    for r in rows:
        try:
            data = enrich_one(r["word"], r["sentence"] or "")
            with db() as conn:
                conn.execute(
                    "UPDATE words SET phonetic=?, part_of_speech=?, definition=?, "
                    "status='enriched' WHERE id=?",
                    (data.get("phonetic"), data.get("part_of_speech"),
                     data.get("definition"), r["id"]),
                )
            print(f"  ✓ {r['word']}")
        except Exception as e:
            print(f"  ✗ {r['word']}: {e}")

def append_csv(word, phonetic, pos, definition):
    new = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["word", "phonetic", "part_of_speech", "definition"])
        w.writerow([word, phonetic, pos, definition])

def push_anki(word, phonetic, pos, definition):
    """Add a Basic (and reversed card) note. Front = 'word (pos) | phonetic', Back = definition."""
    import urllib.request
    front = f"{word} ({pos}) | {phonetic}"
    payload = {
        "action": "addNote",
        "version": 6,
        "params": {"note": {
            "deckName": ANKI_DECK,
            "modelName": ANKI_MODEL,
            "fields": {"Front": front, "Back": definition},
            "options": {"allowDuplicate": False},
            "tags": ["vocab-appliance"],
        }},
    }
    req = urllib.request.Request(
        ANKI_CONNECT_URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
    if result.get("error"):
        raise RuntimeError(result["error"])
    return result.get("result")

def cmd_review(args):
    with db() as conn:
        rows = conn.execute(
            "SELECT * FROM words WHERE status = 'enriched' ORDER BY created_at"
        ).fetchall()
    if not rows:
        print("Nothing to review.")
        return
    for r in rows:
        print("\n" + "-" * 50)
        print(f"word:       {r['word']}")
        print(f"phonetic:   {r['phonetic']}")
        print(f"pos:        {r['part_of_speech']}")
        print(f"definition: {r['definition']}")
        print(f"sentence:   {r['sentence']}")
        action = input("[a]pprove / [e]dit / [s]kip / [q]uit: ").strip().lower()

        if action == "q":
            break
        if action == "s":
            continue
        if action == "e":
            r = dict(r)
            r["phonetic"] = input(f"phonetic [{r['phonetic']}]: ").strip() or r["phonetic"]
            r["part_of_speech"] = input(f"pos [{r['part_of_speech']}]: ").strip() or r["part_of_speech"]
            r["definition"] = input(f"definition [{r['definition']}]: ").strip() or r["definition"]
            with db() as conn:
                conn.execute(
                    "UPDATE words SET phonetic=?, part_of_speech=?, definition=? WHERE id=?",
                    (r["phonetic"], r["part_of_speech"], r["definition"], r["id"]),
                )

        # approve (also reached after edit)
        try:
            append_csv(r["word"], r["phonetic"], r["part_of_speech"], r["definition"])
            push_anki(r["word"], r["phonetic"], r["part_of_speech"], r["definition"])
            with db() as conn:
                conn.execute("UPDATE words SET status='synced' WHERE id=?", (r["id"],))
            print(f"  ✓ synced '{r['word']}' to CSV + Anki")
        except Exception as e:
            print(f"  ✗ sync failed for '{r['word']}': {e} (left as enriched, retry later)")

def cmd_status(args):
    with db() as conn:
        rows = conn.execute(
            "SELECT status, COUNT(*) c FROM words GROUP BY status"
        ).fetchall()
    counts = {r["status"]: r["c"] for r in rows}
    book, chapter = get_session()
    print(f"Session: book={book!r} chapter={chapter!r}")
    for s in ("pending", "enriched", "synced"):
        print(f"  {s:10s}: {counts.get(s, 0)}")

# --- Entry point ------------------------------------------------------------

def main():
    init_db()
    p = argparse.ArgumentParser(prog="vocab")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("session", help="set book/chapter for this session")
    s.add_argument("--book")
    s.add_argument("--chapter")
    s.set_defaults(func=cmd_session)

    a = sub.add_parser("add", help="capture an unknown word from the current page")
    a.add_argument("word")
    a.set_defaults(func=cmd_add)

    sub.add_parser("enrich", help="enrich pending words via Anthropic").set_defaults(func=cmd_enrich)
    sub.add_parser("review", help="approve/edit enriched words -> CSV + Anki").set_defaults(func=cmd_review)
    sub.add_parser("status", help="counts by lifecycle state").set_defaults(func=cmd_status)
    sub.add_parser("capture-test", help="fire camera + OCR, print output, no DB write (tuning)").set_defaults(func=cmd_capture_test)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
