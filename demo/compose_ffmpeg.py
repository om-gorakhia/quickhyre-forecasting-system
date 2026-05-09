"""Compose final video from pre-generated slides + audio using ffmpeg.

Much faster than moviepy for static-image videos. Each scene is a single
image held for the duration of its audio + 1.5s padding. Scenes are
concatenated with a short crossfade.

Usage:
    python demo/compose_ffmpeg.py
"""

import subprocess
import shutil
from pathlib import Path

DEMO_DIR = Path(__file__).parent
SLIDES_DIR = DEMO_DIR / "slides"
AUDIO_DIR = DEMO_DIR / "audio"
TEMP_DIR = DEMO_DIR / "temp_segments"
OUTPUT = DEMO_DIR / "Sales_Forecasting_Demo.mp4"

SCENES = [
    ("title", "01_title.png"),
    ("dataset", "02_dataset.png"),
    ("features", "03_features.png"),
    ("models", "04_models.png"),
    ("results", "05_results.png"),
    ("backtest", "06_backtest.png"),
    ("api", "07_api.png"),
    ("closing", "08_closing.png"),
]

NARRATIONS = {
    "title": (
        "This is a demo of my end-to-end sales forecasting system, "
        "built as a data science assignment for QuickHyre. "
        "The system takes weekly beverage sales data across 43 US states, "
        "trains four different model families, selects the best per state, "
        "and serves 8-week forecasts through a REST API. "
        "Let me walk you through how it works."
    ),
    "dataset": (
        "The dataset contains 8,084 rows of weekly beverage sales across 43 US states, "
        "spanning from January 2019 to March 2023. "
        "The raw data has two challenges: mixed date formats in the same column, "
        "and irregular spacing that leaves 37% of the weekly grid missing after resampling. "
        "The pipeline handles both — parsing mixed dates with pandas, "
        "then filling gaps with linear interpolation. "
        "After cleaning, we have 11,051 rows on a regular weekly grid."
    ),
    "features": (
        "The feature engineering pipeline builds 23 features per row. "
        "These include 4 lag features at 1, 7, 30, and 52 weeks back; "
        "6 rolling statistics — means and standard deviations over 4, 8, and 13-week windows; "
        "cyclical sin-cosine encodings for month and week of year; "
        "a holiday flag, trend proxy, and per-state expanding mean. "
        "The key design point is leakage prevention. "
        "Every rolling feature uses shift-1 before computing the window, "
        "so the current row's target never appears in its own features. "
        "All transforms are per-state via groupby — no cross-state information leakage."
    ),
    "models": (
        "I train four fundamentally different model families. "
        "SARIMA, a classical statistical model fitted per state using auto-ARIMA with yearly seasonality. "
        "Prophet, Facebook's additive model with US holidays and multiplicative seasonality. "
        "XGBoost, a single global gradient boosting model pooling all 43 states — "
        "because 142 rows per state isn't enough for 23 features. "
        "And LSTM, a 2-layer PyTorch recurrent network, also trained globally. "
        "Both XGBoost and LSTM use recursive multi-step forecasting: "
        "predict one week, append the prediction to history, rebuild all features, "
        "and repeat 8 times. No future data is ever used."
    ),
    "results": (
        "Here are the results. LSTM leads with 5.08% median symmetric MAPE, "
        "followed by XGBoost at 6.27%. "
        "SARIMA and Prophet trail at 12 and 18 percent. "
        "For context, the naive baseline — just repeating last week's value — scores 5.58%. "
        "So LSTM does beat naive, but the improvement is modest. "
        "The real story is in the feature importance: "
        "rolling means account for 90% of XGBoost's signal, "
        "meaning this data is dominated by simple momentum. "
        "LSTM wins 27 of 43 states, XGBoost wins 15. "
        "26 states use an ensemble blend of the top two, "
        "which improved 19 of those 26 states by an average 1.4 percentage points."
    ),
    "backtest": (
        "A single train-validation split can be misleading. "
        "The walk-forward backtest runs 3 folds with a 13-week horizon each, "
        "using an expanding training window. "
        "XGBoost proves more stable with a standard deviation of 2.88 "
        "versus 4.57 for LSTM. "
        "This confirms the ensemble strategy: "
        "LSTM adds value on specific states, but XGBoost provides the reliable foundation."
    ),
    "api": (
        "The forecasts are served through a FastAPI application with 6 endpoints. "
        "Health check, list states, forecast by state, model info, and a global leaderboard. "
        "The API loads pre-computed predictions at startup — "
        "no model inference at request time, so responses are sub-millisecond. "
        "State names are case-insensitive. "
        "Interactive Swagger documentation is auto-generated at the docs endpoint. "
        "The entire system is tested with 57 automated tests covering features, "
        "metrics, ingestion, all four models, API endpoints, and validation guards."
    ),
    "closing": (
        "To summarize: this system handles the full lifecycle — "
        "from messy raw data to clean predictions served through an API. "
        "23 leakage-free features, 4 model families, "
        "validated ensemble selection, walk-forward backtesting, "
        "57 tests, and an HTML report that summarizes everything. "
        "The honest limitation is the 37% imputation rate and no exogenous variables. "
        "The natural next step would be probabilistic forecasting — "
        "prediction intervals instead of point estimates. "
        "Thanks for watching."
    ),
}


def get_audio_duration(path):
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def srt_timestamp(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def main():
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Create per-scene video segments (image + audio)
    segment_paths = []
    srt_entries = []
    cumulative = 0.0

    for scene_id, slide_file in SCENES:
        slide = SLIDES_DIR / slide_file
        audio = AUDIO_DIR / f"{scene_id}.mp3"
        segment = TEMP_DIR / f"{scene_id}.mp4"

        audio_dur = get_audio_duration(audio)
        total_dur = audio_dur + 1.5  # 1.5s breathing room after narration

        print(f"  {scene_id}: {audio_dur:.1f}s audio -> {total_dur:.1f}s segment")

        # ffmpeg: loop image for duration, add audio, pad with silence
        subprocess.run([
            "ffmpeg", "-y",
            "-loop", "1", "-i", str(slide),
            "-i", str(audio),
            "-c:v", "libx264", "-tune", "stillimage",
            "-c:a", "aac", "-b:a", "192k",
            "-pix_fmt", "yuv420p",
            "-t", f"{total_dur:.2f}",
            "-shortest",
            "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=#0F172A",
            "-r", "24",
            str(segment),
        ], capture_output=True, check=True)

        segment_paths.append(segment)

        # SRT entry
        start = srt_timestamp(cumulative)
        end = srt_timestamp(cumulative + audio_dur)
        narration = NARRATIONS[scene_id]
        srt_entries.append(f"{len(srt_entries)+1}\n{start} --> {end}\n{narration}\n")
        cumulative += total_dur

    # Step 2: Concatenate segments
    concat_file = TEMP_DIR / "concat.txt"
    with open(concat_file, "w") as f:
        for seg in segment_paths:
            f.write(f"file '{seg}'\n")

    print(f"\nConcatenating {len(segment_paths)} segments ({cumulative:.0f}s total)...")
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-c:v", "libx264", "-preset", "medium", "-crf", "23",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(OUTPUT),
    ], capture_output=True, check=True)

    print(f"Video saved: {OUTPUT}")

    # Step 3: Write SRT subtitles
    srt_path = DEMO_DIR / "subtitles.srt"
    srt_path.write_text("\n".join(srt_entries), encoding="utf-8")
    print(f"Subtitles saved: {srt_path}")

    # Step 4: Thumbnail
    thumb = DEMO_DIR / "thumbnail.png"
    shutil.copy2(SLIDES_DIR / "01_title.png", thumb)
    print(f"Thumbnail saved: {thumb}")

    # Step 5: Narration script
    script_path = DEMO_DIR / "narration_script.md"
    lines = ["# Narration Script\n",
             f"**Total duration**: ~{cumulative:.0f} seconds ({cumulative/60:.1f} minutes)\n"]
    scene_start = 0.0
    for scene_id, _ in SCENES:
        audio_dur = get_audio_duration(AUDIO_DIR / f"{scene_id}.mp3")
        total_dur = audio_dur + 1.5
        lines.append(f"## {scene_id.title()} [{srt_timestamp(scene_start)} -> {srt_timestamp(scene_start + audio_dur)}]\n")
        lines.append(f"{NARRATIONS[scene_id]}\n")
        scene_start += total_dur
    script_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Script saved: {script_path}")

    # Step 6: Clean temp
    shutil.rmtree(TEMP_DIR)
    print("Temp files cleaned.")

    # Final size
    size_mb = OUTPUT.stat().st_size / (1024 * 1024)
    print(f"\nDone! {OUTPUT.name}: {size_mb:.1f} MB, ~{cumulative:.0f}s")


if __name__ == "__main__":
    main()
