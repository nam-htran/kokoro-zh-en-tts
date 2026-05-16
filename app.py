from functools import lru_cache
import re
import tempfile
import unicodedata

import gradio as gr
import numpy as np
import soundfile as sf
import torch
from kokoro import KModel, KPipeline


REPO_ID = "hexgrad/Kokoro-82M-v1.1-zh"
SAMPLE_RATE = 24000
MAX_TEXT_CHARS = 1800
SILENCE_SECONDS = 0.18
MODE_SINGLE = "Single voice"
MODE_DIALOGUE = "Dialogue"

EN_US_VOICES = "af_maple af_sol".split()
EN_GB_VOICES = "bf_vale".split()
ZH_VOICES = """
zf_001 zf_002 zf_003 zf_004 zf_005 zf_006 zf_007 zf_008 zf_017 zf_018
zf_019 zf_021 zf_022 zf_023 zf_024 zf_026 zf_027 zf_028 zf_032 zf_036
zf_038 zf_039 zf_040 zf_042 zf_043 zf_044 zf_046 zf_047 zf_048 zf_049
zf_051 zf_059 zf_060 zf_067 zf_070 zf_071 zf_072 zf_073 zf_074 zf_075
zf_076 zf_077 zf_078 zf_079 zf_083 zf_084 zf_085 zf_086 zf_087 zf_088
zf_090 zf_092 zf_093 zf_094 zf_099 zm_009 zm_010 zm_011 zm_012 zm_013
zm_014 zm_015 zm_016 zm_020 zm_025 zm_029 zm_030 zm_031 zm_033 zm_034
zm_035 zm_037 zm_041 zm_045 zm_050 zm_052 zm_053 zm_054 zm_055 zm_056
zm_057 zm_058 zm_061 zm_062 zm_063 zm_064 zm_065 zm_066 zm_068 zm_069
zm_080 zm_081 zm_082 zm_089 zm_091 zm_095 zm_096 zm_097 zm_098 zm_100
""".split()
ZH_FEMALE_VOICES = [voice for voice in ZH_VOICES if voice.startswith("zf_")]
ZH_MALE_VOICES = [voice for voice in ZH_VOICES if voice.startswith("zm_")]

VOICE_BY_LANGUAGE = {
    "z": ZH_VOICES,
    "a": EN_US_VOICES,
    "b": EN_GB_VOICES,
}
ALL_PRESET_VOICES = ZH_VOICES + EN_US_VOICES + EN_GB_VOICES


@lru_cache(maxsize=1)
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def get_model():
    model = KModel(repo_id=REPO_ID).to(get_device()).eval()
    return model


@lru_cache(maxsize=1)
def get_en_phoneme_pipeline():
    return KPipeline(lang_code="a", repo_id=REPO_ID, model=False)


def en_callable(text):
    if text == "Kokoro":
        return "k\u02c8Ok\u0259\u0279O"
    if text == "Sol":
        return "s\u02c8Ol"
    return next(get_en_phoneme_pipeline()(text)).phonemes


@lru_cache(maxsize=3)
def get_pipeline(lang_code):
    model = get_model()
    if lang_code == "z":
        return KPipeline(
            lang_code="z",
            repo_id=REPO_ID,
            model=model,
            en_callable=en_callable,
        )
    return KPipeline(lang_code=lang_code, repo_id=REPO_ID, model=model)


def split_text(text):
    pieces = []
    for paragraph in re.split(r"\n+", text.strip()):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        sentences = re.split(r"(?<=[。！？!?；;])\s*", paragraph)
        pieces.extend(sentence.strip() for sentence in sentences if sentence.strip())
    return pieces or [text.strip()]


def chinese_speed(base_speed):
    def speed_for_phonemes(len_ps):
        speed = float(base_speed)
        if 83 < len_ps < 183:
            speed *= 1 - (len_ps - 83) / 500
        elif len_ps >= 183:
            speed *= 0.8
        return max(0.5, min(1.6, speed))

    return speed_for_phonemes


def result_audio(result):
    if hasattr(result, "audio"):
        return result.audio
    if isinstance(result, tuple) and result:
        return result[-1]
    raise gr.Error("Kokoro did not return audio for this input.")


def resolve_voice(language, preset_voice):
    choices = VOICE_BY_LANGUAGE[language]
    if preset_voice not in choices:
        return choices[0]
    return preset_voice


def parse_dialogue_lines(text):
    turns = []

    for line_number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue

        match = re.match(r"^([^:：-]+)\s*[:：-]\s*(.+)$", line)
        if not match:
            raise gr.Error(f"Dialogue line {line_number} needs a label, for example: Nam: ... or Nu: ...")

        speaker = normalize_speaker_label(match.group(1))
        content = match.group(2).strip()

        if content:
            turns.append((speaker, content))

    return turns


def normalize_speaker_label(label):
    original = label.strip().lower()
    ascii_label = (
        unicodedata.normalize("NFKD", original)
        .encode("ascii", "ignore")
        .decode("ascii")
        .replace(" ", "")
    )

    if original in {"男", "男声"} or ascii_label in {"nam", "male", "m"}:
        return "male"
    if original in {"女", "女声"} or ascii_label in {"nu", "female", "f"}:
        return "female"

    raise gr.Error(f"Unknown speaker label: {label}. Use Nam: or Nu:")


def text_items(text, mode, language, preset_voice, male_voice, female_voice):
    if mode == MODE_SINGLE:
        voice = resolve_voice(language, preset_voice)
        return [(voice, chunk) for chunk in split_text(text)]

    if language != "z":
        raise gr.Error("Dialogue mode needs Chinese voices because this repo only has zf/zm male-female presets.")

    male_voice = male_voice if male_voice in ZH_MALE_VOICES else ZH_MALE_VOICES[0]
    female_voice = female_voice if female_voice in ZH_FEMALE_VOICES else ZH_FEMALE_VOICES[0]
    voice_by_speaker = {
        "male": male_voice,
        "female": female_voice,
    }

    items = []
    for speaker, line in parse_dialogue_lines(text):
        items.extend((voice_by_speaker[speaker], chunk) for chunk in split_text(line))
    return items


def synthesize(text, language, mode, preset_voice, male_voice, female_voice, speed):
    text = (text or "").strip()
    if not text:
        raise gr.Error("Please enter text.")
    if len(text) > MAX_TEXT_CHARS:
        raise gr.Error(f"Please keep text under {MAX_TEXT_CHARS} characters.")

    pipeline = get_pipeline(language)
    items = text_items(text, mode, language, preset_voice, male_voice, female_voice)
    speed_arg = chinese_speed(speed) if language == "z" else float(speed)
    silence = np.zeros(int(SAMPLE_RATE * SILENCE_SECONDS), dtype=np.float32)
    wavs = []

    for index, (voice, chunk) in enumerate(items):
        generator = pipeline(chunk, voice=voice, speed=speed_arg)
        chunk_wavs = [np.asarray(result_audio(result), dtype=np.float32) for result in generator]
        if not chunk_wavs:
            continue
        if index and SILENCE_SECONDS:
            wavs.append(silence)
        wavs.extend(chunk_wavs)

    if not wavs:
        raise gr.Error("No audio was generated.")

    audio = np.concatenate(wavs)
    output = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    output.close()
    sf.write(output.name, audio, SAMPLE_RATE)
    return output.name


def update_preset_voice_choices(language):
    choices = VOICE_BY_LANGUAGE[language]
    return gr.update(choices=choices, value=choices[0])


def update_mode(mode):
    return (
        gr.update(visible=mode == MODE_SINGLE),
        gr.update(visible=mode == MODE_DIALOGUE),
    )


with gr.Blocks(title="Kokoro 82M v1.1 zh TTS") as demo:
    gr.Markdown("# Kokoro 82M v1.1 zh TTS")
    with gr.Row():
        with gr.Column(scale=2):
            text = gr.Textbox(
                label="Text",
                value="Kokoro 是一系列体积虽小但功能强大的 TTS 模型。",
                lines=7,
                max_lines=12,
            )
            with gr.Row():
                language = gr.Dropdown(
                    label="Language",
                    choices=[
                        ("Chinese", "z"),
                        ("English US", "a"),
                        ("English UK", "b"),
                    ],
                    value="z",
                )
                preset_voice = gr.Dropdown(
                    label="Preset voice",
                    choices=ALL_PRESET_VOICES,
                    value="zf_001",
                )
            mode = gr.Radio(
                label="Mode",
                choices=[MODE_SINGLE, MODE_DIALOGUE],
                value=MODE_SINGLE,
            )
            with gr.Row(visible=False) as dialogue_voices:
                male_voice = gr.Dropdown(
                    label="Male voice",
                    choices=ZH_MALE_VOICES,
                    value=ZH_MALE_VOICES[0],
                )
                female_voice = gr.Dropdown(
                    label="Female voice",
                    choices=ZH_FEMALE_VOICES,
                    value=ZH_FEMALE_VOICES[0],
                )
            speed = gr.Slider(
                label="Speed",
                minimum=0.6,
                maximum=1.4,
                value=1.0,
                step=0.05,
            )
            submit = gr.Button("Generate", variant="primary")
        with gr.Column(scale=1):
            audio = gr.Audio(label="Audio", type="filepath")

    gr.Examples(
        examples=[
            [
                "Kokoro 是一系列体积虽小但功能强大的 TTS 模型。",
                "z",
                MODE_SINGLE,
                "zf_001",
                ZH_MALE_VOICES[0],
                ZH_FEMALE_VOICES[0],
                1.0,
            ],
            [
                "这是一段用于测试中文语音合成的简短文本。",
                "z",
                MODE_SINGLE,
                "zm_010",
                ZH_MALE_VOICES[0],
                ZH_FEMALE_VOICES[0],
                1.0,
            ],
            [
                "Nam: 今天我们开始测试双人对话。\nNữ: 好的，我会接着说下一句。\nNam: 这样听起来就像两个人轮流说话。",
                "z",
                MODE_DIALOGUE,
                "zf_001",
                "zm_010",
                "zf_001",
                1.0,
            ],
            [
                "Kokoro is a small but powerful text to speech model.",
                "a",
                MODE_SINGLE,
                "af_maple",
                ZH_MALE_VOICES[0],
                ZH_FEMALE_VOICES[0],
                1.0,
            ],
        ],
        inputs=[text, language, mode, preset_voice, male_voice, female_voice, speed],
        cache_examples=False,
    )

    language.change(update_preset_voice_choices, inputs=language, outputs=preset_voice)
    mode.change(update_mode, inputs=mode, outputs=[preset_voice, dialogue_voices])
    submit.click(
        synthesize,
        inputs=[text, language, mode, preset_voice, male_voice, female_voice, speed],
        outputs=audio,
    )


if __name__ == "__main__":
    demo.queue(max_size=8).launch()
