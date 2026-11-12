# Promptwood

Chain image-to-video clips into one film: every prompt continues from the last frame of the clip before. Video
Chain is a desktop app (PySide6) and a command line that drive a local [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
with the MiniMax H3 video model. Around the clips it has a storyboard with branches, pictures (create, edit,
inpaint), a cast with faces and voices, spoken lines (text to speech), a prompt refiner with a local language model,
head and face swap, 2K refine and upscaled export.

Everything runs on your own computer. Nothing is uploaded, unless you choose ElevenLabs for voices or look up LoRA
trigger words at CivitAI (both optional, both with your own key).

## What you need

| | |
|---|---|
| System | Linux, x86-64 (tested on Fedora 44; Ubuntu and Arch should work the same) |
| Graphics card | NVIDIA with 24 GB of memory (16 GB works, slower); a current driver (580 or newer for CUDA 13) |
| Disk | 145 GB free for the standard set, 55 GB for video clips only |
| Main memory | 64 GB recommended |
| Internet | for the installation only (about 126 GB of models) |

You do **not** need Python, ComfyUI, ffmpeg, git or a compiler beforehand: the installer brings what is missing.
The one thing it cannot install is the NVIDIA driver; it tells you the command for your system.

## Install

```bash
git clone <this repository> video-chain
cd video-chain
./install.sh
```

That is the whole installation. The script

1. gets [uv](https://docs.astral.sh/uv/) (if you have none) and installs the app into `.venv` with the tested
   package versions (`uv.lock`),
2. checks the computer (graphics card, driver, disk space, ffmpeg, the libraries the window needs) and names
   anything it cannot fix itself, with the command for your system,
3. shows the plan and the licences and asks once,
4. installs ComfyUI (at the revision the app was tested with) with PyTorch for your driver, the custom nodes, the
   model files (resumed after an interruption, each checked by SHA-256), the text to speech environment and
   llama.cpp,
5. checks the result and **proves it**: it starts ComfyUI, asks it for every node and model the app uses, renders a
   short real clip with sound, speaks a test line and asks the language model a question.

When it ends with `Promptwood is ready`, every part of the app works. Start it from the applications menu or with
`video-chain-gui`.

Nothing needs root and nothing is written outside two folders: the app's own folder and its *home*
(`~/.local/share/video-chain`, or `$VIDEO_CHAIN_HOME`), which holds ComfyUI, the models and the tools.
To remove everything, delete these two folders (and `~/.local/share/applications/video-chain.desktop`).

### Options

```bash
./install.sh --profile core           # video clips only (40 GB)
./install.sh --profile full           # also extra speed modes, Krea-2 and both head swaps (184 GB)
./install.sh --with image-krea2       # the standard set plus one part
./install.sh --without tts,llm        # the standard set minus parts
./install.sh --models-from ~/ComfyUI  # a new ComfyUI that reads the models of the one you have, in place
./install.sh --hf-cache ~/.cache/huggingface/hub       # text to speech / language models you already have
./install.sh --comfy-dir ~/ComfyUI    # use the ComfyUI you already have (add --comfy-python if it has no .venv)
./install.sh --yes --accept-licenses  # no questions (for scripts)
VIDEO_CHAIN_HOME=/big/disk/video-chain ./install.sh    # everything on another disk
./install.sh --help                   # all options and all parts with their sizes
```

**Nothing you already have is downloaded again.** Setup looks into ComfyUI's model folders (also the ones from
its `extra_model_paths.yaml`, and the ones you name with `--models-from`) and into your Hugging Face cache; another
build of the same model (bf16 instead of int8, another turbo LoRA, your own GGUF language model with its vision
part) counts as well, exactly as the app itself would accept it. `./install.sh --dry-run` shows what would be
downloaded without changing anything. [docs/MODELS.md](docs/MODELS.md) lists every download with its source and licence.

### Later

```bash
video-chain doctor                    # what is installed, what is missing, and the command that fixes it
video-chain doctor --render           # also render a real test clip
video-chain setup                     # install whatever is missing (safe to run again at any time)
video-chain setup --with headswap-wan # add a part
```

The same report is in the app under **Help > Setup check**.

## Use

- `video-chain-gui` opens the app. It starts ComfyUI itself when a clip is to be made (the Start button next to the
  ComfyUI status), or uses the one that is running.
- `video-chain start.jpg prompts.txt --out ~/chains/test1` renders a chain from the command line: one prompt per
  line, each clip starts from the last frame of the one before. `video-chain --help` lists the options.

Where the app looks for things is set by the installer in `<home>/config.json` and can be changed in the app
(**Settings > ComfyUI and preferences**) or with environment variables; `src/video_chain/paths.py` lists them.

## Development

```bash
uv sync --extra faces --extra dev
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest          # no models, no GPU, no network
.venv/bin/python -m pytest -m models                           # with real models in a running ComfyUI
python tools/build_manifest.py                                 # after changing the list of downloads
```

The installer lives in `src/video_chain/setup/` (standard library only): `manifest.json` is the list of parts,
`installer.py` the steps, `doctor.py` the checks, `download.py` the resumable downloads.

## Licences of what gets installed

Promptwood installs programs and models made by others. Each keeps its own licence; the installer shows the ones
with special terms before it downloads anything (the MiniMax H3 models come under the MiniMax H3 Community
License). See [docs/MODELS.md](docs/MODELS.md).
