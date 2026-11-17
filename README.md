# Promptwood

Chain image-to-video clips into one film: every prompt continues from the last frame of the clip before. Video
Chain is a desktop app (PySide6) and a command line that drive a local [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
with the MiniMax H3 video model. Around the clips it has a storyboard with branches, pictures (create, edit,
inpaint), a cast with faces and voices, spoken lines (text to speech), a prompt refiner with a local language model,
head and face swap, 2K refine and upscaled export.

Everything runs on your own computer. Nothing is uploaded, unless you choose ElevenLabs for voices, look up LoRA
trigger words at CivitAI, or upscale an export on a RunPod cloud GPU (all optional, each with your own key; RunPod
gets the clips it upscales).

## What you need

| | |
|---|---|
| System | Linux, x86-64 (tested on Fedora 44; Ubuntu and Arch should work the same) |
| Graphics card | NVIDIA with 24 GB of memory (16 GB works, slower); a current driver (580 or newer for CUDA 13) |
| Disk | 145 GB free for the standard set, 55 GB for video clips only, 270 GB for everything |
| Main memory | 64 GB recommended |
| Internet | for the installation only (about 126 GB of models) |

You do **not** need Python, ComfyUI, ffmpeg, git or a compiler beforehand: the installer brings what is missing.
The one thing it cannot install is the NVIDIA driver; it tells you the command for your system.

**Windows:** prepared but not yet tested. The app has Windows branches for everything that differs there (they
are in `src/video_chain/win.py`), and `install.ps1` is the counterpart of `install.sh`; neither has run on a
Windows computer yet. On Linux none of that code runs.

## Install

```bash
git clone https://github.com/mschindler83/promptwood.git
cd promptwood
./install.sh
```

Without git: download **Source code (tar.gz)** of the latest release from https://github.com/mschindler83/promptwood/releases, unpack
it, and run `./install.sh` in that folder.

That is the whole installation. The script

1. asks first, before anything big is installed: whether you already have a ComfyUI (its models are then used in
   place), and how much you want (video clips only, the recommended set, or everything), with what each choice
   still has to download on your computer,
2. shows the plan, checks the computer (graphics card, driver, disk space, ffmpeg) and names anything it cannot
   fix itself, with the command for your system,
3. shows the licence every publisher states for its models and asks whether you accept them,
4. installs, in numbered steps with one line each: the app (with the package versions it was tested with),
   ComfyUI at the tested revision with PyTorch for your driver, the custom nodes, the model files (resumed after an
   interruption, each checked by SHA-256), text to speech and llama.cpp,
5. checks the result and **tests it with the real models**: it starts ComfyUI, asks it for every node and model
   the app uses, renders a short clip with sound, speaks a test line and asks the language model a question.

The screen stays short; everything the tools print goes to a log in `<home>/logs/`, and when a step fails its last
lines and the log's path are shown.

When it ends with `Promptwood is ready`, every part of the app works. Start it from the applications menu or with
`promptwood-gui`.

Nothing needs root and nothing is written outside two folders: the app's own folder and its *home*
(`~/.local/share/video-chain`, or `$VIDEO_CHAIN_HOME`), which holds ComfyUI, the models and the tools.
To remove everything, delete these two folders (and `~/.local/share/applications/promptwood.desktop`).

### Options

```bash
./install.sh --dry-run                # only show what would be installed and downloaded
./install.sh --profile core           # video clips only (40 GB)
./install.sh --profile full           # also extra speed modes, Krea-2, Qwen-Image-Edit 2511, MOSS-Image,
                                      # Fish S2 Pro voices and both head swaps (235 GB)
./install.sh --with image-krea2       # the standard set plus one part (also: image-edit, image-moss, tts-fish)
./install.sh --without tts,llm        # the standard set minus parts
./install.sh --models-from ~/ComfyUI  # a new ComfyUI that reads the models of the one you have, in place
./install.sh --hf-cache ~/.cache/huggingface/hub       # text to speech / language models you already have
./install.sh --comfy-dir ~/ComfyUI    # use the ComfyUI you already have (add --comfy-python if it has no .venv)
./install.sh --yes --accept-licenses  # no questions (for scripts)
VIDEO_CHAIN_HOME=/big/disk/promptwood ./install.sh    # everything on another disk
./install.sh --help                   # all options and all parts with their sizes
```

**Nothing you already have is downloaded again.** Setup looks into ComfyUI's model folders (also the ones from
its `extra_model_paths.yaml`, and the ones you name with `--models-from`) and into your Hugging Face cache; another
build of the same model (bf16 instead of int8, another turbo LoRA, your own GGUF language model with its vision
part) counts as well, exactly as the app itself would accept it. `./install.sh --dry-run` shows what would be
downloaded without changing anything. [docs/MODELS.md](docs/MODELS.md) lists every download with its source and licence.

A Hugging Face token is optional. Every model in the standard set downloads without one, but Hugging Face may slow
or limit anonymous downloads. With a free read token (https://huggingface.co/settings/tokens) they run smoother:
`HF_TOKEN=hf_… ./install.sh`, or log in once with `hf auth login`. Setup reads the token, sends it to
huggingface.co only (not to the storage servers downloads are redirected to) and never stores or prints it.

### Later

```bash
promptwood doctor                    # what is installed, what is missing, and the command that fixes it
promptwood doctor --render           # also render a real test clip
promptwood setup                     # install whatever is missing (safe to run again at any time)
promptwood setup --with headswap-wan # add a part
promptwood setup --steps comfyui     # repeat one install step (system, comfyui, nodes)
```

The same report is in the app under **Help > Setup check**.

## Use

- `promptwood-gui` opens the app. It starts ComfyUI itself when a clip is to be made (the Start button next to the
  ComfyUI status), or uses the one that is running.
- `promptwood start.jpg prompts.txt --out ~/chains/test1` renders a chain from the command line: one prompt per
  line, each clip starts from the last frame of the one before. `promptwood --help` lists the options.

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

## Responsible use

Promptwood can put a real person's face into a video, swap a head, and speak in a cloned voice. Use these only
with people who agreed to it:

- **Consent.** Use a person's face, likeness or voice only with their permission, and only for what they agreed to.
  This includes the pictures you give the cast, the voices you clone or record, and every face or head swap.
- **No harm.** Do not make content that impersonates, deceives, harasses, defames or sexualises a real person, that
  shows a minor in any sexual or harmful way, or that is meant to mislead people about real events (fake news,
  fraud, political deception).
- **Say that it is made with AI.** Label generated or altered videos, pictures and voices of real people as such
  when you publish them. Several laws require it (for example the EU AI Act for "deepfakes"), and so do many
  platforms.
- **Rights in what you put in.** Use pictures, videos, music and voices you are allowed to use (your own, licensed,
  or with permission).
- **The models' own rules.** Each model comes with its publisher's licence and use policy (see
  [docs/MODELS.md](docs/MODELS.md)); several forbid exactly the uses above, and some allow no commercial use at all.

You are responsible for what you make with Promptwood and for following the law where you live and publish. The
software comes without any warranty (see the licence).

## Licences of what gets installed

Promptwood installs programs and models made by others, and each keeps its own licence. **Many of the models are
not free for every use:** the MiniMax H3 video models come under MiniMax's community licence, Qwen-Image 2.1 under
a research licence, the InsightFace face models are for non-commercial research only, Fish S2 Pro (part
`tts-fish`, not in the standard set) under the Fish Audio Research License (research and non-commercial use only),
and for some files (the HyperSwap and GPEN face models) the publisher states no licence at all. The installer shows, per part, what each
publisher states and asks before it downloads; [docs/MODELS.md](docs/MODELS.md) has the full list with links.
Whether your use is allowed is yours to check; parts you do not want can be left out (`--without faces`).

## Licence

Promptwood is free for **non-commercial use** under the [PolyForm Noncommercial License 1.0.0](LICENSE.md):
personal projects, hobby, study and research, and non-profit, educational and public organisations may use, change
and share it. Its source is public, but it is not "open source" in the OSI sense: **commercial use** (in a business,
for clients, for monetised content) needs a separate licence from the author: ask through the project page,
https://github.com/mschindler83/promptwood.

The ComfyUI nodes in `comfy_nodes/video_chain_nodes/` run inside ComfyUI and use its code, so they are licensed under
the [GNU GPL 3.0](comfy_nodes/video_chain_nodes/LICENSE), as ComfyUI is.

A licence for Promptwood covers Promptwood only, never the models it uses: those keep their publishers' licences
(above), whatever your licence for the app.

## Contributing

Bug reports and ideas are welcome as issues. Code contributions can only be taken with a contributor agreement that
lets the author also offer them under the commercial licence; please ask in an issue before you start on one.
