"""video-chain setup / video-chain doctor (also: python -m video_chain.setup …)."""
from __future__ import annotations

import argparse
import sys

from .. import paths
from . import manifest as M


def _list(text: str) -> list[str]:
    return [t.strip() for t in text.split(",") if t.strip()]


def build_parser() -> argparse.ArgumentParser:
    man = M.load()
    parts = "\n".join(f"  {cid:14} {c['bytes'] / 2**30:6.1f} GB  [{c['profile']}]  {c['title']}"
                      for cid, c in man["components"].items())
    p = argparse.ArgumentParser(prog="video-chain", formatter_class=argparse.RawDescriptionHelpFormatter,
                                description="Install and check everything Promptwood needs.",
                                epilog=f"Parts (for --only / --with / --without):\n{parts}\n\n"
                                       "Steps for --only: system (ffmpeg), comfyui, nodes\n"
                                       f"Everything goes into {paths.home()} (set VIDEO_CHAIN_HOME for another "
                                       "place).\nA token for licence-gated downloads is read from HF_TOKEN.")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup", help="install what is missing (safe to run again)",
                       formatter_class=argparse.RawDescriptionHelpFormatter, epilog=p.epilog)
    s.add_argument("--profile", choices=M.PROFILES, default="standard",
                   help="core: " + M.PROFILE_TEXT["core"] + "; standard: " + M.PROFILE_TEXT["standard"]
                        + "; full: " + M.PROFILE_TEXT["full"])
    s.add_argument("--only", type=_list, default=[], metavar="NAMES", help="only these parts / steps (comma list)")
    s.add_argument("--with", dest="add", type=_list, default=[], metavar="NAMES", help="also these parts")
    s.add_argument("--without", type=_list, default=[], metavar="NAMES", help="leave these parts out")
    s.add_argument("--comfy-dir", help="use the ComfyUI in this folder instead of installing one")
    s.add_argument("--comfy-python", help="the Python that runs that ComfyUI (when it has no .venv)")
    s.add_argument("--comfy-launcher", help="the command that starts that ComfyUI (default: one is written)")
    s.add_argument("--models-from", action="append", default=[], metavar="DIR",
                   help="models you already have: another ComfyUI's folder (or a models folder). The new ComfyUI "
                        "reads them in place; only what is missing there is downloaded. Can be given several times")
    s.add_argument("--hf-cache", metavar="DIR", help="use this Hugging Face cache (text to speech and language "
                                                     "models you already have) instead of a new one")
    s.add_argument("--comfy-port", type=int, help="ComfyUI's port (default 8188)")
    s.add_argument("--torch-index", help="PyTorch package index (default: chosen from the graphics driver; "
                                         "'pypi' for the standard one)")
    s.add_argument("--yes", "-y", action="store_true", help="do not ask before starting")
    s.add_argument("--accept-licenses", action="store_true", help="accept the models' licences without being asked")
    s.add_argument("--force", action="store_true", help="install even when the graphics card or disk check fails")
    s.add_argument("--no-test", dest="test", action="store_false",
                   help="skip the test clip, test line and language model test at the end")
    s.add_argument("--no-desktop", dest="desktop", action="store_false", help="no menu entry, no ~/.local/bin links")
    s.add_argument("--workers", type=int, default=4, help="downloads at the same time (default 4)")
    s.add_argument("--dry-run", action="store_true", help="show the plan and stop")
    d = sub.add_parser("doctor", help="report what is installed and what is missing")
    d.add_argument("--deep", action="store_true", help="also start ComfyUI and ask it for its nodes and models")
    d.add_argument("--render", action="store_true", help="also render one short real clip (implies --deep)")
    d.add_argument("--speech", action="store_true", help="also speak one test line with the local text to speech")
    d.add_argument("--llm", action="store_true", help="also ask the llama.cpp language model one question")
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    paths.use_private_tools()
    if a.cmd == "setup":
        from .installer import Options, main_setup
        return main_setup(Options(profile=a.profile, only=a.only, add=a.add, without=a.without,
                                  comfy_dir=a.comfy_dir, comfy_python=a.comfy_python,
                                  comfy_launcher=a.comfy_launcher, comfy_port=a.comfy_port, models_from=a.models_from,
                                  hf_cache=a.hf_cache,
                                  torch_index=a.torch_index, yes=a.yes, accept_licenses=a.accept_licenses,
                                  force=a.force, test=a.test, desktop=a.desktop, workers=a.workers,
                                  dry_run=a.dry_run))
    from . import doctor as D
    checks = D.check_all()
    parts = D.installed_components() or M.select(M.load(), "standard")
    say = lambda t: print("  " + t, flush=True)      # noqa: E731
    if (a.deep or a.render) and not D.failed(checks):
        url = paths.comfy_url()
        try:
            with D.ComfySession(url, log=say):
                checks += D.deep_checks(url, parts)
                if a.render and not D.failed(checks):
                    checks.append(D.render_test(url, log=say))
        except Exception as e:  # noqa: BLE001
            checks.append(D.Check("ComfyUI (running)", "Start ComfyUI and ask it", D.FAIL, str(e)))
    if a.speech and not D.failed(checks):
        checks.append(D.speech_test(log=say))
    if a.llm and not D.failed(checks):
        checks.append(D.llm_test(log=say))
    print(D.report(checks, colour=sys.stdout.isatty()))
    print()
    print(D.summary(checks))
    return 1 if D.failed(checks) else 0


if __name__ == "__main__":
    sys.exit(main())
