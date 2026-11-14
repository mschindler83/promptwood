"""Problems with the sound references a clip names."""
from __future__ import annotations

from pathlib import Path

from .model import Tree, short_id
from .timeline import full_prompt
from .cast import audio_mentioned, audio_refs


def audio_problems(project: Path, tree: Tree, todo: list[str]) -> list[str]:
    """Why clips to be made cannot get the sounds their prompts name (one message per problem)."""
    refs = audio_refs(tree)
    out = []
    for nid in todo:
        node = tree.nodes[nid]
        if node.is_video:
            continue
        n = audio_mentioned(full_prompt(tree.options, node.prompt))
        if n > len(refs):
            out.append(f"clip #{short_id(nid)} names <Audio {n}>, but there {'is' if len(refs) == 1 else 'are'} only "
                       f"{len(refs)} sound reference{'' if len(refs) == 1 else 's'}")
        elif any(not (Path(project) / a["file"]).is_file() for a in refs[:n]):
            out.append(f"a sound reference of clip #{short_id(nid)} is missing in refs/")
    return out
