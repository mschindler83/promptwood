"""The image dialog's prompt: variations glue, seeds and the language-model refiner.

A mixin of ImageEditDialog (ui/image_edit.py): the widgets it uses are created there."""
from __future__ import annotations

import inspect
import json
import random
import re
from collections.abc import Mapping
from typing import Any

from ..variations import MAX_IMAGES, Value, Variable

from .dialogs import confirm as ask
from . import settings_keys as SK
from .image_common import MAX_RESULTS, SEED_MAX
from .image_picker import pick_image
from .image_host import ImageEditHost
from .prompt_refiner import replace_text



class PromptMixin(ImageEditHost):
    # ---- prompt
    def _negative_used(self) -> bool:
        """The negative prompt goes with the next Generate (create mode, its field shown for this model)."""
        return self.mode() == "create" and self.negative.isVisibleTo(self)

    def _templates_changed(self):
        self.variations.set_templates(self.prompt.toPlainText(),
                                      self.negative.text() if self._negative_used() else "")

    def _variations_changed(self):
        if self.qs is not None:
            st = self.variations.state()
            self.qs.setValue(SK.PROMPT_VARIATIONS, json.dumps(st, ensure_ascii=False))
        if not self._rehighlighting:
            self._rehighlighting = True
            try:
                self._highlight.rehighlight()
            finally:
                self._rehighlighting = False
        if not self._built:
            return
        active = self.variations.active()
        self.tries.setEnabled(not active)
        self.tries.setToolTip("Variations decide how many images are made (Make … below the prompt)" if active
                              else f"How many results at once (1 … {MAX_RESULTS}; each with the next seed), to "
                                   "pick the best one. They are made one after another and each appears when it "
                                   "is ready; Stop ends it after the current one")
        per = len(self._lora_runs())
        if self.batch and self.mode() == "edit":
            per *= len(self.batch)
        n = (self.variations.amount() if active else self.tries.value()) * per
        self.b_gen.setText(f"Generate {n} image{'s' if n != 1 else ''}" if active or per > 1 or self.batch
                           else "Generate")

    def _insert_placeholder(self, name: str):
        cur = self.prompt.textCursor()
        before = self.prompt.toPlainText()[:cur.position()]
        cur.insertText(("" if not before or before[-1].isspace() else " ") + "{" + name + "}")
        self.prompt.setTextCursor(cur)
        self.prompt.setFocus()

    def _rename_placeholder(self, old: str, new: str):
        pat = re.compile(r"\{(\??)" + re.escape(old) + r"([|}])")
        replace_text(self.prompt, pat.sub(lambda m: "{" + m.group(1) + new + m.group(2), self.prompt.toPlainText()))
        self.negative.setText(pat.sub(lambda m: "{" + m.group(1) + new + m.group(2), self.negative.text()))

    def _pick_list(self) -> str | None:
        return pick_image(self, "Choose a wildcard list (.txt, one value per line)",
                          self.qs, kind="text")

    def _variation_jobs(self) -> list | None:
        """The next Generate's prompts when variations are active (None: not active; [] after a refusal)."""
        if not self.variations.active():
            return None
        self._templates_changed()
        many = self.variations.too_many()
        if many and not ask(self, "Variations", f"Every combination makes {many:,} images"
                                    + (f" (one Generate makes at most {MAX_IMAGES:,})" if many > MAX_IMAGES
                                       else "") + ". Make them all?"):
            return []
        jobs = self.variations.jobs()
        key = self.variations.position_key()
        for j in jobs:                                     # (as the plain prompt: one line, single spaces)
            j.prompt, j.negative, j.key = " ".join(j.prompt.split()), " ".join(j.negative.split()), key
        return jobs

    def new_seed(self):
        """A fresh random seed in the field."""
        self.seed.setValue(random.randint(1, SEED_MAX))

    def _seeds(self, n: int) -> list[int]:
        """The seeds of the next n results (the field's, +1 ...); a fresh one in the field afterwards if wanted."""
        first = self.seed.value()
        seeds = [(first - 1 + k) % SEED_MAX + 1 for k in range(n)]
        if self.fresh_seed.isChecked():
            self.new_seed()
        return seeds

    def _use_example(self, text: str):
        """An example prompt from the LoRA window replaces the prompt (one step Ctrl+Z takes back)."""
        replace_text(self.prompt, text)

    def prompt_kind(self) -> str:
        return "image" if self.mode() == "create" else "edit"

    def _refine_prompt(self):
        if self.refine is None:
            return
        params: Mapping[str, Any]
        try:
            params = inspect.signature(self.refine).parameters
        except (TypeError, ValueError):
            params = {}
        more = {"pictures": self._refine_pictures()} if "pictures" in params else {}
        if "loras" in params:                      # the refiner is told what is known of the chosen LoRAs
            more["loras"] = [it["path"] for it in self.loras.active_items()]
        new = self.refine(self.prompt.toPlainText(), self.prompt_kind(), **more)
        if new is None:
            return
        if new.ideas:
            self.use_ideas(new.ideas)
        elif new.text:
            replace_text(self.prompt, new.text)

    def use_ideas(self, prompts: list[str]):
        """All ideas from the refiner as one Generate: the prompt becomes {idea}, its values the ideas, every
        combination once (Ctrl+Z in the prompt brings the bullet points back)."""
        p = self.variations
        name, k = "idea", 1
        while (var := p.variable(name)) is not None and [v.text for v in var.values] != prompts:
            k += 1                                  # (an {idea} with other values stays: {idea2} …)
            name = f"idea{k}"
        replace_text(self.prompt, "{" + name + "}")
        p.vars = [v for v in p.vars if v.name != name] + [Variable(name, [Value(t) for t in prompts])]
        p.on.setChecked(True)
        p.mode_buttons["order"].setChecked(True)
        p.repeat.setValue(1)
        p.all.setChecked(True)
        self._templates_changed()
        p._rebuild()
        self.state.setText(f"{len(prompts)} ideas are the values of {{{name}}} in Variations: Generate makes one image "
                           "each. Ctrl+Z in the prompt brings your bullet points back ({" + name + "} stays in "
                           "Variations until you remove it).")

    def _refine_pictures(self) -> list[tuple[str, str, bool]]:
        """What the refiner's model may see: the image to edit and the extra pictures (edit), the reference
        pictures (create), and the selected result (off at first)."""
        out = []
        if self.mode() == "edit":
            if self.source:
                out.append(("The image to edit (Picture 1)", self.source, True))
            for k, p in ((2, self.pic2.path), (3, self.pic3.path)):
                if p:
                    out.append((f"Picture {k}", p, True))
        elif self.use_refs.isChecked():
            out += [(f"Reference picture {k + 1}", s.path, True) for k, s in enumerate(self.cpics) if s.path]
        sel = self.selected()
        if sel is not None:
            out.append(("A result made here", sel[0], False))
        return out
