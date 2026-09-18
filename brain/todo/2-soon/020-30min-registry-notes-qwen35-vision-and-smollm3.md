# T-020 — Two registry notes nobody has written

**Effort:** 30 min · **Produces:** `models.json` note fields ·
**Area:** `models.json`

Menu order and the `notes` field *are* the recommendation — that is the lesson
from the small-model slot, where sorting `small_model: true` to the top was
half the fix.

- [ ] **Qwen3.5-4B vision verdict.** It has been exercised on the 140V for tool
      calling; the registry says nothing about how it reads images.
- [ ] **SmolLM3 entries should mention thinking mode and `/no_think`.** The
      INT4 entry already carries the "NOT for a small_model slot" warning; the
      INT8 entry does not, and neither says the toggle exists.

## Done when

Someone picking from the menu can tell which models think, and what that costs
them on a side-request slot.
