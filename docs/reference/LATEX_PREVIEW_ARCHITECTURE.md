# LaTeX preview architecture

The OpenGL GUI renders equation, boundary, and domain-shape previews **in-process** without spawning Python or matplotlib.

## Data flow

1. Panel text input sets `LatexTexture::dirty` and updates `last_edit`.
2. Each frame, `UpdateLatexTexture` → `LatexRenderService::RequestRender` (250 ms debounce).
3. A background worker calls `RenderLatexToBitmap` (CPU mathtext layout via FreeType/STB; optional MicroTeX when `USE_MICROTEX_LATEX_PREVIEW=ON`).
4. The main thread `PollCompletedRenders` uploads RGBA with `UploadTextureFromRGBA` (OpenGL).
5. `DrawLatexPreview` displays the texture in ImGui.

Semantic validation for solves remains in `LatexParser` (`validation.cpp` / equation panel parse badge). Preview is WYSIWYG on the user string.

## Threading

| Thread | Work |
|--------|------|
| GUI | Debounce, queue jobs, GL upload, ImGui draw |
| Worker | Parse/layout/rasterize to CPU `RGBA8` buffer |

OpenGL calls occur **only** on the GUI thread.

## Cache

In-memory cache keyed by `(source, fg_hex, font_size)`. Changing global font size in preferences invalidates the cache via `SetGlobalStyle`.

## MicroTeX backend (default when enabled)

When built with `USE_MICROTEX_LATEX_PREVIEW=ON` (requires `tinyxml2` and the `third_party/MicroTeX` submodule):

- `gui_gl/latex/microtex_rgba_graphic.cpp` implements MicroTeX `Graphics2D` / `Font` / `TextLayout` into an RGBA buffer.
- `MicroTeXInit` loads resources from `res/` next to the executable (copied at build time from `third_party/MicroTeX/res`).
- Fractions (`\frac`), functions (`\sin`, `\cos`), nested scripts, and general bracketed expressions render with proper layout.
- Preview text uses a sans-serif bold system font when available (Arial Bold on macOS, then Comic Sans MS Bold / DejaVu Sans Bold), with bundled Computer Modern Sans Bold as fallback.

If MicroTeX fails to parse or draw, the mathtext CPU fallback runs (no matplotlib).

## Failure modes

- Render failure: monospace raw LaTeX + red render error (`DrawLatexPreviewError`). No matplotlib fallback.
- Parse failure: yellow/red parse badge; preview may still show if layout succeeded.
- Mathtext fallback only: install Arial Unicode or DejaVu Sans (see `mathtext_bitmap_renderer.cpp` search paths).
- MicroTeX disabled at configure time: install `tinyxml2` (`brew install tinyxml2` on macOS) and reconfigure with `-DUSE_MICROTEX_LATEX_PREVIEW=ON`.

## Legacy tooling

`tools/render_latex.py` is retained for offline debugging only; the GUI no longer calls it.

## Related docs

- [LATEX_TOKEN_REGISTRY.md](LATEX_TOKEN_REGISTRY.md) — supported tokens and normalizations
- `gui_gl/latex/latex_render_service.h` — public API
