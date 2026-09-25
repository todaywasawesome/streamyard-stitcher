# streamyard-stitcher

Rebuild a StreamYard cloud master with your high-res local tracks — keeping every layout switch, graphic, transition, and overlay from the web recording, but with the crisp video and audio that only your local files have.

StreamYard's cloud recording is the best *edit* (layouts, lower-thirds, branding) but it's a re-encoded, lower-bitrate master with compressed audio. Your local per-participant recordings are the best *source*. `streamyard-stitcher` combines them: it detects where each participant sits on screen in the cloud master, replaces those regions with the matching high-res local feed, and re-aligns the high-res local audio.

```
 input/                         output/
 ┌────────────────────────┐      ┌────────────────────────┐
 │ master.mp4 (cloud)     │      │                        │
 │ Dan-…-webcam-…mp4  ────┼─────►│  master.mp4            │
 │ Dan-…-screen-…mp4      │      │  cloud layout + gfx    │
 │ Guest-…-webcam-…mp4    │      │  high-res participant  │
 │ …                      │      │  regions + hi-res audio│
 └────────────────────────┘      └────────────────────────┘
```

## Requirements

- **Rust** (stable, 1.82+)
- **ffmpeg / ffprobe** on your `PATH` (decoding and encoding are delegated to the installed binary — no Rust video crates). Homebrew's `ffmpeg` is fine.

## Install

```sh
cargo build --release          # binary at target/release/streamyard-stitcher
# or, from this repository:
cargo install --path .
# or grab a prebuilt binary from the Releases page.
```

## Quick start

Point it at the folder StreamYard downloaded (the cloud master + every local track, all in one directory) and name the output:

```sh
streamyard-stitcher \
  -i "input/My Show - Episode 12" \
  -o output/master.mp4
```

Useful variations:

```sh
# Quick validation encode of just the first 400 seconds
streamyard-stitcher -i <dir> -o /tmp/test.mp4 --limit 400

# Hardware encoding (VideoToolbox on macOS, NVENC on Linux)
streamyard-stitcher -i <dir> -o output/master.mp4 --gpu

# Inspect what it would do — prints the detected layout timeline and the
# exact ffmpeg commands, without encoding anything
streamyard-stitcher -i <dir> -o /tmp/test.mp4 --dry-run

# Debug the tile→participant matching at one moment in time
streamyard-stitcher -i <dir> -o /tmp/x.mp4 --debug-identity 660
```

### Flags

| Flag | Default | Meaning |
|---|---|---|
| `-i, --input DIR` | — | Directory with the cloud master + local track files |
| `-o, --output FILE` | — | Destination for the upgraded master |
| `--gpu` | off | Hardware encoder (VideoToolbox / NVENC) |
| `--limit SECS` | full length | Cap the output length — cheap way to test a run |
| `--dry-run` | off | Print layout + ffmpeg commands, encode nothing |
| `--samples N` | auto (~40) | Timeline samples for layout detection |
| `--width PX` | 1280 | Frame width for detection (higher keeps thin PIP borders sharp) |
| `--layout FILE` | auto-detect | Use a pre-made layout JSON instead of detecting |
| `--debug-identity TIME` | — | Print detected tiles + similarity matrix at `TIME`, then exit |

## How it works

1. **Classify** — the file without a `-webcam`/`-screen` token is the cloud master; the rest are participants. StreamYard's `HHh_MMm_SSs_mmmms` filename token seeds the expected offset.
2. **Sync** — decode every track to raw PCM and find each local track's offset in cloud time via normalized cross-correlation (with a fine second pass for sub-second accuracy and drift estimation).
3. **Detect layout** — sample frames across the timeline, detect participant tiles (content boxes that persist), and match each tile to a participant by visual similarity. Brief mis-detections are filtered by layout-stability voting.
4. **Encode** — the timeline is split into *windows* (maximal intervals where the on-screen tile set is constant). Each window is one small ffmpeg command: the cloud master with that window's local feeds cover-fit into their boxes, plus the restored chrome. Windows are concatenated without re-encoding. Batching keeps peak memory bounded by the busiest single window — one giant filtergraph OOMs.
5. **Audio** — the high-res local tracks are delayed to their alignment offsets and mixed (un-normalized, behind a limiter) over the video, replacing the cloud's compressed audio.

**Preserved chrome:** the fixed web overlays that a full-frame feed would cover are restored from the cloud master itself — the bottom subscribe band (blue pill, text included) and the top-right mascot badge. Both are measured from frames and composited as one alpha mask on top of every window.

## Notes & limitations

- **Name bars / lower-thirds are not restored.** They move and carry arbitrary text, so a fixed-region mask can't recover them — a full-frame local feed simply covers them. The subscribe band and mascot badge are fixed-position chrome and *are* preserved.
- The cloud master and local tracks must come from the same StreamYard session (matching naming tokens).
- `--gpu` uses `h264_videotoolbox` on macOS and `h264_nvenc` on Linux; otherwise `libx264 -crf 20`.
- Detection runs at 1280px by default; raise `--width` if stacked PIPs merge.

## Development

```sh
cargo test          # unit tests (pure logic — no ffmpeg needed)
cargo clippy --all-targets
cargo fmt --check
```

Tests cover the pure core: window building (gap-fill, clipping), the ffmpeg command builders — including the two regressions that cost hours (`-t` must be an *output* option in window commands, and the final mux must *always* cap duration, since the `apad`-ed audio is otherwise infinite) — plus the PPM parser and filename offset parsing.

CI (`.github/workflows/ci.yml`) runs fmt + clippy (deny warnings) + tests on Ubuntu and macOS. Pushing a `v*` tag builds per-platform binaries and publishes a GitHub Release (`.github/workflows/release.yml`).

See [DESIGN.md](DESIGN.md) for the original design doc and phase plan.
