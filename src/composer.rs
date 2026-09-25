//! Phase 4/5 — turn the layout timeline into a *batched* set of small ffmpeg
//! commands and run them sequentially.
//!
//! Why batch: a single filter graph holding every local feed open at once plus
//! one overlay node per spatial segment (100+ nodes) buffers far more than the
//! machine has RAM — macOS jetsam SIGKILLs it mid-encode. Instead we render the
//! timeline as a sequence of **windows** (maximal intervals where the on-screen
//! tile set is constant). Each window's ffmpeg only opens the cloud master and
//! the 1–4 local feeds actually visible there, with just that window's overlays.
//! Peak memory is bounded by the busiest single window (5 decoders + 4 overlays),
//! not by the whole timeline. The window clips are concatenated, and the
//! high-res audio (cheap — no video decode) is mixed and muxed on top.

use std::fmt::Write as _;

use crate::model::{LayoutTimeline, Participant};

/// One maximal interval where the set of visible local tiles is constant.
#[derive(Debug)]
pub struct Window {
    pub start: f64,
    pub end: f64,
    /// `(participant index into `participants`, bounding box (x, y, w, h))`,
    /// in composite (bottom-to-top) order.
    pub tiles: Vec<(usize, (u32, u32, u32, u32))>,
}

/// Pick the encoder for a target platform; `gpu=true` prefers hardware accel.
fn video_codec(gpu: bool) -> (String, Vec<String>) {
    let os = std::env::consts::OS;
    let mk = |v: Vec<&str>| v.into_iter().map(str::to_string).collect::<Vec<_>>();
    if gpu {
        match os {
            "macos" => ("h264_videotoolbox".to_string(), mk(vec!["-b:v", "12M"])),
            "linux" => (
                "h264_nvenc".to_string(),
                mk(vec!["-preset", "p5", "-cq", "23"]),
            ),
            _ => (
                "libx264".to_string(),
                mk(vec!["-preset", "medium", "-crf", "20"]),
            ),
        }
    } else {
        (
            "libx264".to_string(),
            mk(vec![
                "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
            ]),
        )
    }
}

/// Group the spatial segments into constant-layout windows covering `[0, end)`,
/// inserting pass-through (no-tile) windows for gaps such as the intro, and
/// clipping everything to `end` (the `--limit` or cloud duration).
pub fn build_windows(
    timeline: &LayoutTimeline,
    participants: &[Participant],
    end: f64,
) -> Vec<Window> {
    // Keep valid, in-order (start, end, participant, box) rows.
    type Seg = (f64, f64, usize, (u32, u32, u32, u32));
    let mut segs: Vec<Seg> = Vec::new();
    for s in &timeline.segments {
        let pi = match participants.iter().position(|p| p.id == s.participant_id) {
            Some(i) => i,
            None => continue, // unknown participant; skip
        };
        let (x, y, w, h) = s.bounding_box;
        if w == 0 || h == 0 {
            continue;
        }
        segs.push((s.start_time, s.end_time, pi, (x, y, w, h)));
    }
    segs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Segments sharing a (start, end) belong to the same window.
    let mut grouped: Vec<Window> = Vec::new();
    for (st, en, pi, box_) in segs {
        if let Some(w) = grouped.last_mut() {
            if (w.start - st).abs() < 1e-6 && (w.end - en).abs() < 1e-6 {
                w.tiles.push((pi, box_));
                continue;
            }
        }
        grouped.push(Window {
            start: st,
            end: en,
            tiles: vec![(pi, box_)],
        });
    }

    // Lay windows over [0, end), filling gaps with pass-through and clipping.
    let mut out: Vec<Window> = Vec::new();
    let mut cursor = 0.0f64;
    for w in grouped {
        if w.start >= end {
            break;
        }
        let ws = w.start.max(cursor);
        let we = w.end.min(end);
        if we <= ws {
            continue;
        }
        if ws > cursor + 1e-6 {
            out.push(Window {
                start: cursor,
                end: ws,
                tiles: vec![],
            });
        }
        out.push(Window {
            start: ws,
            end: we,
            tiles: w.tiles,
        });
        cursor = we;
    }
    if cursor < end - 1e-6 {
        out.push(Window {
            start: cursor,
            end,
            tiles: vec![],
        });
    }
    out
}

/// ffmpeg command that renders one window to a video-only clip: the cloud master
/// cut to `[start, end)` with the window's local feeds cover-fit into their boxes,
/// and (optionally) the subscribe-band mask composited on top.
///
/// No `enable='between(…)'` gating is needed — the whole clip *is* one window, so
/// every tile is visible for its full duration. Both the cloud and each local feed
/// are input-seeked (`-ss`) to the matching time, so their PTS start together and
/// no `setpts` shift is required.
pub fn video_window_cmd(
    cloud: &str,
    participants: &[Participant],
    window: &Window,
    overlay: Option<&str>,
    out_clip: &str,
    gpu: bool,
) -> Vec<String> {
    let dur = window.end - window.start;
    let n = window.tiles.len();

    let mut cmd: Vec<String> = vec!["ffmpeg".into(), "-y".into()];
    // input 0: cloud, seeked to this window's start. NOTE: the duration cap is an
    // *output* option below, not here — `-t` as an input option is defeated by the
    // `-loop 1` overlay input (ffmpeg keeps the graph alive off the infinite
    // overlay and encodes the whole master instead of the window).
    cmd.extend([
        "-ss".into(),
        fmt_t(window.start),
        "-i".into(),
        cloud.to_string(),
    ]);
    // inputs 1..=n: this window's local feeds, seeked to the matching local time
    for (pi, _) in &window.tiles {
        let p = &participants[*pi];
        let seek = (window.start - p.offset_secs.unwrap_or(0.0)).max(0.0);
        cmd.extend(["-ss".into(), fmt_t(seek), "-i".into(), p.file.clone()]);
    }
    // input n+1: the looped subscribe-band mask (only when we have one)
    let ovl_input = overlay.map(|_| 1 + n);
    if let Some(o) = overlay {
        cmd.extend([
            "-loop".into(),
            "1".into(),
            "-framerate".into(),
            "30".into(),
            "-i".into(),
            o.to_string(),
        ]);
    }

    let mut g = String::new();
    let _ = write!(g, "[0:v]format=yuv420p,setsar=1[b0];");
    for (i, (_pi, (x, y, w, h))) in window.tiles.iter().enumerate() {
        let in_idx = 1 + i;
        let next_base = i + 1;
        // Cover-fit: scale to fill the box preserving aspect, centre-crop overflow.
        let _ = write!(
            g,
            "[{in_idx}:v]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},setsar=1[t{i}];[b{i}][t{i}]overlay=x={x}:y={y}[b{next_base}];"
        );
    }

    let final_label = match ovl_input {
        Some(ovl) => {
            let _ = write!(g, "[b{n}][{ovl}:v]overlay=x=0:y=0[outv];");
            "outv".to_string()
        }
        None => format!("b{n}"),
    };

    cmd.push("-filter_complex".into());
    cmd.push(g);
    cmd.push("-map".into());
    cmd.push(format!("[{final_label}]"));
    cmd.push("-an".into());
    // Cap the output to this window's duration. Must be an *output* option (after
    // the inputs) so the `-loop 1` overlay can't extend the clip past it.
    cmd.push("-t".into());
    cmd.push(fmt_t(dur));
    let (codec, extra) = video_codec(gpu);
    cmd.push("-c:v".into());
    cmd.push(codec);
    cmd.extend(extra);
    cmd.push(out_clip.to_string());
    cmd
}

/// ffmpeg command that concatenates the window clips (same codec/res/fps, each
/// starting on a keyframe) without re-encoding. `list_file` holds one
/// `file '<path>'` line per clip, in order.
pub fn concat_cmd(list_file: &str, out_video: &str) -> Vec<String> {
    vec![
        "ffmpeg".into(),
        "-y".into(),
        "-f".into(),
        "concat".into(),
        "-safe".into(),
        "0".into(),
        "-i".into(),
        list_file.to_string(),
        "-c".into(),
        "copy".into(),
        out_video.to_string(),
    ]
}

/// ffmpeg command that muxes the concatenated video (stream-copied) with the
/// high-res audio: each local track is shifted to its alignment offset and mixed
/// un-normalized behind a limiter, replacing the lower-bitrate cloud audio.
///
/// `duration_secs` MUST be passed: the audio inputs are `apad`-ed (infinite
/// silence tail) so `amix=duration=first` can never terminate them — without an
/// explicit `-t` the mux encodes silence forever past the end of the video.
pub fn final_mux_cmd(
    video: &str,
    participants: &[Participant],
    audio_idxs: &[usize],
    output: &str,
    duration_secs: f64,
) -> Vec<String> {
    let mut cmd: Vec<String> = vec!["ffmpeg".into(), "-y".into()];
    cmd.extend(["-i".into(), video.to_string()]);
    for &i in audio_idxs {
        cmd.extend(["-i".into(), participants[i].file.clone()]);
    }

    let mut g = String::new();
    for (k, &i) in audio_idxs.iter().enumerate() {
        let in_idx = 1 + k;
        let off_ms = (participants[i].offset_secs.unwrap_or(0.0) * 1000.0).round() as i64;
        let _ = write!(g, "[{in_idx}:a]aformat=channel_layouts=mono,asetpts=PTS-STARTPTS,adelay={off_ms},apad[aud{k}];");
    }
    let mix_inputs: String = audio_idxs
        .iter()
        .enumerate()
        .map(|(k, _)| format!("[aud{k}]"))
        .collect();
    let _ = write!(
        g,
        "{mix_inputs}amix=inputs={n}:duration=first:normalize=0,alimiter=limit=0.95[aout];",
        n = audio_idxs.len()
    );

    cmd.push("-filter_complex".into());
    cmd.push(g);
    cmd.push("-map".into());
    cmd.push("0:v".into());
    cmd.push("-map".into());
    cmd.push("[aout]".into());
    cmd.push("-c:v".into());
    cmd.push("copy".into());
    cmd.push("-c:a".into());
    cmd.push("aac".into());
    cmd.push("-b:a".into());
    cmd.push("256k".into());
    cmd.push("-movflags".into());
    cmd.push("+faststart".into());
    // Always cap the output: the apad-ed inputs are infinite, so `-t` is the only
    // thing that terminates the mux at the end of the video.
    cmd.push("-t".into());
    cmd.push(fmt_t(duration_secs));
    cmd.push(output.to_string());
    cmd
}

/// Format a time for `-ss`/`-t` with enough precision to stay frame-accurate.
fn fmt_t(t: f64) -> String {
    format!("{t:.3}")
}

/// Run a command, streaming ffmpeg's stderr for progress/health.
pub fn run_command(cmd: &[String]) -> anyhow::Result<()> {
    let cmdline = cmd.join(" ");
    tracing::info!("running: {cmdline}");
    let mut child = std::process::Command::new(&cmd[0])
        .args(&cmd[1..])
        .stderr(std::process::Stdio::piped())
        .spawn()?;
    let stderr = child.stderr.as_mut().expect("stderr was piped at spawn");
    use std::io::Read;
    let mut buf = [0u8; 4096];
    let mut last: Vec<u8> = Vec::new(); // bounded tail of stderr, for progress + failure
    loop {
        let n = stderr.read(&mut buf)?;
        if n == 0 {
            break;
        }
        last.extend_from_slice(&buf[..n]);
        if last.len() > 8192 {
            last.drain(..last.len() - 8192);
        }
        // Surface the most recent progress line (ffmpeg writes \r-separated).
        let tail = match last.iter().rposition(|b| *b == b'\r' || *b == b'\n') {
            Some(p) => &last[p + 1..],
            None => &last[..],
        };
        let s = String::from_utf8_lossy(tail);
        if let Some(pos) = s.rfind("time=") {
            tracing::info!("[encode] {}", s[pos..].trim());
        }
    }
    let status = child.wait()?;
    if !status.success() {
        // Surface the tail of ffmpeg's stderr — without this a killed ffmpeg is
        // just "exit status: N" with no cause.
        let tail = String::from_utf8_lossy(&last);
        anyhow::bail!("ffmpeg failed with {status}\n{tail}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{LayoutTimeline, SpatialSegment};

    fn part(id: &str, off: f64) -> Participant {
        Participant {
            id: id.into(),
            name: id.into(),
            source: "webcam".into(),
            file: format!("/{id}.mp4"),
            duration: 9999.0,
            offset_secs: Some(off),
            drift_per_sec: None,
            confidence: None,
        }
    }

    fn tl(segs: Vec<SpatialSegment>) -> LayoutTimeline {
        LayoutTimeline {
            cloud: "c.mp4".into(),
            width: 1920,
            height: 1080,
            duration: 100.0,
            segments: segs,
        }
    }

    fn seg(p: &str, st: f64, en: f64, box_: (u32, u32, u32, u32)) -> SpatialSegment {
        SpatialSegment {
            start_time: st,
            end_time: en,
            participant_id: p.into(),
            bounding_box: box_,
        }
    }

    #[test]
    fn windows_group_shared_bounds_and_fill_gaps() {
        let ps = vec![part("a", 0.5), part("b", 1.0)];
        let t = tl(vec![
            seg("a", 10.0, 40.0, (0, 0, 960, 1080)),
            seg("b", 10.0, 40.0, (960, 0, 960, 1080)),
            seg("a", 50.0, 80.0, (0, 0, 1920, 1080)),
        ]);
        let w = build_windows(&t, &ps, 100.0);
        assert_eq!(w.len(), 5, "{w:?}");
        assert_eq!(
            (w[0].start, w[0].end, w[0].tiles.len()),
            (0.0, 10.0, 0),
            "intro pass-through"
        );
        assert_eq!(w[1].tiles.len(), 2, "two tiles share the 10-40 window");
        assert_eq!(
            (w[2].start, w[2].end, w[2].tiles.len()),
            (40.0, 50.0, 0),
            "gap pass-through"
        );
        assert_eq!(w[3].tiles.len(), 1);
        assert_eq!(
            (w[4].start, w[4].end, w[4].tiles.len()),
            (80.0, 100.0, 0),
            "tail pass-through"
        );
    }

    #[test]
    fn windows_clip_to_end_and_skip_invalid() {
        let ps = vec![part("a", 0.5)];
        let t = tl(vec![
            seg("a", 90.0, 200.0, (0, 0, 1920, 1080)), // runs past the clip
            seg("zz", 10.0, 20.0, (0, 0, 100, 100)),   // unknown participant
            seg("a", 30.0, 40.0, (0, 0, 0, 100)),      // zero-width box
        ]);
        let w = build_windows(&t, &ps, 100.0);
        assert_eq!(w.len(), 2, "{w:?}");
        assert_eq!(
            (w[1].start, w[1].end, w[1].tiles.len()),
            (90.0, 100.0, 1),
            "clipped to end"
        );
    }

    #[test]
    fn video_window_t_is_an_output_option() {
        let ps = vec![part("a", 0.5)];
        let w = Window {
            start: 100.0,
            end: 130.0,
            tiles: vec![(0, (10, 20, 320, 180))],
        };
        let cmd = video_window_cmd("cloud.mp4", &ps, &w, Some("mask.png"), "clip.mp4", false);
        // Regression: as an *input* option, `-t` is defeated by the `-loop 1` mask
        // input and the whole master gets encoded instead of the window.
        let last_i = cmd.iter().rposition(|a| a == "-i").unwrap();
        let t = cmd.iter().position(|a| a == "-t").unwrap();
        assert!(t > last_i, "-t must come after every input: {cmd:?}");
        assert!(cmd.iter().any(|a| a == "30.000"), "duration: {cmd:?}");
        assert!(
            cmd.iter().any(|a| a == "99.500"),
            "local seek 100-0.5: {cmd:?}"
        );
        assert!(cmd.iter().any(|a| a == "-an"));
        let g = cmd.iter().find(|a| a.starts_with("[0:v]")).unwrap();
        assert!(g.contains("overlay=x=10:y=20"), "{g}");

        let cmd2 = video_window_cmd("cloud.mp4", &ps, &w, None, "clip.mp4", false);
        assert!(
            !cmd2.iter().any(|a| a == "-loop"),
            "no mask input without an overlay"
        );
    }

    #[test]
    fn final_mux_always_caps_duration() {
        let ps = vec![part("a", 0.5), part("b", 2.25)];
        // Regression: without `-t`, the apad-ed (infinite) audio makes `amix
        // duration=first` never fire and the mux encodes silence forever.
        let cmd = final_mux_cmd("v.mp4", &ps, &[0, 1], "out.mp4", 3946.22);
        let t = cmd
            .iter()
            .position(|a| a == "-t")
            .expect("missing -t: infinite mux");
        assert_eq!(cmd[t + 1], "3946.220");
        assert!(cmd.iter().any(|a| a.contains("adelay=500")));
        assert!(cmd.iter().any(|a| a.contains("adelay=2250")));
        assert!(cmd.iter().any(|a| a.contains("amix=inputs=2")));
        assert!(!cmd.iter().any(|a| a == "-an"), "the mux keeps audio");
    }

    #[test]
    fn concat_is_a_stream_copy() {
        let cmd = concat_cmd("list.txt", "v.mp4");
        assert!(cmd.iter().any(|a| a == "list.txt"));
        let c = cmd.iter().position(|a| a == "-c").unwrap();
        assert_eq!(cmd[c + 1], "copy");
    }
}
