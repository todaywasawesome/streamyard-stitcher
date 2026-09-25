//! Thin wrappers around the `ffmpeg` / `ffprobe` binaries.
//!
//! Decoding is delegated to the installed ffmpeg (explicitly sanctioned by the
//! design doc); we only ever ask it to produce byte streams we can parse by
//! hand (raw PCM, PPM frames), so no `image`/`symphonia`/`hound` deps are needed.

use anyhow::ensure;

use crate::model::ProbeInfo;

/// Decode a whole file to mono 16-bit little-endian raw PCM at `rate` Hz.
pub fn to_raw_pcm(input: &str, out: &str, rate: u32) -> anyhow::Result<()> {
    let st = std::process::Command::new("ffmpeg")
        .args([
            "-y",
            "-v",
            "error",
            "-i",
            input,
            "-vn",
            "-ac",
            "1",
            "-ar",
            &rate.to_string(),
            "-f",
            "s16le",
            out,
        ])
        .status()?;
    ensure!(st.success(), "PCM extraction failed for {input}");
    Ok(())
}

/// Read a raw s16le PCM file into `f32` samples in [-1, 1].
pub fn read_pcm(path: &str) -> anyhow::Result<Vec<f32>> {
    let buf = std::fs::read(path)?;
    let n = buf.len() / 2;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let s = i16::from_le_bytes([buf[2 * i], buf[2 * i + 1]]);
        out.push(s as f32 / 32768.0);
    }
    Ok(out)
}

/// Grab one frame at time `t` (seconds) as a P6 PPM, scaled to `width` px wide.
pub fn sample_ppm(input: &str, t: f64, width: u32, out: &str) -> anyhow::Result<()> {
    let tf = format!("{:.3}", t);
    let st = std::process::Command::new("ffmpeg")
        .args([
            "-y",
            "-v",
            "error",
            "-ss",
            &tf,
            "-i",
            input,
            "-frames:v",
            "1",
            "-vf",
            &format!("scale={width}:-2"),
            "-f",
            "image2",
            "-c:v",
            "ppm",
            out,
        ])
        .status()?;
    ensure!(st.success(), "frame sample failed at t={t}s for {input}");
    Ok(())
}

/// The subscribe band's stadium geometry (cx, cy, half-width, radius) plus the
/// widest contiguous blue run in the frame. `maxrun` is the discriminator: the
/// true pill is the *widest* blue bar (~half the frame); a participant's blue or
/// a lower-third is a narrower run.
type Band = (i32, i32, i32, i32, i32);

/// Measure the subscribe-band pill from a frame.
///
/// The band is a wide solid blue bar. For each row in the lower part of the frame
/// we find the longest contiguous blue run; the band's rows are the ones whose
/// run is a large fraction of the width, and their blue extent gives the pill's
/// position and size. A frame whose widest run is still short carries no subscribe
/// pill (only a decoy) and is rejected.
fn measure_band(f: &PpmFrame) -> Option<Band> {
    let w = f.width as usize;
    let h = f.height as usize;
    let y0 = h * 6 / 10; // lower 40% only — the band sits at the very bottom
    let is_blue = |px: &[u8; 3]| {
        px[2] > 120 && (px[2] as i32) > (px[0] as i32) + 40 && (px[2] as i32) > (px[1] as i32) + 20
    };
    let mut run_len = vec![0usize; h]; // longest contiguous blue run per row
    let mut lo = vec![w; h]; // leftmost blue px per row
    let mut hi = vec![0usize; h]; // rightmost blue px per row
    let mut maxrun = 0usize;
    for y in y0..h {
        let mut l = w;
        let mut r = 0usize;
        let mut cur = 0usize;
        let mut best = 0usize;
        for x in 0..w {
            if is_blue(&f.pixels[y * w + x]) {
                cur += 1;
                if x < l {
                    l = x;
                }
                if x > r {
                    r = x;
                }
                if cur > best {
                    best = cur;
                }
            } else {
                cur = 0;
            }
        }
        run_len[y] = best;
        lo[y] = l;
        hi[y] = r;
        if best > maxrun {
            maxrun = best;
        }
    }
    if maxrun < w * 5 / 10 {
        return None; // no wide pill in this frame
    }
    let thr = w * 3 / 10;
    let mut miny = h;
    let mut maxy = 0usize;
    let mut minx = w;
    let mut maxx = 0usize;
    for y in y0..h {
        if run_len[y] > thr {
            if y < miny {
                miny = y;
            }
            if y > maxy {
                maxy = y;
            }
            if lo[y] < minx {
                minx = lo[y];
            }
            if hi[y] > maxx {
                maxx = hi[y];
            }
        }
    }
    Some((
        ((minx + maxx) / 2) as i32,
        ((miny + maxy) / 2) as i32,
        ((maxx - minx) / 2) as i32,
        ((maxy - miny) / 2) as i32,
        maxrun as i32,
    ))
}

/// Measure the top-right mascot badge (fixed chrome: a white badge with a light-blue
/// rim and an orange octopus) from a frame: its centre and radius.
///
/// The badge's rim is the *corner-most* cool-bright blob — cool because it is
/// light-blue/neutral (a room or shirt is warm, `b < r`), and nearest the top-right
/// corner because a bright object in the feed sits lower/more central, farther from
/// `(w,0)`. So: find the cool-bright pixel nearest the corner (on the rim), then
/// take the cool-bright cluster within one badge-diameter of it, and fit a circle to
/// that cluster's bounding box.
///
/// ponytail: `R` (the cluster window) is a fixed 1.5× the badge diameter and assumes
/// the badge is the nearest cool-bright blob to the corner. A cool object held up
/// dead in the top-right corner could false-positive. Upgrade path: template-match.
fn measure_mascot(f: &PpmFrame) -> Option<(i32, i32, i32)> {
    let w = f.width as usize;
    let h = f.height as usize;
    let x0 = w * 72 / 100;
    let y1 = h * 4 / 10;
    let is_badge = |px: &[u8; 3]| {
        let (r, g, b) = (px[0] as i32, px[1] as i32, px[2] as i32);
        b > 190 && g > 190 && b >= r - 8 // cool + bright: light-blue rim / neutral white, not a warm room
    };
    // Pass 1: the cool-bright pixel nearest the top-right corner (w, 0).
    let mut nearest: Option<(usize, usize)> = None;
    let mut best_d2 = u64::MAX;
    for y in 0..y1 {
        for x in x0..w {
            if is_badge(&f.pixels[y * w + x]) {
                let dx = (w - x) as u64;
                let dy = y as u64;
                let d2 = dx * dx + dy * dy;
                if d2 < best_d2 {
                    best_d2 = d2;
                    nearest = Some((x, y));
                }
            }
        }
    }
    let (nx, ny) = nearest?; // no cool-bright corner blob → no mascot
                             // Pass 2: the badge is the cool-bright cluster within R of that corner-most pixel.
    let rwin = 150u64;
    let (mut minx, mut miny, mut maxx, mut maxy, mut n) = (w, h, 0usize, 0usize, 0usize);
    for y in 0..y1 {
        for x in x0..w {
            if is_badge(&f.pixels[y * w + x]) {
                let dx = x as u64 - nx as u64;
                let dy = y as u64 - ny as u64;
                if dx * dx + dy * dy <= rwin * rwin {
                    n += 1;
                    if x < minx {
                        minx = x;
                    }
                    if x > maxx {
                        maxx = x;
                    }
                    if y < miny {
                        miny = y;
                    }
                    if y > maxy {
                        maxy = y;
                    }
                }
            }
        }
    }
    if n < 300 {
        return None;
    }
    let cx = (minx + maxx) / 2;
    let cy = (miny + maxy) / 2;
    let rad = (maxx - minx).max(maxy - miny) / 2;
    Some((cx as i32, cy as i32, rad as i32 + 2)) // +2 margin so the rim isn't clipped
}

/// Build the persistent subscribe-band mask (PNG with alpha) for the web
/// master's bottom band, which a full-frame solo feed would otherwise cover.
///
/// The band is a blue pill carrying white text, present in every layout. A
/// per-pixel blue-dominance mask (the old approach) restores the pill but leaves
/// the white text pixels transparent, so the high-res feed ghosts through the
/// letters. Instead we measure the pill's envelope and restore the whole stadium
/// shape from the cloud — which carries the crisp text.
///
/// The band is fixed chrome at a constant position, but a given frame may carry a
/// participant's blue or a decoy lower-third that swells or narrows the measured
/// box. So we measure several frames spread across the clip and keep the one with
/// the widest contiguous blue run — the true pill is the widest blue bar, so the
/// decoys (narrower runs) lose to it.
///
/// The top-right mascot badge (a white circular badge with a light-blue rim) is
/// captured the same way — a fixed-chrome region restored from the cloud. It is a
/// self-contained badge, so a corner-restricted brightness test isolates it
/// (unlike the band's white *text*, which ghosts through a colour mask).
///
/// ponytail: name bars / lower-thirds are NOT captured — they move and carry
/// arbitrary text, so a fixed-region mask can't restore them; a full-frame feed
/// covers them. Samples 7 frames, so cost is a handful of quick decodes; fine
/// for a one-shot master.
pub fn make_overlay_mask(cloud: &str, duration: f64, out: &str) -> anyhow::Result<()> {
    let probe = format!("band_probe_{}", std::process::id());
    let mut best: Option<(f64, Band)> = None; // (time, band)
    let mut best_frame: Option<PpmFrame> = None;
    for k in 0..7 {
        let t = duration * (0.05 + 0.9 * (k as f64) / 6.0); // 5% .. 95%
        sample_ppm(cloud, t, 1920, &probe)?;
        let f = read_ppm(&probe)?;
        if let Some(b) = measure_band(&f) {
            if best.is_none_or(|(_, prev)| b.4 > prev.4) {
                best = Some((t, b));
                best_frame = Some(f); // mascot is measured from this same (fixed-chrome) frame
            }
        }
    }
    std::fs::remove_file(&probe).ok();
    let (band_time, (cx, cy, halfw, r, maxrun)) =
        best.ok_or_else(|| anyhow::anyhow!("no subscribe band found in any sampled frame"))?;
    tracing::info!(
        "subscribe band: cx={cx} cy={cy} halfw={halfw} r={r} maxrun={maxrun} (t={band_time:.2})"
    );

    // Alpha = inside the stadium (central bar + two end circles). This ffmpeg's
    // geq has no `and`/`or` functions, so AND = multiply, OR = add (clamped).
    // RGB = the cloud's own pixels, which carry the band's white text.
    //
    // The measured `halfw` is half the band's *total* width (maxx−minx)/2. A
    // stadium of half-width `halfw` PLUS end-caps of radius `r` would be `2r`
    // wider than the band, leaking the dark pixels behind its right edge as a
    // black pill. So the central bar's half-width is `halfw − r`; the caps then
    // reach exactly minx…maxx.
    let hw = (halfw - r).max(0);
    // The band's alpha expression: the stadium (central bar + two end caps). Each
    // `lt(…)` is 0/1; the terms are summed (geq has no `or`), and `gt(…,0)` ORs them.
    let band_a = format!(
        "lt(abs(X-{cx}),{hw})*lt(abs(Y-{cy}),{r})+lt((abs(X-{cx})-{hw})*(abs(X-{cx})-{hw})+(Y-{cy})*(Y-{cy}),{r}*{r})"
    );
    // The mascot badge's circle, OR'd into the same mask when present.
    let mascot = best_frame.as_ref().and_then(measure_mascot);
    if let Some((mx, my, mr)) = mascot {
        tracing::info!("mascot badge: cx={mx} cy={my} r={mr}");
    }
    let alpha = match mascot {
        Some((mx, my, mr)) => format!("{band_a}+lt((X-{mx})*(X-{mx})+(Y-{my})*(Y-{my}),{mr}*{mr})"),
        None => band_a,
    };
    let geq =
        format!("format=rgba,geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':a='if(gt({alpha},0),255,0)'");
    let mut args: Vec<String> = ["-y", "-v", "error"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    args.push("-ss".into());
    args.push(format!("{band_time:.3}"));
    args.push("-i".into());
    args.push(cloud.to_string());
    args.push("-vf".into());
    args.push(geq);
    args.extend([
        "-frames:v".into(),
        "1".into(),
        "-f".into(),
        "image2".into(),
        "-c:v".into(),
        "png".into(),
    ]);
    args.push(out.to_string());

    let o = std::process::Command::new("ffmpeg").args(&args).output()?;
    if !o.status.success() {
        let err = String::from_utf8_lossy(&o.stderr);
        anyhow::bail!("overlay mask encode failed for {cloud}\n{err}");
    }
    Ok(())
}

/// A decoded P6 PPM frame (24-bit RGB, row-major).
#[derive(Debug, Clone)]
pub struct PpmFrame {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<[u8; 3]>,
}

/// Parse a P6 PPM file. Handles comments and single/newline whitespace in the header.
pub fn read_ppm(path: &str) -> anyhow::Result<PpmFrame> {
    let b = std::fs::read(path)?;
    let mut i = 0usize;

    fn token(b: &[u8], i: &mut usize) -> anyhow::Result<String> {
        loop {
            while *i < b.len() && (b[*i] as char).is_whitespace() {
                *i += 1;
            }
            if *i < b.len() && b[*i] == b'#' {
                while *i < b.len() && b[*i] != b'\n' {
                    *i += 1;
                }
            } else {
                break;
            }
        }
        let start = *i;
        while *i < b.len() && !(b[*i] as char).is_whitespace() {
            *i += 1;
        }
        Ok(String::from_utf8_lossy(&b[start..*i]).into_owned())
    }

    let magic = token(&b, &mut i)?;
    ensure!(magic == "P6", "expected P6 ppm, got {magic}");
    let width: u32 = token(&b, &mut i)?.parse()?;
    let height: u32 = token(&b, &mut i)?.parse()?;
    let maxval: u32 = token(&b, &mut i)?.parse()?;
    ensure!(maxval == 255, "unsupported ppm maxval {maxval}");
    i += 1; // exactly one whitespace byte after maxval

    let need = (width as usize) * (height as usize) * 3;
    ensure!(b.len() >= i + need, "truncated ppm");
    let (chunks, _rem) = b[i..i + need].as_chunks::<3>();
    let pixels = chunks.iter().map(|c| [c[0], c[1], c[2]]).collect();
    Ok(PpmFrame {
        width,
        height,
        pixels,
    })
}

fn parse_rate(s: &str) -> Option<f64> {
    let (a, b) = s.split_once('/')?;
    let a: f64 = a.parse().ok()?;
    let b: f64 = b.parse().ok()?;
    (b > 0.0).then(|| a / b)
}

/// Probe a media file: dimensions, duration, audio presence, fps.
pub fn probe(path: &str) -> anyhow::Result<ProbeInfo> {
    let out = std::process::Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            path,
        ])
        .output()?;
    if !out.status.success() {
        anyhow::bail!(
            "ffprobe failed for {path}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let v: serde_json::Value = serde_json::from_slice(&out.stdout)?;

    let (mut width, mut height, mut fps, mut has_audio) = (0u32, 0u32, None, false);
    if let Some(streams) = v.get("streams").and_then(|s| s.as_array()) {
        for s in streams {
            match s.get("codec_type").and_then(|t| t.as_str()) {
                Some("video") => {
                    width = s.get("width").and_then(|w| w.as_u64()).unwrap_or(0) as u32;
                    height = s.get("height").and_then(|h| h.as_u64()).unwrap_or(0) as u32;
                    fps = s
                        .get("avg_frame_rate")
                        .and_then(|x| x.as_str())
                        .and_then(parse_rate);
                }
                Some("audio") => has_audio = true,
                _ => {}
            }
        }
    }
    let duration = v
        .get("format")
        .and_then(|f| f.get("duration"))
        .and_then(|d| d.as_str())
        .and_then(|d| d.parse::<f64>().ok())
        .unwrap_or(0.0);

    Ok(ProbeInfo {
        path: path.to_string(),
        width,
        height,
        duration,
        has_audio,
        video_fps: fps,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_ppm_parses_p6() {
        let (w, h) = (4u32, 3u32);
        let mut buf = b"P6\n4 3\n255\n".to_vec();
        for i in 0..(w * h) {
            buf.extend_from_slice(&[i as u8, (i * 2) as u8, (i * 3) as u8]);
        }
        let pid = std::process::id();
        let path = std::env::temp_dir().join(format!("stitcher_test_ppm_{pid}.ppm"));
        std::fs::write(&path, &buf).unwrap();
        let f = read_ppm(&path.display().to_string()).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!((f.width, f.height), (4, 3));
        assert_eq!(f.pixels.len(), 12);
        assert_eq!(f.pixels[0], [0, 0, 0]);
        assert_eq!(f.pixels[11], [11, 22, 33]);
    }
}
