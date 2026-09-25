mod composer;
mod ffmpeg;
mod model;
mod sync;
mod vision;

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};

use model::{Participant, SyncReport};

#[derive(Parser, Debug)]
#[command(
    name = "streamyard-stitcher",
    version,
    about = "Rebuild a StreamYard cloud master with high-res local tracks."
)]
struct Cli {
    /// Input directory containing the cloud recording + local track recordings.
    #[arg(short, long, value_name = "DIR")]
    input: PathBuf,

    /// Destination path for the upgraded video.
    #[arg(short, long, value_name = "FILE")]
    output: PathBuf,

    /// Enable hardware encoding (VideoToolbox on macOS, NVENC on Linux).
    #[arg(long)]
    gpu: bool,

    /// Print the layout JSON + generated ffmpeg command without encoding.
    #[arg(long)]
    dry_run: bool,

    /// Number of timeline samples for layout detection (default auto, ~40).
    #[arg(long, value_name = "N")]
    samples: Option<usize>,

    /// Width (px) at which frames are decoded for detection. Higher keeps the thin
    /// black borders between stacked PIPs sharp (at 480 they blur into one merged
    /// column); 1280 is the default.
    #[arg(long, value_name = "PX", default_value_t = 1280)]
    detect_width: u32,

    /// Use a pre-made layout JSON (LayoutTimeline) instead of auto-detecting boxes.
    #[arg(long, value_name = "FILE")]
    layout: Option<PathBuf>,

    /// Cap the output length to N seconds (useful for quick validation encodes).
    #[arg(long, value_name = "SECS")]
    limit: Option<u64>,

    /// Debug: print the detected tiles + tile↔participant similarity matrix at
    /// cloud time TIME, then exit (no audio, no encode).
    #[arg(long, value_name = "TIME")]
    debug_identity: Option<f64>,
}

/// Parse StreamYard's `HHh_MMm_SSs_mmmms` start-time token from a filename.
/// Returns the offset in seconds, or `None` if the token is absent.
fn parse_name_offset(path: &str) -> Option<f64> {
    fn digits_before(name: &str, idx: usize) -> Option<f64> {
        let run = name[..idx].rsplit(|c: char| !c.is_ascii_digit()).next()?;
        if run.is_empty() {
            None
        } else {
            run.parse().ok()
        }
    }
    let hi = path.find("h_")?;
    let h = digits_before(path, hi)?;
    let mi = path[hi..].find("m_").map(|p| hi + p)?;
    let m = digits_before(path, mi)?;
    let si = path[mi..].find("s_").map(|p| mi + p)?;
    let s = digits_before(path, si)?;
    let msi = path[si..].find("ms").map(|p| si + p)?;
    let ms = digits_before(path, msi).unwrap_or(0.0);
    Some(h * 3600.0 + m * 60.0 + s + ms / 1000.0)
}

fn slugify(s: &str) -> String {
    let mut out = String::new();
    let mut dash = false;
    for c in s.to_lowercase().chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c);
            dash = false;
        } else if !dash {
            out.push('-');
            dash = true;
        }
    }
    out.trim_matches('-').to_string()
}

/// If `out` names a directory (already one, or ends with a path separator), pick a
/// unique `<cloud-stem>.<ext>` inside it — `name.mp4`, then `name-1.mp4`, … —
/// creating the directory if needed. Otherwise return `out` unchanged (treated
/// as an explicit file path).
fn resolve_output(out: &Path, cloud: &Path) -> Result<PathBuf> {
    let is_dir = out.is_dir() || out.to_string_lossy().ends_with(std::path::MAIN_SEPARATOR);
    if !is_dir {
        return Ok(out.to_path_buf());
    }
    let stem = cloud
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("streamyard");
    let ext = cloud.extension().and_then(|e| e.to_str()).unwrap_or("mp4");
    std::fs::create_dir_all(out)?;
    let mut k = 0u32;
    loop {
        let p = match k {
            0 => out.join(format!("{stem}.{ext}")),
            n => out.join(format!("{stem}-{n}.{ext}")),
        };
        if !p.exists() {
            return Ok(p);
        }
        k += 1;
    }
}

/// Classify the files in a directory into a cloud master + local participant tracks.
fn classify(dir: &Path) -> Result<(PathBuf, Vec<Participant>)> {
    let mut cloud: Option<Participant> = None;
    let mut locals: Vec<Participant> = Vec::new();

    for entry in std::fs::read_dir(dir).context("reading input dir")? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if !["mp4", "mov", "mkv", "webm"].contains(&ext.to_lowercase().as_str()) {
            continue;
        }
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();
        let lower = stem.to_lowercase();
        let source = if lower.contains("-webcam") {
            Some("webcam")
        } else if lower.contains("-screen") {
            Some("screen")
        } else {
            None
        };

        match source {
            Some(src) => {
                // Name is the token before the source token, minus the show-name prefix.
                let pre = lower.split(&format!("-{src}")).next().unwrap_or(&lower);
                let name = pre.split("___-").last().unwrap_or(pre).trim().to_string();
                let display = name.replace('_', " ");
                let id = format!("{}-{src}", slugify(&name));
                let p = Participant {
                    id,
                    name: display,
                    source: src.to_string(),
                    file: path.display().to_string(),
                    duration: 0.0,
                    offset_secs: None,
                    drift_per_sec: None,
                    confidence: None,
                };
                locals.push(p);
            }
            None => {
                if cloud.is_none() {
                    let name = stem.replace('_', " ");
                    cloud = Some(Participant {
                        id: "cloud".into(),
                        name,
                        source: "cloud".into(),
                        file: path.display().to_string(),
                        duration: 0.0,
                        offset_secs: None,
                        drift_per_sec: None,
                        confidence: None,
                    });
                }
            }
        }
    }

    if locals.is_empty() {
        bail!("no local track recordings found (expected files with a -webcam or -screen token)");
    }
    let cloud = cloud
        .context("no cloud recording found (expected one file without a -webcam/-screen token)")?;
    if cloud.file.is_empty() {
        bail!("no cloud master file found");
    }

    Ok((PathBuf::from(&cloud.file), locals))
}

/// Build one appearance signature per participant. A participant's camera can be
/// off (black) for stretches, so sample several frames across their feed and use
/// the one with the most content as the reference.
fn build_signatures(
    participants: &[Participant],
    wdir: &Path,
    width: u32,
) -> Vec<vision::Signature> {
    let mut sigs = Vec::new();
    for p in participants {
        if p.duration < 2.0 {
            continue;
        }
        let times: Vec<f64> = (1..=8)
            .map(|i| p.duration * i as f64 / 9.0)
            .filter(|t| *t > 1.0 && *t < p.duration - 1.0)
            .collect();
        let f = wdir.join(format!("sig_{}.ppm", p.id));
        let mut best: Option<(f32, ffmpeg::PpmFrame)> = None;
        for t in times {
            if ffmpeg::sample_ppm(&p.file, t, width, &f.display().to_string()).is_err() {
                continue;
            }
            let Ok(frame) = ffmpeg::read_ppm(&f.display().to_string()) else {
                continue;
            };
            let score = vision::content_score(&frame);
            if best.as_ref().is_none_or(|(b, _)| score > *b) {
                best = Some((score, frame));
            }
        }
        match best {
            Some((score, frame)) => {
                if score < 1.0 {
                    tracing::warn!(
                        "no visible content found in {} — signature may be unreliable",
                        p.name
                    );
                }
                sigs.push(vision::Signature::from_frame(&p.id, &frame, 24));
            }
            None => tracing::warn!("signature sample failed for {}", p.name),
        }
    }
    sigs
}

/// Phase 3 auto-detection: sample cloud frames, build participant signatures,
/// detect tiles, match them, and cluster into a `LayoutTimeline`.
fn detect_layout(
    cloud: &Path,
    participants: &[Participant],
    wdir: &Path,
    cli: &Cli,
    cloud_probe: &model::ProbeInfo,
) -> Result<model::LayoutTimeline> {
    // Finer default sampling (~15 s spacing) so brief layout changes are caught.
    let n_samples = cli
        .samples
        .unwrap_or_else(|| (cloud_probe.duration as usize / 15).clamp(24, 600));
    let spacing = (cloud_probe.duration / n_samples.max(1) as f64).max(1.0);
    let times: Vec<f64> = (0..n_samples)
        .map(|i| ((i as f64 + 0.5) * spacing).min(cloud_probe.duration - 0.5))
        .collect();
    tracing::info!(
        "detecting layout across {} samples (spacing≈{:.0}s, width={}) ...",
        times.len(),
        spacing,
        cli.detect_width
    );

    // Participant appearance signatures (one reference frame each).
    let sigs = build_signatures(participants, wdir, cli.detect_width);
    if sigs.is_empty() {
        anyhow::bail!("no participant signatures could be decoded");
    }

    let out_w = cloud_probe.width;
    let out_h = cloud_probe.height;
    let cloud_file = cloud.display().to_string();
    let width = cli.detect_width;

    // Decode the cloud samples with bounded concurrency (don't spawn hundreds of ffmpegs).
    let frame_paths: Vec<PathBuf> = times
        .iter()
        .map(|t| wdir.join(format!("frame_{}.ppm", format!("{t:.3}").replace('.', "_"))))
        .collect();
    let concurrency = 8usize;
    for start in (0..times.len()).step_by(concurrency) {
        let end = (start + concurrency).min(times.len());
        std::thread::scope(|scope| {
            for i in start..end {
                let t = times[i];
                let f = frame_paths[i].clone();
                let cf = cloud_file.clone();
                scope.spawn(move || {
                    let _ = ffmpeg::sample_ppm(&cf, t, width, &f.display().to_string());
                });
            }
        });
    }

    // Classify each sample into (time, matched tiles).
    let mut coarse: Vec<(f64, Vec<vision::TileScore>)> = Vec::new();
    for (t, f) in times.iter().zip(frame_paths.iter()) {
        if let Ok(frame) = ffmpeg::read_ppm(&f.display().to_string()) {
            coarse.push((*t, vision::match_tiles(&frame, &sigs, out_w, out_h)));
        }
    }
    if coarse.is_empty() {
        anyhow::bail!("no layout samples decoded");
    }

    // Classifier for exact boundary refinement: decode a frame at time t and match it.
    let classifier = {
        let sigs = sigs.clone();
        let cf = cloud_file.clone();
        let tmp = wdir.join("bisect.ppm");
        Box::new(move |t: f64| -> Vec<vision::TileScore> {
            let tp = tmp.display().to_string();
            match ffmpeg::sample_ppm(&cf, t, width, &tp) {
                Ok(()) => match ffmpeg::read_ppm(&tp) {
                    Ok(frame) => vision::match_tiles(&frame, &sigs, out_w, out_h),
                    Err(_) => Vec::new(),
                },
                Err(_) => Vec::new(),
            }
        })
    };

    Ok(vision::build_timeline(
        coarse,
        &*classifier,
        &cloud_file,
        out_w,
        out_h,
        cloud_probe.duration,
    ))
}

fn work_dir() -> Result<PathBuf> {
    let dir = std::env::temp_dir()
        .join("streamyard_stitcher")
        .join(format!(
            "run_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "streamyard_stitcher=info,info".into()),
        )
        .init();

    if !cli.input.is_dir() {
        bail!("input dir not found: {}", cli.input.display());
    }

    // ---- Phase 1: classification + probe + audio extraction ----------------
    let (cloud, mut participants) = classify(&cli.input)?;
    tracing::info!("cloud master : {}", cloud.display());
    for p in &participants {
        tracing::info!("local track : {} ({}) → {}", p.name, p.source, p.file);
    }

    let output = resolve_output(&cli.output, &cloud)?;
    if output != cli.output {
        tracing::info!("output dir  : auto-generated {}", output.display());
    }

    let cloud_probe = ffmpeg::probe(&cloud.display().to_string())?;
    tracing::info!(
        "cloud size   : {}x{} @ {:.2}s",
        cloud_probe.width,
        cloud_probe.height,
        cloud_probe.duration
    );
    for p in &mut participants {
        let info = ffmpeg::probe(&p.file)?;
        p.duration = info.duration;
    }

    let wdir = work_dir()?;

    // Debug: inspect tile detection + identity matching at one cloud time.
    if let Some(t) = cli.debug_identity {
        let sigs = build_signatures(&participants, &wdir, cli.detect_width);
        if sigs.is_empty() {
            bail!("no participant signatures could be decoded");
        }
        let fp = wdir.join("debug.ppm");
        ffmpeg::sample_ppm(
            &cloud.display().to_string(),
            t,
            cli.detect_width,
            &fp.display().to_string(),
        )?;
        let frame = ffmpeg::read_ppm(&fp.display().to_string())?;
        let (tiles, matrix) = vision::tile_similarities(&frame, &sigs);
        println!(
            "\n=== DEBUG IDENTITY @ t={t:.1}s  ({} participants) ===",
            sigs.len()
        );
        println!("{} tile(s) detected:", tiles.len());
        for (i, r) in tiles.iter().enumerate() {
            let sims: Vec<String> = matrix[i]
                .iter()
                .enumerate()
                .map(|(s, v)| format!("{}={v:.3}", sigs[s].id))
                .collect();
            println!(
                "  [{i}] x={} y={} w={} h={}   {}",
                r.x,
                r.y,
                r.w,
                r.h,
                sims.join("  ")
            );
        }
        return Ok(());
    }

    let cloud_pcm = wdir.join("cloud.raw");
    let local_pcms: Vec<PathBuf> = participants
        .iter()
        .map(|p| wdir.join(format!("{}.raw", p.id)))
        .collect();
    tracing::info!(
        "extracting analysis audio (mono {sr} Hz) ...",
        sr = sync::ANALYSIS_SR
    );

    std::thread::scope(|scope| {
        let cloud_file = cloud.display().to_string();
        let cloud_out = cloud_pcm.display().to_string();
        scope.spawn(move || {
            if let Err(e) = ffmpeg::to_raw_pcm(&cloud_file, &cloud_out, sync::ANALYSIS_SR as u32) {
                tracing::error!("PCM extraction failed for cloud: {e}");
            }
        });
        for (p, out) in participants.iter().zip(local_pcms.iter()) {
            let file = p.file.clone();
            let out = out.display().to_string();
            scope.spawn(move || {
                if let Err(e) = ffmpeg::to_raw_pcm(&file, &out, sync::ANALYSIS_SR as u32) {
                    tracing::error!("PCM extraction failed for {file}: {e}");
                }
            });
        }
    });

    // ---- Phase 2: audio alignment -----------------------------------------
    let cloud_audio = ffmpeg::read_pcm(&cloud_pcm.display().to_string())?;
    tracing::info!(
        "aligning {} local tracks against the cloud master ...",
        participants.len()
    );
    let pb = ProgressBar::new(participants.len() as u64);
    pb.set_style(
        ProgressStyle::with_template("{spinner} aligning {pos}/{len} [{elapsed}]").expect("style"),
    );
    let mut report = SyncReport {
        cloud: cloud.display().to_string(),
        cloud_duration: cloud_probe.duration,
        sample_rate: sync::ANALYSIS_SR as u32,
        participants: Vec::new(),
    };
    for p in &mut participants {
        let local_path = wdir.join(format!("{}.raw", p.id));
        let hint = parse_name_offset(&p.file);
        match ffmpeg::read_pcm(&local_path.display().to_string()) {
            Ok(local) => {
                let (offset, drift, conf) = sync::align(&cloud_audio, &local, hint);
                p.offset_secs = Some(offset);
                p.drift_per_sec = Some(drift);
                p.confidence = Some(conf);
                report.participants.push(sync::to_sync_row(
                    &p.id, &p.name, &p.source, offset, drift, conf,
                ));
                tracing::info!(
                    "  {:<40} offset={offset:+8.3}s drift={drift:+.5} conf={conf:.3}",
                    p.name
                );
            }
            Err(e) => tracing::warn!("  {} alignment skipped: {e}", p.name),
        }
        pb.inc(1);
    }
    pb.finish_and_clear();

    // Write the sync report (always emitted, even on dry-run).
    let report_path = wdir.join("sync_report.json");
    std::fs::write(&report_path, serde_json::to_string_pretty(&report)?)?;
    tracing::info!("sync report  : {}", report_path.display());

    // ---- Phase 3: layout (supplied or auto-detected) ----------------------
    let timeline = match &cli.layout {
        Some(path) => {
            let json = std::fs::read_to_string(path)
                .with_context(|| format!("reading layout file {}", path.display()))?;
            let tl: model::LayoutTimeline = serde_json::from_str(&json)?;
            tracing::info!("using supplied layout: {} segments", tl.segments.len());
            tl
        }
        None => detect_layout(&cloud, &participants, &wdir, &cli, &cloud_probe)?,
    };
    tracing::info!(
        "layout: {} spatial segments across {} participants",
        timeline.segments.len(),
        participants.len()
    );

    // ---- Phase 4: subscribe-band overlay mask ------------------------------
    // A full-frame solo feed covers the web master's persistent subscribe band.
    // The band is a blue pill carrying white text, present in every layout; we
    // restore its whole pill shape from the cloud so the crisp text survives a
    // solo's feed. make_overlay_mask samples frames itself and picks the thinnest
    // centred band, so a solo or a participant's blue can't pollute it.
    let overlay_path = wdir.join("overlay.png");
    let mask_res = ffmpeg::make_overlay_mask(
        &cloud.display().to_string(),
        cloud_probe.duration,
        &overlay_path.display().to_string(),
    );
    let overlay_ok = mask_res.as_ref().is_ok_and(|_| {
        std::path::Path::new(&overlay_path)
            .metadata()
            .map(|m| m.len() > 0)
            .unwrap_or(false)
    });
    if overlay_ok {
        tracing::info!("subscribe-band overlay mask baked");
    } else {
        tracing::warn!("no overlay mask — solo feeds will cover the subscribe band ({mask_res:?})");
    }
    let overlay = overlay_ok.then(|| overlay_path.display().to_string());

    // ---- Phase 5: batched encode -------------------------------------------
    // Render the timeline as a sequence of constant-layout windows (each a small
    // ffmpeg), concatenate them, then mux the high-res audio on top. Batching
    // bounds peak memory to the busiest single window instead of the whole graph.
    let end_time = cli
        .limit
        .map(|l| (l as f64).min(cloud_probe.duration))
        .unwrap_or(cloud_probe.duration);
    let windows = composer::build_windows(&timeline, &participants, end_time);
    tracing::info!("{} layout window(s) over {:.0}s", windows.len(), end_time);

    let audio_idxs: Vec<usize> = participants
        .iter()
        .enumerate()
        .filter(|(_, p)| !p.file.is_empty())
        .map(|(i, _)| i)
        .collect();

    let clips_dir = wdir.join("clips");
    std::fs::create_dir_all(&clips_dir)?;
    let clip_paths: Vec<PathBuf> = (0..windows.len())
        .map(|i| clips_dir.join(format!("clip_{i}.mp4")))
        .collect();
    let list_path = clips_dir.join("concat.txt");
    let video_only = wdir.join("video_only.mp4");
    let overlay_opt = overlay.as_deref();

    // Assemble every step (also emitted on --dry-run).
    let mut cmds: Vec<Vec<String>> = Vec::new();
    for (i, w) in windows.iter().enumerate() {
        cmds.push(composer::video_window_cmd(
            &cloud.display().to_string(),
            &participants,
            w,
            overlay_opt,
            &clip_paths[i].display().to_string(),
            cli.gpu,
        ));
    }
    cmds.push(composer::concat_cmd(
        &list_path.display().to_string(),
        &video_only.display().to_string(),
    ));
    cmds.push(composer::final_mux_cmd(
        &video_only.display().to_string(),
        &participants,
        &audio_idxs,
        &output.display().to_string(),
        end_time,
    ));

    // Persist the layout + commands for inspection / dry-run.
    let layout_path = wdir.join("layout.json");
    std::fs::write(&layout_path, serde_json::to_string_pretty(&timeline)?)?;
    let cmd_path = wdir.join("commands.txt");
    let cmd_text: String = cmds
        .iter()
        .map(|c| c.join(" "))
        .collect::<Vec<_>>()
        .join("\n\n");
    std::fs::write(&cmd_path, &cmd_text)?;

    tracing::info!("layout       : {}", layout_path.display());

    if cli.dry_run {
        tracing::info!(
            "--dry-run: no encode. layout + commands written to {}.",
            wdir.display()
        );
        println!(
            "\n=== LAYOUT ({} segments → {} windows) ===\n{}",
            timeline.segments.len(),
            windows.len(),
            serde_json::to_string_pretty(&timeline)?
        );
        println!("\n=== SYNC ===\n{}", serde_json::to_string_pretty(&report)?);
        println!(
            "\n=== FFMPEG COMMANDS ({} steps) ===\n{}",
            cmds.len(),
            cmd_text
        );
        return Ok(());
    }

    // ---- Encode: windows → concat → mux ------------------------------------
    tracing::info!("encoding → {}", output.display());
    for (i, w) in windows.iter().enumerate() {
        let label = if w.tiles.is_empty() {
            "cloud pass-through".to_string()
        } else {
            format!("{} tile(s)", w.tiles.len())
        };
        tracing::info!(
            "[window {}/{}] [{:.1}–{:.1}] {}",
            i + 1,
            windows.len(),
            w.start,
            w.end,
            label
        );
        composer::run_command(&cmds[i])?;
    }
    // Write the concat list, then concatenate + mux.
    let list: String = clip_paths
        .iter()
        .map(|p| format!("file '{}'", p.display()))
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(&list_path, list)?;
    tracing::info!("[concat] {} clip(s) → video_only", windows.len());
    composer::run_command(&cmds[windows.len()])?;
    tracing::info!("[mux] video + audio → {}", output.display());
    composer::run_command(&cmds[windows.len() + 1])?;
    tracing::info!("done: {}", output.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_name_offset_reads_streamyard_token() {
        // StreamYard encodes the local start offset in the filename.
        assert_eq!(
            parse_name_offset("x-screen-00h_11m_36s_488ms-StreamYard.mp4"),
            Some(11.0 * 60.0 + 36.0 + 0.488)
        );
        assert_eq!(
            parse_name_offset("x-webcam-00h_00m_00s_433ms-StreamYard.mp4"),
            Some(0.433)
        );
        assert_eq!(parse_name_offset("no-token.mp4"), None);
    }
}
