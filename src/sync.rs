//! Phase 2 — audio alignment via normalized cross-correlation, plus a two-point
//! drift estimate.
//!
//! The cloud master's audio is a *mix* that contains each participant's voice,
//! so correlating a participant's local track against the mix peaks at the exact
//! offset where the two lines up. We only ever need the start and end of the
//! local track, so memory stays small even for hour-long recordings.

use crate::model::ParticipantSync;

/// Analysis sample rate (Hz). Correlation is rate-independent once converted to
/// seconds, so a modest 16 kHz keeps memory low without hurting offset accuracy.
pub const ANALYSIS_SR: usize = 16_000;

/// Normalized cross-correlation of `a[ao..ao+win]` against `b[bo..bo+win]` in [-1,1].
fn ncc(a: &[f32], ao: usize, b: &[f32], bo: usize, win: usize) -> f64 {
    if win == 0 || ao + win > a.len() || bo + win > b.len() {
        return 0.0;
    }
    let mut sa = 0.0f64;
    let mut sb = 0.0f64;
    for k in 0..win {
        sa += a[ao + k] as f64;
        sb += b[bo + k] as f64;
    }
    let ma = sa / win as f64;
    let mb = sb / win as f64;
    let mut num = 0.0f64;
    let mut da = 0.0f64;
    let mut db = 0.0f64;
    for k in 0..win {
        let x = a[ao + k] as f64 - ma;
        let y = b[bo + k] as f64 - mb;
        num += x * y;
        da += x * x;
        db += y * y;
    }
    let den = (da * db).sqrt();
    if den > 1e-9 {
        num / den
    } else {
        0.0
    }
}

/// RMS of a window.
fn window_rms(a: &[f32], off: usize, win: usize) -> f64 {
    if off + win > a.len() {
        return 0.0;
    }
    let mut s = 0.0f64;
    for k in 0..win {
        s += (a[off + k] as f64) * (a[off + k] as f64);
    }
    (s / win as f64).sqrt()
}

/// Find the cloud offset `T` that best matches `local[local_off..local_off+win]`,
/// searching `T` in `[lo, hi]`. Two-stage: coarse 0.5 s grid, then a fine 10 ms pass.
fn best_offset_at(
    cloud: &[f32],
    local: &[f32],
    local_off: usize,
    win: usize,
    lo: usize,
    hi: usize,
) -> (usize, f64) {
    let sr = ANALYSIS_SR;
    let hi = hi.min(cloud.len().saturating_sub(win));
    if hi <= lo {
        return (lo, ncc(local, local_off, cloud, lo, win));
    }

    let step = (sr / 2).max(1); // 0.5 s
    let (ct, _) = (lo..=hi)
        .step_by(step)
        .map(|t| (t, ncc(local, local_off, cloud, t, win)))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((lo, 0.0));

    let fine = (sr / 100).max(1); // 10 ms
    let flo = ct.saturating_sub(step);
    let fhi = (ct as isize + step as isize).min(hi as isize) as usize;
    (flo..=fhi)
        .step_by(fine)
        .map(|t| (t, ncc(local, local_off, cloud, t, win)))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((ct, 0.0))
}

/// Align one local track against the cloud master.
///
/// Returns `(offset_secs, drift_per_sec, confidence)`:
///   - `offset_secs`: cloud time at which local t=0 lands.
///   - `drift_per_sec`: fractional local-vs-cloud rate difference (e.g. 0.001 = +0.1 %).
///   - `confidence`: peak normalized cross-correlation in [0,1].
///
/// `hint_secs` is an a-priori offset (StreamYard bakes each local track's start
/// time into its filename, e.g. `...-00h_11m_36s_488ms-...`). When present we
/// search only a narrow band around it — searching a short guest utterance across
/// the whole hour-long mix is ambiguous, but refining a known start to sub-frame
/// precision is reliable. Without a hint we fall back to a global search.
///
/// The anchor is the local track's **most active** window (guaranteed signal),
/// not its start, because the first seconds are often silence.
pub fn align(cloud: &[f32], local: &[f32], hint_secs: Option<f64>) -> (f64, f64, f64) {
    let sr = ANALYSIS_SR;
    if local.len() < sr || cloud.len() < sr {
        return (hint_secs.unwrap_or(0.0), 0.0, 0.0);
    }
    let win = (2 * sr).min(local.len());
    let max_t = cloud.len().saturating_sub(win);

    // Most active 1 s-spaced window of the local track.
    let mut best_off = 0usize;
    let mut best_rms = -1.0f64;
    let mut off = 0usize;
    while off + win <= local.len() {
        let r = window_rms(local, off, win);
        if r > best_rms {
            best_rms = r;
            best_off = off;
        }
        off += sr;
    }
    if best_rms < 1e-3 {
        best_off = 0;
    }

    // With a filename hint, the offset is authoritative (StreamYard bakes the
    // local track's start time into the name); cross-correlation validates it and
    // supplies confidence. Without a hint, cross-correlation *is* the offset.
    let (offset, conf0) = match hint_secs {
        Some(h) => {
            let center = (h * sr as f64 + best_off as f64) as isize;
            let margin = (5 * sr) as isize;
            let lo = (center - margin).max(0) as usize;
            let hi = ((center + margin).min(max_t as isize) as usize).max(lo);
            let (_, c) = best_offset_at(cloud, local, best_off, win, lo, hi);
            (h, c)
        }
        None => {
            let (t0, c) = best_offset_at(cloud, local, best_off, win, 0, max_t);
            ((t0 as f64 - best_off as f64) / sr as f64, c)
        }
    };

    // Drift: find a second active window near the end and compare its realised
    // gap to the constant-rate prediction.
    let drift = if local.len() > 40 * sr {
        let end_off = local.len() - win;
        let t0 = (offset * sr as f64 + best_off as f64) as usize;
        let expected = t0 + (end_off - best_off);
        let margin = 5 * sr;
        let lo2 = expected.saturating_sub(margin);
        let hi2 = (expected as isize + margin as isize).min(max_t as isize) as usize;
        let (t1, _) = best_offset_at(cloud, local, end_off, win, lo2, hi2);
        let dt_actual = (t1 as f64 - t0 as f64) / sr as f64;
        let dt_expected = (end_off - best_off) as f64 / sr as f64;
        (dt_actual - dt_expected) / dt_expected.max(1.0)
    } else {
        0.0
    };

    (offset, drift, conf0)
}

/// Convenience that turns an aligned participant's numbers into a report row.
pub fn to_sync_row(
    id: &str,
    name: &str,
    source: &str,
    offset: f64,
    drift: f64,
    conf: f64,
) -> ParticipantSync {
    ParticipantSync {
        id: id.to_string(),
        name: name.to_string(),
        source: source.to_string(),
        offset_secs: offset,
        drift_per_sec: drift,
        confidence: conf,
    }
}
