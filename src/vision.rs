//! Phase 3 — layout detection, participant identity, and exact change-point timing.
//!
//! Pipeline (dependency-free, no OpenCV):
//!  1. Coarse-sample the cloud timeline (main.rs decodes frames in parallel).
//!  2. `detect_tiles` finds the video regions: a real feed is high-detail, a
//!     StreamYard background/margin is flat, so a gradient "detail map" +
//!     threshold + connected components isolates the tiles.
//!  3. `match_tiles` assigns each tile to a participant by a saturation-weighted
//!     colour histogram (spatially invariant, so it tolerates StreamYard's
//!     per-tile re-framing) — shirt colour is what actually distinguishes the
//!     people, and weighting by saturation makes the shirt dominate the wall.
//!  4. Consecutive samples with the same tile *structure* form a **layout run**;
//!     the identity of each box is the run's collective best match, which is
//!     robust to per-frame flicker.
//!  5. Run boundaries are refined by binary-searching a classifier so the
//!     reported start/end are the *exact* transition times, not sample midpoints.

use std::collections::HashMap;

use crate::ffmpeg::PpmFrame;
use crate::model::{LayoutTimeline, SpatialSegment};

/// Axis-aligned rectangle in full-resolution cloud pixels.
#[derive(Debug, Clone, Copy)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

impl Rect {
    fn iou(&self, o: &Rect) -> f64 {
        let ix1 = self.x.max(o.x) as f64;
        let iy1 = self.y.max(o.y) as f64;
        let ix2 = (self.x + self.w).min(o.x + o.w) as f64;
        let iy2 = (self.y + self.h).min(o.y + o.h) as f64;
        if ix2 <= ix1 || iy2 <= iy1 {
            return 0.0;
        }
        let inter = (ix2 - ix1) * (iy2 - iy1);
        let a = self.w as f64 * self.h as f64;
        let b = o.w as f64 * o.h as f64;
        inter / (a + b - inter).max(1e-9)
    }

    /// Fraction of `self`'s area that lies inside `o` (1.0 = fully contained).
    /// A tile that is mostly inside a larger one is clutter within that feed
    /// (a backpack, a shelf), not a separate person window — unlike IoU, which
    /// reads a small-in-large pair as ~0 and can't tell the two apart.
    fn covered_by(&self, o: &Rect) -> f64 {
        let ix1 = self.x.max(o.x) as f64;
        let iy1 = self.y.max(o.y) as f64;
        let ix2 = (self.x + self.w).min(o.x + o.w) as f64;
        let iy2 = (self.y + self.h).min(o.y + o.h) as f64;
        if ix2 <= ix1 || iy2 <= iy1 {
            return 0.0;
        }
        let inter = (ix2 - ix1) * (iy2 - iy1);
        inter / (self.w as f64 * self.h as f64).max(1e-9)
    }
}

fn luma(p: [u8; 3]) -> f32 {
    0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32
}

fn sat(p: [u8; 3]) -> f32 {
    let mx = *p.iter().max().unwrap() as f32;
    let mn = *p.iter().min().unwrap() as f32;
    if mx < 1e-3 {
        0.0
    } else {
        (mx - mn) / mx
    }
}

/// Find the video tiles in a frame (full-resolution pixels).
///
/// StreamYard composites the participant feeds on a near-black background, so
/// the tiles are exactly the *bright* regions and the pure-black gaps between
/// them are the borders/margins. A fixed floor just above black separates
/// content from background cleanly — a global/Otsu threshold is a trap here,
/// because it can split *within* the bright content (missing darker tiles) and
/// because the tile borders are the sharpest edges in the frame (so a detail
/// detector connects across them). The remaining large components are kept, and
/// the overlays (thin subscribe band, small mascot, corner shapes) are rejected
/// by aspect ratio and area.
///
/// ponytail: assumes a dark background. A bright branded background would need
/// a different split (e.g. Otsu or colour); add when such layouts appear.
pub fn detect_tiles(frame: &PpmFrame, max_tiles: usize) -> Vec<Rect> {
    let w = frame.width as usize;
    let h = frame.height as usize;
    let lum: Vec<f32> = frame.pixels.iter().map(|p| luma(*p)).collect();
    let threshold = 20.0f32;
    // The StreamYard brand-blue overlays (mascot, subscribe band, corner shapes)
    // are bright and sit flush against the tiles, so a plain brightness mask
    // merges them in. They are strongly blue (high B, low R/G); the participant
    // feeds are natural-colour, so excluding the blue isolates the tiles.
    // ponytail: specific to the brand-blue overlays — a blue-shirt tile loses
    // only its blue pixels (box shrinks slightly), which is fine.
    let is_blue = |i: usize| {
        let p = frame.pixels[i];
        let (r, g, b) = (p[0] as f32, p[1] as f32, p[2] as f32);
        b > 120.0 && b > r + 40.0 && b > g + 20.0
    };

    fn find_root(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    let mut parent: Vec<usize> = (0..(w * h)).collect();
    let on = |i: usize| lum[i] > threshold && !is_blue(i);
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if !on(i) {
                continue;
            }
            let mut roots = Vec::new();
            if x > 0 && on(i - 1) {
                roots.push(find_root(&mut parent, i - 1));
            }
            if y > 0 && on(i - w) {
                roots.push(find_root(&mut parent, i - w));
            }
            if let Some(&first) = roots.first() {
                parent[i] = first;
                for &r2 in roots.iter().skip(1) {
                    let r = find_root(&mut parent, r2);
                    parent[r] = first;
                }
            } else {
                parent[i] = i;
            }
        }
    }
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if !on(i) {
                continue;
            }
            if x + 1 < w && on(i + 1) {
                let a = find_root(&mut parent, i + 1);
                let b = find_root(&mut parent, i);
                parent[a] = b;
            }
            if y + 1 < h && on(i + w) {
                let a = find_root(&mut parent, i + w);
                let b = find_root(&mut parent, i);
                parent[a] = b;
            }
        }
    }

    let mut boxes: HashMap<usize, [i64; 4]> = HashMap::new();
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if !on(i) {
                continue;
            }
            let root = find_root(&mut parent, i);
            let b = boxes
                .entry(root)
                .or_insert([i64::MAX, i64::MAX, i64::MIN, i64::MIN]);
            b[0] = b[0].min(x as i64);
            b[1] = b[1].min(y as i64);
            b[2] = b[2].max(x as i64);
            b[3] = b[3].max(y as i64);
        }
    }

    let area = (w * h) as f64;
    // Small enough to keep the tiny PIP webcams (stream-share column) yet the
    // brand-blue overlays are already excluded by the `is_blue` mask, so this
    // floor mainly drops slivers and dust.
    let min_area = area * 0.015;
    // Participant tiles are near-square to 16:9; reject the very wide/short strips
    // (name bars, subscribe band) by aspect and the small corner graphics (mascot)
    // by area.
    let plausible = |r: &Rect| {
        let ar = r.w as f64 / r.h as f64;
        (ar > 0.4 && ar < 2.6) && (r.w as f64 * r.h as f64) >= min_area
    };
    let mut rects: Vec<Rect> = boxes
        .into_values()
        .map(|[x1, y1, x2, y2]| Rect {
            x: x1.max(0) as u32,
            y: y1.max(0) as u32,
            w: (x2 - x1 + 1).max(1) as u32,
            h: (y2 - y1 + 1).max(1) as u32,
        })
        .filter(|r| plausible(r))
        .collect();

    // Drop boxes that are a duplicate of (IoU) or mostly inside (coverage) a larger
    // kept box — the latter is clutter within a full-frame feed (a backpack, a shelf),
    // not a separate person window. Keep the biggest `max_tiles`.
    rects.sort_by_key(|r| std::cmp::Reverse(r.w as u64 * r.h as u64));
    let mut out = Vec::new();
    for r in rects {
        if out.iter().any(|kept: &Rect| {
            kept.w as u64 * kept.h as u64 > r.w as u64 * r.h as u64
                && (r.iou(kept) > 0.5 || r.covered_by(kept) > 0.7)
        }) {
            continue;
        }
        out.push(r);
        if out.len() == max_tiles {
            break;
        }
    }
    out
}

/// Saturation-weighted 1-D RGB histogram of a region (normalised to sum to 1).
/// Saturated (colourful) pixels — the shirts — are weighted more heavily so they
/// dominate the flat backgrounds, which is what distinguishes the participants.
fn histogram(frame: &PpmFrame, r: &Rect, bins: usize) -> Vec<f32> {
    let w = frame.width as usize;
    let x0 = r.x as usize;
    let y0 = r.y as usize;
    let x1 = (x0 + r.w as usize).min(frame.width as usize);
    let y1 = (y0 + r.h as usize).min(frame.height as usize);
    // Centre-weighting: the person (shirt/face) sits in the middle of the frame,
    // the room (a red gaming chair, etc.) crowds the edges. Falling the weight off
    // toward the edges keeps the shirt — the real distinguisher — in charge instead
    // of the background (which made one PIP read as another participant's shirt).
    let cx = (x0 + x1) as f64 / 2.0;
    let cy = (y0 + y1) as f64 / 2.0;
    let hx = (x1 - x0) as f64 / 2.0;
    let hy = (y1 - y0) as f64 / 2.0;
    let mut h = vec![0f32; bins * 3];
    for y in y0..y1 {
        for x in x0..x1 {
            let p = frame.pixels[y * w + x];
            let nx = if hx > 0.0 { (x as f64 - cx) / hx } else { 0.0 };
            let ny = if hy > 0.0 { (y as f64 - cy) / hy } else { 0.0 };
            let rad = (nx * nx + ny * ny).sqrt().min(1.414) / 1.414; // 0 centre → 1 corner
            let center_f = (1.0 - 0.7 * rad) as f32; // 1.0 centre, ~0.3 edges
            let weight = (0.4 + sat(p) * 2.2) * center_f;
            let bi = ((p[0] as usize * bins) / 256).min(bins - 1);
            let bg = ((p[1] as usize * bins) / 256).min(bins - 1);
            let bb = ((p[2] as usize * bins) / 256).min(bins - 1);
            h[bi] += weight;
            h[bins + bg] += weight;
            h[2 * bins + bb] += weight;
        }
    }
    let s = h.iter().sum::<f32>().max(1.0);
    for v in h.iter_mut() {
        *v /= s;
    }
    h
}

/// Inset a rect by a fraction of its own size on every side. The signatures are
/// centre-weighted to the person, so a tile's histogram should be too — dropping
/// the room/chair at the tile edges that otherwise swamps the shirt colour (a red
/// gaming chair made one PIP look like another participant's red shirt).
fn center_rect(r: &Rect, m: f64) -> Rect {
    let dx = (r.w as f64 * m).round() as u32;
    let dy = (r.h as f64 * m).round() as u32;
    Rect {
        x: r.x.saturating_add(dx),
        y: r.y.saturating_add(dy),
        w: r.w.saturating_sub(2 * dx).max(1),
        h: r.h.saturating_sub(2 * dy).max(1),
    }
}

/// Bhattacharyya similarity in [0,1] (1 = identical).
fn bhattacharyya(a: &[f32], b: &[f32]) -> f32 {
    (0..a.len()).map(|i| (a[i] * b[i]).sqrt()).sum::<f32>()
}

/// "Is there a person in frame?" score: luma variance. A blank/black frame
/// (camera off) scores ~0; a person in a room scores high. Used to pick a
/// reference frame that actually shows the participant.
pub fn content_score(frame: &PpmFrame) -> f32 {
    let n = frame.pixels.len() as f32;
    if n == 0.0 {
        return 0.0;
    }
    let mut mean = 0f32;
    for p in &frame.pixels {
        mean += luma(*p);
    }
    mean /= n;
    let mut var = 0f32;
    for p in &frame.pixels {
        let d = luma(*p) - mean;
        var += d * d;
    }
    var / n
}

/// A participant's appearance signature, sampled from their local feed.
#[derive(Clone)]
pub struct Signature {
    pub id: String,
    pub hist: Vec<f32>,
}

impl Signature {
    pub fn from_frame(id: &str, frame: &PpmFrame, bins: usize) -> Self {
        // Center 80% avoids any letterboxing in the local recording.
        let cx = frame.width * 10 / 100;
        let cy = frame.height * 10 / 100;
        let center = Rect {
            x: cx,
            y: cy,
            w: frame.width - 2 * cx,
            h: frame.height - 2 * cy,
        };
        Signature {
            id: id.to_string(),
            hist: histogram(frame, &center, bins),
        }
    }
}

/// A tile assigned to a participant, with the match similarity.
#[derive(Clone, Debug)]
pub struct TileScore {
    pub participant_id: String,
    pub rect: Rect,
    pub sim: f32,
}

/// A tile is a real participant if its best match is confident (a stable webcam
/// feed, 0.85+) OR clearly the best by a margin (a screen share: it matches its
/// own track by a lot but lower in absolute terms because the shared content
/// scrolls). StreamYard graphics (the intro countdown, promo cards) are close to
/// several people at once — best around 0.77 with a small margin — so they fail
/// both tests and are left as the (cloud) background.
/// ponytail: the margin test can in theory accept a graphic that happens to beat
/// the runner-up by MARGIN at ~NOISE_FLOOR. A per-tile texture/detail check would
/// kill that; add it only if such a graphic actually shows up.
const MATCH_FLOOR: f32 = 0.85;
/// Absolute minimum similarity for a box to be considered at all (noise/dust).
const NOISE_FLOOR: f32 = 0.60;
/// How far a box's best match must beat its runner-up to be accepted below FLOOR.
const MARGIN: f32 = 0.15;

/// For one cloud frame: detect tiles and assign each to the best-matching
/// participant (one tile per participant, greedy on similarity).
pub fn match_tiles(frame: &PpmFrame, sigs: &[Signature], out_w: u32, out_h: u32) -> Vec<TileScore> {
    let bins = 24;
    let tiles = detect_tiles(frame, sigs.len().max(1));
    if tiles.is_empty() {
        return Vec::new();
    }
    let sx = out_w as f64 / frame.width as f64;
    let sy = out_h as f64 / frame.height as f64;

    // Per tile: its best match + the runner-up. A tile is accepted if the best is
    // confident, or clearly the best by a margin (a screen share). A tile whose
    // best is taken by a stronger tile is dropped (left as cloud), never downgraded
    // to a weaker person.
    let mut accepted: Vec<(f32, usize, usize)> = Vec::new(); // (best_sim, tile_idx, best_sig)
    for (ti, t) in tiles.iter().enumerate() {
        let tc = center_rect(t, 0.10);
        let th = histogram(frame, &tc, bins);
        let mut best = (0.0f32, 0usize);
        let mut second = 0.0f32;
        for (si, s) in sigs.iter().enumerate() {
            let sim = bhattacharyya(&th, &s.hist);
            if sim > best.0 {
                second = best.0;
                best = (sim, si);
            } else if sim > second {
                second = sim;
            }
        }
        if best.0 >= MATCH_FLOOR || (best.0 >= NOISE_FLOOR && best.0 - second >= MARGIN) {
            accepted.push((best.0, ti, best.1));
        }
    }
    accepted.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut used_tile = vec![false; tiles.len()];
    let mut used_sig = vec![false; sigs.len()];
    let mut out: Vec<TileScore> = Vec::new();
    for (sim, ti, si) in accepted {
        if used_tile[ti] || used_sig[si] {
            continue;
        }
        used_tile[ti] = true;
        used_sig[si] = true;
        let t = &tiles[ti];
        out.push(TileScore {
            participant_id: sigs[si].id.clone(),
            rect: Rect {
                x: (t.x as f64 * sx).round() as u32,
                y: (t.y as f64 * sy).round() as u32,
                w: ((t.w as f64 * sx).round() as u32).max(1),
                h: ((t.h as f64 * sy).round() as u32).max(1),
            },
            sim,
        });
    }
    out
}

/// Debug: raw detected tiles + the full (tile × signature) similarity matrix
/// (no 0.5 floor, no greedy assignment) so the identity decision is inspectable.
pub fn tile_similarities(frame: &PpmFrame, sigs: &[Signature]) -> (Vec<Rect>, Vec<Vec<f32>>) {
    let bins = 24;
    let tiles = detect_tiles(frame, sigs.len().max(1));
    let mut m = Vec::new();
    for t in &tiles {
        let tc = center_rect(t, 0.10);
        let th = histogram(frame, &tc, bins);
        m.push(sigs.iter().map(|s| bhattacharyya(&th, &s.hist)).collect());
    }
    (tiles, m)
}

/// Two frames have the same *layout* if they have the same number of tiles and
/// every tile in one overlaps a tile in the other (identity ignored).
fn same_layout(a: &[Rect], b: &[Rect]) -> bool {
    if a.is_empty() || a.len() != b.len() {
        return a.is_empty() && b.is_empty();
    }
    a.iter().all(|ra| b.iter().any(|rb| ra.iou(rb) > 0.5))
        && b.iter().all(|rb| a.iter().any(|ra| ra.iou(rb) > 0.5))
}

/// A run of consecutive samples sharing one layout.
struct Run {
    first_t: f64,
    last_t: f64,
    samples: Vec<(f64, Vec<TileScore>)>,
}

/// The classifier: given a cloud time, return the classified tiles there.
/// main.rs implements this by decoding a frame + `match_tiles`.
type Classifier<'a> = dyn Fn(f64) -> Vec<TileScore> + 'a;

/// Bisect the transition time in `(lo, hi)` where the layout flips relative to
/// `canon`. `lo_is_canon` tells which side of the boundary `lo` sits on.
fn bisect_boundary(
    classifier: &Classifier,
    mut lo: f64,
    mut hi: f64,
    canon: &[Rect],
    lo_is_canon: bool,
) -> f64 {
    if hi - lo < 0.25 {
        return (lo + hi) / 2.0;
    }
    for _ in 0..6 {
        let mid = (lo + hi) / 2.0;
        let mid_rects: Vec<Rect> = classifier(mid).iter().map(|t| t.rect).collect();
        let mid_is_canon = same_layout(&mid_rects, canon);
        if mid_is_canon == lo_is_canon {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo < 0.5 {
            break;
        }
    }
    (lo + hi) / 2.0
}

/// Group the coarse samples into layout runs, pick each box's identity from the
/// run's collective evidence, refine the boundaries to the exact transition times,
/// and emit one `SpatialSegment` per (run, box).
pub fn build_timeline(
    coarse: Vec<(f64, Vec<TileScore>)>,
    classifier: &Classifier,
    cloud: &str,
    width: u32,
    height: u32,
    duration: f64,
) -> LayoutTimeline {
    let mut coarse = coarse;
    coarse.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Group into runs by layout structure.
    let mut runs: Vec<Run> = Vec::new();
    for (t, tiles) in coarse {
        let rects: Vec<Rect> = tiles.iter().map(|x| x.rect).collect();
        if let Some(last) = runs.last_mut() {
            let last_rects: Vec<Rect> = last
                .samples
                .last()
                .unwrap()
                .1
                .iter()
                .map(|x| x.rect)
                .collect();
            if same_layout(&last_rects, &rects) {
                last.last_t = t;
                last.samples.push((t, tiles));
                continue;
            }
        }
        runs.push(Run {
            first_t: t,
            last_t: t,
            samples: vec![(t, tiles)],
        });
    }

    // Drop runs with no participants (the intro countdown, any blank state) — no
    // local feed is composited over them, and their neighbours' boundaries are
    // bisected around them.
    // ponytail: assumes blank states only ever sit at the head (the countdown).
    // A blank slide *between* two real layouts would merge them across it; add
    // blank-aware boundaries when such a show appears.
    let runs: Vec<Run> = runs
        .into_iter()
        .filter(|r| r.samples.iter().any(|(_, tiles)| !tiles.is_empty()))
        .collect();
    // A one-sample run is a transition frame (a layout caught mid-animation, e.g.
    // the PIPs sliding into place) that a single sample happened to land on — not a
    // real layout. Drop it; the two neighbours' boundary is then bisected across it.
    // ponytail: a real layout lasts many seconds and lands on 2+ samples so it
    // survives; only sub-spacing transitions drop out.
    let runs: Vec<Run> = runs.into_iter().filter(|r| r.samples.len() >= 2).collect();
    if runs.is_empty() {
        return LayoutTimeline {
            cloud: cloud.to_string(),
            width,
            height,
            duration,
            segments: Vec::new(),
        };
    }

    // Canonical box layout of a run: the boxes from its most-populated sample.
    let canon_boxes = |r: &Run| -> Vec<Rect> {
        r.samples
            .iter()
            .max_by_key(|(_, tiles)| tiles.len())
            .unwrap()
            .1
            .iter()
            .map(|x| x.rect)
            .collect()
    };

    // Shared boundaries: bounds[i] is the boundary *before* run i, so run i is
    // active on [bounds[i], bounds[i + 1]]. Computing each boundary once per
    // adjacent pair (instead of an independent start + end per run) makes the runs
    // exactly contiguous — no gaps (local feeds vanishing) and no overlaps.
    let mut bounds: Vec<f64> = Vec::with_capacity(runs.len() + 1);
    bounds.push(bisect_boundary(
        classifier,
        0.0,
        runs[0].first_t,
        &canon_boxes(&runs[0]),
        false,
    ));
    for i in 1..runs.len() {
        let lo = runs[i - 1].last_t.max(0.0);
        let hi = runs[i].first_t;
        if hi - lo < 0.25 {
            bounds.push((lo + hi) / 2.0);
        } else {
            // `lo` holds the previous run's layout, so it is NOT this run's → lo_is_canon=false.
            bounds.push(bisect_boundary(
                classifier,
                lo,
                hi,
                &canon_boxes(&runs[i]),
                false,
            ));
        }
    }
    bounds.push(duration);

    // Emit one segment per (run, box): identity = the participant with the highest
    // summed similarity across the whole run (robust to per-frame flicker), box =
    // the mean box across that run.
    let mut segments: Vec<SpatialSegment> = Vec::new();
    for (i, run) in runs.iter().enumerate() {
        let start = bounds[i];
        let end = bounds[i + 1];
        if end <= start {
            continue;
        }
        let canon = canon_boxes(run);
        for box_rect in &canon {
            // Aggregate identity evidence + the mean box for this box across the run.
            let mut per_part: HashMap<&str, f32> = HashMap::new();
            let mut acc = [0.0f64; 4];
            let mut cnt = 0.0f64;
            for (_t, tiles) in &run.samples {
                for ts in tiles {
                    if ts.rect.iou(box_rect) > 0.5 {
                        *per_part.entry(ts.participant_id.as_str()).or_insert(0.0) += ts.sim;
                        acc[0] += ts.rect.x as f64;
                        acc[1] += ts.rect.y as f64;
                        acc[2] += ts.rect.w as f64;
                        acc[3] += ts.rect.h as f64;
                        cnt += 1.0;
                    }
                }
            }
            if cnt == 0.0 {
                continue;
            }
            let (best_part, _) = per_part
                .into_iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or_default();
            if best_part.is_empty() {
                continue;
            }
            let mean = Rect {
                x: (acc[0] / cnt).round() as u32,
                y: (acc[1] / cnt).round() as u32,
                w: (acc[2] / cnt).round().max(1.0) as u32,
                h: (acc[3] / cnt).round().max(1.0) as u32,
            };
            segments.push(SpatialSegment {
                start_time: start,
                end_time: end,
                participant_id: best_part.to_string(),
                bounding_box: (mean.x, mean.y, mean.w, mean.h),
            });
        }
    }

    segments.sort_by(|a, b| {
        a.start_time
            .partial_cmp(&b.start_time)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    LayoutTimeline {
        cloud: cloud.to_string(),
        width,
        height,
        duration,
        segments,
    }
}
