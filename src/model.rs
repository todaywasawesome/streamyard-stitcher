use serde::{Deserialize, Serialize};

/// Result of probing a single media file with ffprobe.
#[derive(Debug, Clone, Serialize)]
pub struct ProbeInfo {
    pub path: String,
    pub width: u32,
    pub height: u32,
    pub duration: f64,
    pub has_audio: bool,
    pub video_fps: Option<f64>,
}

/// A single participant's local recording, plus its computed alignment.
#[derive(Debug, Clone, Serialize)]
pub struct Participant {
    /// Stable slug used across the timeline/filtergraph (e.g. `dan-garfield-webcam`).
    pub id: String,
    /// Human name parsed from the filename (e.g. `Dan Garfield`).
    pub name: String,
    /// `webcam` or `screen` (parsed from the StreamYard filename token).
    pub source: String,
    /// Absolute path to the local recording.
    pub file: String,
    /// Duration of the local recording in seconds.
    pub duration: f64,
    /// Cloud timeline (seconds) at which local t=0 lands. `None` until aligned.
    pub offset_secs: Option<f64>,
    /// Fractional sample-rate difference vs the cloud master. `None` until aligned.
    pub drift_per_sec: Option<f64>,
    /// Peak normalized cross-correlation confidence in [0,1]. `None` until aligned.
    pub confidence: Option<f64>,
}

/// One contiguous interval where a participant occupies a fixed on-screen box.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialSegment {
    pub start_time: f64,
    pub end_time: f64,
    pub participant_id: String,
    /// (x, y, width, height) in full-resolution cloud pixels.
    pub bounding_box: (u32, u32, u32, u32),
}

/// The full set of spatial segments — the "layout timeline".
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutTimeline {
    pub cloud: String,
    pub width: u32,
    pub height: u32,
    pub duration: f64,
    pub segments: Vec<SpatialSegment>,
}

/// Per-participant alignment result (the Phase-2 sync report entry).
#[derive(Debug, Clone, Serialize)]
pub struct ParticipantSync {
    pub id: String,
    pub name: String,
    pub source: String,
    pub offset_secs: f64,
    pub drift_per_sec: f64,
    pub confidence: f64,
}

/// The Phase-2 synchronization report.
#[derive(Debug, Clone, Serialize)]
pub struct SyncReport {
    pub cloud: String,
    pub cloud_duration: f64,
    pub sample_rate: u32,
    pub participants: Vec<ParticipantSync>,
}
