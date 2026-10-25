#![allow(clippy::needless_range_loop)]

pub mod app;
pub mod camera;
pub mod config;
pub mod face_model;
pub mod filter;
pub mod hotkeys;
pub mod image_ops;
pub mod models;
pub mod output;
pub mod pose;
pub mod tracker;

/// Log file location (next to the config).
pub fn log_path() -> Option<std::path::PathBuf> {
    config::config_path().and_then(|p| p.parent().map(|d| d.join("mytracker.log")))
}
