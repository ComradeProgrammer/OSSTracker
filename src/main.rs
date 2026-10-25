// No console window in Windows release builds.
#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

use mytracker::app::App;
use mytracker::camera::{CameraHandle, CameraRequest};
use mytracker::config::Config;
use mytracker::hotkeys::{spawn_joystick, JoyState};
use mytracker::output::{self, OutputState};
use mytracker::tracker::{self, Shared};
use parking_lot::Mutex;
use std::sync::Arc;

fn init_logging() {
    let mut builder = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"));
    if let Some(path) = mytracker::log_path() {
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        if let Ok(file) = std::fs::File::create(&path) {
            builder.target(env_logger::Target::Pipe(Box::new(file)));
        }
    }
    let _ = builder.try_init();
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        log::error!("panic: {info}");
        default_hook(info);
    }));
}

fn main() -> eframe::Result {
    init_logging();
    log::info!("MyTracker {} starting", env!("CARGO_PKG_VERSION"));

    // macOS asks for camera permission here; on other platforms this is a no-op.
    nokhwa::nokhwa_initialize(|granted| log::info!("camera permission granted: {granted}"));

    let cfg = Config::load();
    let shared = Shared::new(cfg.clone());
    let camera = CameraHandle::new();
    camera.start(CameraRequest { name: cfg.camera_name.clone(), mode: cfg.camera_mode });
    tracker::spawn(shared.clone(), camera.frames.clone());
    let out_state = Arc::new(Mutex::new(OutputState::default()));
    output::spawn(shared.clone(), out_state.clone());
    let joy = Arc::new(Mutex::new(JoyState::default()));
    spawn_joystick(shared.clone(), joy.clone());

    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("MyTracker")
            .with_inner_size([1120.0, 760.0])
            .with_min_inner_size([720.0, 480.0]),
        persist_window: true,
        ..Default::default()
    };
    let shared2 = shared.clone();
    let r = eframe::run_native(
        "MyTracker",
        options,
        Box::new(move |cc| Ok(Box::new(App::new(cc, shared2, camera, out_state, joy)))),
    );
    shared.quit.store(true, std::sync::atomic::Ordering::SeqCst);
    r
}
