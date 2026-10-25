//! egui front end: camera preview with overlay, live pose readout, and all settings.

use crate::camera::{CameraHandle, CameraRequest};
use crate::config::{AxisConfig, Config, JoyBinding};
use crate::filter::map_axis;
use crate::hotkeys::{Hotkeys, JoyState};
use crate::output::{freetrack, OutputState};
use crate::pose::AXIS_NAMES;
use crate::tracker::Shared;
use eframe::egui::{self, Color32, Pos2, Rect, Stroke, Vec2};
use parking_lot::Mutex;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(PartialEq, Clone, Copy)]
enum JoyTarget {
    Recenter,
    Pause,
}

pub struct App {
    shared: Arc<Shared>,
    camera: CameraHandle,
    out: Arc<Mutex<OutputState>>,
    joy: Arc<Mutex<JoyState>>,
    hotkeys: Hotkeys,
    cfg: Config,
    cam_req: CameraRequest,
    texture: Option<egui::TextureHandle>,
    texture_seq: u64,
    dirty_since: Option<Instant>,
    hotkey_edit: (String, String),
    joy_learning: Option<JoyTarget>,
}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>, shared: Arc<Shared>, camera: CameraHandle, out: Arc<Mutex<OutputState>>, joy: Arc<Mutex<JoyState>>) -> Self {
        let ctx = cc.egui_ctx.clone();
        *shared.notify.lock() = Some(Box::new(move || ctx.request_repaint()));
        let cfg = shared.config.read().clone();
        let mut hotkeys = Hotkeys::new(shared.clone());
        hotkeys.update(&cfg.hotkey_recenter, &cfg.hotkey_pause);
        let cam_req = CameraRequest { name: cfg.camera_name.clone(), mode: cfg.camera_mode };
        let hotkey_edit = (cfg.hotkey_recenter.clone(), cfg.hotkey_pause.clone());
        Self {
            shared,
            camera,
            out,
            joy,
            hotkeys,
            cfg,
            cam_req,
            texture: None,
            texture_seq: 0,
            dirty_since: None,
            hotkey_edit,
            joy_learning: None,
        }
    }

    fn apply_config(&mut self) {
        self.cfg = self.cfg.clone().sanitized();
        if *self.shared.config.read() != self.cfg {
            *self.shared.config.write() = self.cfg.clone();
            self.dirty_since.get_or_insert(Instant::now());
        }
        let req = CameraRequest { name: self.cfg.camera_name.clone(), mode: self.cfg.camera_mode };
        if req != self.cam_req {
            self.cam_req = req.clone();
            self.camera.start(req);
        }
        self.hotkeys.update(&self.cfg.hotkey_recenter, &self.cfg.hotkey_pause);
        if self.dirty_since.is_some_and(|t| t.elapsed() > Duration::from_secs(1)) {
            self.cfg.save();
            self.dirty_since = None;
        }
        // Finish "press a joystick button" binding.
        if let Some(target) = self.joy_learning {
            let mut j = self.joy.lock();
            if let Some(b) = j.learned.take() {
                match target {
                    JoyTarget::Recenter => self.cfg.joy_recenter = Some(b),
                    JoyTarget::Pause => self.cfg.joy_pause = Some(b),
                }
                self.joy_learning = None;
            } else if !j.learning {
                self.joy_learning = None;
            }
        }
    }

    fn top_bar(&mut self, ui: &mut egui::Ui) {
        let st = self.shared.state.lock().clone();
        let paused = self.shared.paused.load(Ordering::SeqCst);
        ui.horizontal(|ui| {
            if ui.add(egui::Button::new("Recenter").min_size(Vec2::new(100.0, 28.0))).clicked() {
                self.shared.request_recenter();
            }
            let label = if paused { "Resume" } else { "Pause" };
            if ui.add(egui::Button::new(label).min_size(Vec2::new(90.0, 28.0))).clicked() {
                self.shared.toggle_pause();
            }
            ui.separator();
            let (color, text) = if paused {
                (Color32::from_rgb(230, 170, 30), "PAUSED (view frozen)".to_string())
            } else if st.tracking {
                (Color32::from_rgb(60, 190, 90), st.message.clone())
            } else {
                (Color32::from_rgb(220, 80, 70), st.message.clone())
            };
            ui.colored_label(color, egui::RichText::new(text).strong());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(format!("{:.0} fps", st.fps));
            });
        });
    }

    fn preview(&mut self, ui: &mut egui::Ui) {
        let cam = self.camera.status.lock().clone();
        if let Some(err) = &cam.error {
            ui.colored_label(Color32::from_rgb(230, 90, 70), format!("Camera: {err}"));
        }
        let p = self.shared.preview.lock().clone();
        if p.seq != self.texture_seq && p.width > 0 {
            let img = egui::ColorImage::from_rgba_unmultiplied([p.width, p.height], &p.rgba);
            match &mut self.texture {
                Some(t) => t.set(img, egui::TextureOptions::LINEAR),
                None => self.texture = Some(ui.ctx().load_texture("preview", img, egui::TextureOptions::LINEAR)),
            }
            self.texture_seq = p.seq;
        }
        let avail = ui.available_size();
        let aspect = if p.width > 0 { p.width as f32 / p.height as f32 } else { 16.0 / 9.0 };
        let w = avail.x.min((avail.y - 190.0).max(120.0) * aspect).max(160.0);
        let (rect, _) = ui.allocate_exact_size(Vec2::new(w, w / aspect), egui::Sense::hover());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 4.0, Color32::from_gray(20));
        if !self.cfg.show_preview {
            painter.text(rect.center(), egui::Align2::CENTER_CENTER, "Preview off", egui::FontId::proportional(16.0), Color32::GRAY);
        } else if let Some(t) = &self.texture {
            painter.image(t.id(), rect, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
            let at = |q: (f32, f32)| Pos2::new(rect.min.x + q.0 * rect.width(), rect.min.y + q.1 * rect.height());
            if let Some(roi) = p.roi {
                let pts: Vec<Pos2> = roi.iter().map(|&q| at(q)).collect();
                painter.add(egui::Shape::closed_line(pts, Stroke::new(1.5, Color32::from_rgba_unmultiplied(255, 255, 255, 120))));
            }
            for &q in &p.landmarks {
                painter.circle_filled(at(q), 1.0, Color32::from_rgb(80, 220, 120));
            }
            for &q in &p.rigid {
                painter.circle_filled(at(q), 2.0, Color32::from_rgb(255, 90, 60));
            }
            if let Some(a) = p.axes {
                let o = at(a[0]);
                painter.line_segment([o, at(a[1])], Stroke::new(3.0, Color32::from_rgb(230, 60, 60)));
                painter.line_segment([o, at(a[2])], Stroke::new(3.0, Color32::from_rgb(60, 200, 60)));
                painter.line_segment([o, at(a[3])], Stroke::new(3.0, Color32::from_rgb(70, 120, 255)));
            }
        } else {
            painter.text(rect.center(), egui::Align2::CENTER_CENTER, "Waiting for camera…", egui::FontId::proportional(16.0), Color32::GRAY);
        }
    }

    fn readout(&self, ui: &mut egui::Ui) {
        let st = self.shared.state.lock().clone();
        let out = self.out.lock().clone();
        egui::Grid::new("readout").num_columns(3).spacing([12.0, 4.0]).show(ui, |ui| {
            ui.strong("Axis");
            ui.strong("Head");
            ui.strong("Sent to game");
            ui.end_row();
            for i in 0..6 {
                let unit = if i < 3 { "°" } else { " cm" };
                ui.label(AXIS_NAMES[i]);
                ui.monospace(format!("{:>7.1}{unit}", out.smoothed[i]));
                let lim = (self.cfg.axes[i].output_limit as f64).max(1.0);
                bar(ui, out.mapped[i] / lim, format!("{:>7.1}{unit}", out.mapped[i]));
                ui.end_row();
            }
        });
        if !st.tracking && st.lost_since.is_some() {
            ui.weak("Face lost: holding the last view, then easing back to centre.");
        }
    }

    fn settings(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            egui::CollapsingHeader::new("Camera").default_open(true).show(ui, |ui| self.camera_settings(ui));
            egui::CollapsingHeader::new("Output").default_open(true).show(ui, |ui| self.output_settings(ui));
            egui::CollapsingHeader::new("Response curves").default_open(false).show(ui, |ui| self.curve_settings(ui));
            egui::CollapsingHeader::new("Smoothing").default_open(false).show(ui, |ui| self.smoothing_settings(ui));
            egui::CollapsingHeader::new("Hotkeys & buttons").default_open(false).show(ui, |ui| self.hotkey_settings(ui));
            egui::CollapsingHeader::new("Diagnostics").default_open(false).show(ui, |ui| self.diagnostics(ui));
        });
    }

    fn camera_settings(&mut self, ui: &mut egui::Ui) {
        let cam = self.camera.status.lock().clone();
        let current = if self.cfg.camera_name.is_empty() { "First camera".to_string() } else { self.cfg.camera_name.clone() };
        egui::ComboBox::from_label("Device").selected_text(current).show_ui(ui, |ui| {
            ui.selectable_value(&mut self.cfg.camera_name, String::new(), "First camera");
            for d in &cam.devices {
                ui.selectable_value(&mut self.cfg.camera_name, d.clone(), d);
            }
        });
        let mode_text = match self.cfg.camera_mode {
            None => "Auto (full field of view)".to_string(),
            Some((w, h, f)) => format!("{w}x{h} @ {f} fps"),
        };
        egui::ComboBox::from_label("Mode").selected_text(mode_text).show_ui(ui, |ui| {
            ui.selectable_value(&mut self.cfg.camera_mode, None, "Auto (full field of view)");
            for m in &cam.modes {
                ui.selectable_value(&mut self.cfg.camera_mode, Some((m.width, m.height, m.fps)), m.to_string());
            }
        });
        if let Some((m, fmt)) = &cam.active {
            ui.weak(format!("Active: {m} ({fmt})"));
        }
        ui.add(egui::Slider::new(&mut self.cfg.camera_hfov, 40.0..=120.0).text("Horizontal FOV (°)"))
            .on_hover_text("Your camera's horizontal field of view. Typical webcams are 60–80°. Matters most when you sit close to the camera.");
        ui.checkbox(&mut self.cfg.mirror_preview, "Mirror preview");
        ui.checkbox(&mut self.cfg.show_preview, "Show preview");
        ui.checkbox(&mut self.cfg.show_landmarks, "Show landmarks");
    }

    fn output_settings(&mut self, ui: &mut egui::Ui) {
        let out = self.out.lock().clone();
        ui.checkbox(&mut self.cfg.udp.enabled, "opentrack UDP");
        ui.add_enabled_ui(self.cfg.udp.enabled, |ui| {
            ui.horizontal(|ui| {
                ui.label("Host");
                ui.add(egui::TextEdit::singleline(&mut self.cfg.udp.host).desired_width(120.0));
                ui.label("Port");
                ui.add(egui::DragValue::new(&mut self.cfg.udp.port).range(1..=65535));
            });
            ui.checkbox(&mut self.cfg.udp.send_mapped, "Send with my response curves")
                .on_hover_text("Off: send raw head pose and do the mapping in opentrack (opentrack input = \"UDP over network\").");
            if let Some(e) = &out.udp_error {
                ui.colored_label(Color32::from_rgb(230, 90, 70), e);
            }
        });
        ui.separator();
        ui.add_enabled_ui(cfg!(windows), |ui| {
            ui.checkbox(&mut self.cfg.freetrack.enabled, "FreeTrack / TrackIR (DCS direct)")
                .on_hover_text("DCS reads this through NPClient64.dll. opentrack only needs to be installed, not running.");
            ui.add_enabled_ui(self.cfg.freetrack.enabled, |ui| {
                ui.checkbox(&mut self.cfg.freetrack.set_registry, "Point TrackIR registry at NPClient64.dll folder");
                ui.horizontal(|ui| {
                    ui.label("NPClient folder");
                    ui.add(egui::TextEdit::singleline(&mut self.cfg.freetrack.npclient_dir).hint_text("auto-detect").desired_width(200.0));
                });
                if self.cfg.freetrack.npclient_dir.is_empty() {
                    match freetrack::find_npclient_dir("") {
                        Some(d) => ui.weak(format!("Found: {}", d.display())),
                        None => ui.weak("Not found yet (looks in opentrack's install folder and next to this program)"),
                    };
                }
            });
        });
        if !out.freetrack_status.is_empty() {
            ui.weak(&out.freetrack_status);
        }
    }

    fn curve_settings(&mut self, ui: &mut egui::Ui) {
        ui.weak("How far the in-game view turns for a given head movement. Curve > 0 gives finer control near centre.");
        let st = self.out.lock().clone();
        for i in 0..6 {
            let unit = if i < 3 { "°" } else { "cm" };
            egui::CollapsingHeader::new(AXIS_NAMES[i]).id_salt(("axis", i)).default_open(i < 2).show(ui, |ui| {
                let a: &mut AxisConfig = &mut self.cfg.axes[i];
                ui.horizontal(|ui| {
                    ui.checkbox(&mut a.enabled, "Enabled");
                    ui.checkbox(&mut a.invert, "Invert");
                });
                let max_in = if i < 3 { 90.0 } else { 50.0 };
                let max_out = if i < 3 { 180.0 } else { 100.0 };
                ui.add(egui::Slider::new(&mut a.input_limit, 1.0..=max_in).text(format!("head {unit}")));
                ui.add(egui::Slider::new(&mut a.output_limit, 0.0..=max_out).text(format!("game {unit}")));
                ui.add(egui::Slider::new(&mut a.curve, 0.0..=1.0).text("curve"));
                ui.add(egui::Slider::new(&mut a.deadzone, 0.0..=(a.input_limit * 0.5)).text(format!("deadzone {unit}")));
                let x = if a.invert { -st.smoothed[i] } else { st.smoothed[i] };
                curve_plot(ui, a, x);
            });
        }
        if ui.button("Reset curves to defaults").clicked() {
            self.cfg.axes = Config::default().axes;
        }
    }

    fn smoothing_settings(&mut self, ui: &mut egui::Ui) {
        let s = &mut self.cfg.smoothing;
        ui.weak("Rotation: steadiness when still vs. responsiveness when moving.");
        ui.add(egui::Slider::new(&mut s.rot_min_cutoff, 0.05..=5.0).logarithmic(true).text("still cutoff (Hz)"))
            .on_hover_text("Lower = steadier when holding still, but slower to start moving.");
        ui.add(egui::Slider::new(&mut s.rot_beta, 0.0..=1.0).text("speed response"))
            .on_hover_text("Higher = less lag during fast head turns.");
        ui.weak("Position:");
        ui.add(egui::Slider::new(&mut s.pos_min_cutoff, 0.05..=5.0).logarithmic(true).text("still cutoff (Hz)"));
        ui.add(egui::Slider::new(&mut s.pos_beta, 0.0..=2.0).text("speed response"));
        ui.separator();
        ui.add(egui::Slider::new(&mut s.output_smoothing_ms, 0.0..=100.0).text("frame interpolation (ms)"))
            .on_hover_text("Blends between camera frames so the view moves smoothly at the output rate.");
        ui.add(egui::Slider::new(&mut s.prediction, 0.0..=1.0).text("prediction"))
            .on_hover_text("Extrapolates motion to hide the camera delay. Lower it if the view overshoots.");
        ui.add(egui::Slider::new(&mut self.cfg.output_hz, 30..=250).text("output rate (Hz)"));
        ui.add(egui::Slider::new(&mut self.cfg.hold_on_lost_secs, 0.0..=10.0).text("hold view when face lost (s)"));
        if ui.button("Reset smoothing to defaults").clicked() {
            let d = Config::default();
            self.cfg.smoothing = d.smoothing;
            self.cfg.output_hz = d.output_hz;
            self.cfg.hold_on_lost_secs = d.hold_on_lost_secs;
        }
    }

    fn hotkey_settings(&mut self, ui: &mut egui::Ui) {
        ui.weak("Keyboard hotkeys work while the game has focus. Examples: F12, Shift+F12, Ctrl+Alt+KeyR");
        egui::Grid::new("hk").num_columns(2).show(ui, |ui| {
            ui.label("Recenter");
            let r = ui.add(egui::TextEdit::singleline(&mut self.hotkey_edit.0).desired_width(140.0));
            if r.lost_focus() {
                self.cfg.hotkey_recenter = self.hotkey_edit.0.clone();
            }
            ui.end_row();
            ui.label("Pause");
            let r = ui.add(egui::TextEdit::singleline(&mut self.hotkey_edit.1).desired_width(140.0));
            if r.lost_focus() {
                self.cfg.hotkey_pause = self.hotkey_edit.1.clone();
            }
            ui.end_row();
        });
        if let Some(e) = &self.hotkeys.error {
            ui.colored_label(Color32::from_rgb(230, 90, 70), e);
        }
        ui.separator();
        ui.weak("Joystick / HOTAS buttons:");
        for (target, label) in [(JoyTarget::Recenter, "Recenter"), (JoyTarget::Pause, "Pause")] {
            ui.horizontal(|ui| {
                ui.label(label);
                let binding = match target {
                    JoyTarget::Recenter => &self.cfg.joy_recenter,
                    JoyTarget::Pause => &self.cfg.joy_pause,
                };
                let text = binding_text(binding);
                if self.joy_learning == Some(target) {
                    ui.colored_label(Color32::from_rgb(230, 170, 30), "Press a button…");
                    if ui.button("Cancel").clicked() {
                        self.joy_learning = None;
                        self.joy.lock().learning = false;
                    }
                } else {
                    ui.monospace(text);
                    if ui.button("Bind").clicked() {
                        self.joy_learning = Some(target);
                        let mut j = self.joy.lock();
                        j.learned = None;
                        j.learning = true;
                    }
                    if ui.button("Clear").clicked() {
                        match target {
                            JoyTarget::Recenter => self.cfg.joy_recenter = None,
                            JoyTarget::Pause => self.cfg.joy_pause = None,
                        }
                    }
                }
            });
        }
        let j = self.joy.lock();
        if let Some(e) = &j.error {
            ui.colored_label(Color32::from_rgb(230, 90, 70), e);
        } else if j.devices.is_empty() {
            ui.weak("No joysticks detected.");
        } else {
            ui.weak(format!("Detected: {}", j.devices.join(", ")));
        }
    }

    fn diagnostics(&mut self, ui: &mut egui::Ui) {
        let st = self.shared.state.lock().clone();
        let cam = self.camera.status.lock().clone();
        egui::Grid::new("diag").num_columns(2).show(ui, |ui| {
            let row = |ui: &mut egui::Ui, k: &str, v: String| {
                ui.label(k);
                ui.monospace(v);
                ui.end_row();
            };
            row(ui, "Camera fps", format!("{:.1}", cam.fps));
            row(ui, "Frame decode", format!("{:.1} ms", cam.decode_ms));
            row(ui, "Tracking fps", format!("{:.1}", st.fps));
            row(ui, "Face detection", format!("{:.1} ms", st.timings.detect_ms));
            row(ui, "Landmarks", format!("{:.1} ms", st.timings.landmark_ms));
            row(ui, "Pose solve", format!("{:.2} ms", st.timings.pose_ms));
            row(ui, "Total per frame", format!("{:.1} ms", st.timings.total_ms));
            row(ui, "Fit error", format!("{:.1} px", st.reproj_px));
            row(ui, "Distance to camera", format!("{:.0} cm", st.distance_cm));
        });
        if let Some(p) = crate::config::config_path() {
            ui.weak(format!("Config: {}", p.display()));
        }
        if let Some(p) = crate::log_path() {
            ui.weak(format!("Log: {}", p.display()));
        }
    }
}

fn binding_text(b: &Option<JoyBinding>) -> String {
    match b {
        Some(b) => format!("{} · button {}", b.device, b.code),
        None => "not set".into(),
    }
}

/// Centred horizontal bar for a value in -1..1.
fn bar(ui: &mut egui::Ui, frac: f64, text: String) {
    let (rect, _) = ui.allocate_exact_size(Vec2::new(180.0, 16.0), egui::Sense::hover());
    let p = ui.painter_at(rect);
    p.rect_filled(rect, 3.0, ui.visuals().extreme_bg_color);
    let c = rect.center().x;
    let x = c + (frac.clamp(-1.0, 1.0) as f32) * rect.width() * 0.5;
    let r = Rect::from_min_max(Pos2::new(c.min(x), rect.min.y + 2.0), Pos2::new(c.max(x), rect.max.y - 2.0));
    p.rect_filled(r, 2.0, ui.visuals().selection.bg_fill);
    p.line_segment([Pos2::new(c, rect.min.y), Pos2::new(c, rect.max.y)], Stroke::new(1.0, Color32::GRAY));
    p.text(rect.center(), egui::Align2::CENTER_CENTER, text, egui::FontId::monospace(11.0), ui.visuals().text_color());
}

fn curve_plot(ui: &mut egui::Ui, a: &AxisConfig, x: f64) {
    let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width().min(260.0), 90.0), egui::Sense::hover());
    let p = ui.painter_at(rect);
    p.rect_filled(rect, 3.0, ui.visuals().extreme_bg_color);
    let xmax = a.input_limit as f64 * 1.2;
    let ymax = (a.output_limit as f64).max(1.0) * 1.05;
    let to = |u: f64, v: f64| Pos2::new(rect.min.x + (u / xmax) as f32 * rect.width(), rect.max.y - (v / ymax) as f32 * rect.height());
    let pts: Vec<Pos2> = (0..=60).map(|k| {
        let u = xmax * k as f64 / 60.0;
        to(u, map_axis(u, a))
    }).collect();
    p.add(egui::Shape::line(pts, Stroke::new(2.0, ui.visuals().selection.bg_fill)));
    let ux = x.abs().min(xmax);
    p.circle_filled(to(ux, map_axis(ux, a)), 4.0, Color32::from_rgb(255, 120, 60));
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::Panel::top("top").show(ui, |ui| {
            ui.add_space(4.0);
            self.top_bar(ui);
            ui.add_space(4.0);
        });
        egui::Panel::right("settings").resizable(true).default_size(340.0).min_size(280.0).show(ui, |ui| self.settings(ui));
        egui::CentralPanel::default().show(ui, |ui| {
            self.preview(ui);
            ui.add_space(6.0);
            self.readout(ui);
        });
        self.apply_config();
        // Keep the readout live even without new camera frames.
        ui.ctx().request_repaint_after(Duration::from_millis(50));
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.shared.quit.store(true, Ordering::SeqCst);
        self.camera.stop();
        self.cfg.save();
    }
}
