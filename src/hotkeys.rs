//! Recentre / pause triggers: global keyboard hotkeys (work while the game has focus) and
//! joystick/HOTAS buttons.

use crate::config::JoyBinding;
use crate::tracker::Shared;
use global_hotkey::hotkey::HotKey;
use global_hotkey::{GlobalHotKeyEvent, GlobalHotKeyManager, HotKeyState};
use parking_lot::Mutex;
use std::str::FromStr;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Owns the OS hotkey registrations. Must be created and updated on the main (UI) thread.
pub struct Hotkeys {
    manager: Option<GlobalHotKeyManager>,
    registered: Vec<HotKey>,
    current: (String, String),
    ids: Arc<(AtomicU32, AtomicU32)>,
    pub error: Option<String>,
}

impl Hotkeys {
    pub fn new(shared: Arc<Shared>) -> Self {
        let ids = Arc::new((AtomicU32::new(0), AtomicU32::new(0)));
        let manager = match GlobalHotKeyManager::new() {
            Ok(m) => Some(m),
            Err(e) => {
                log::warn!("global hotkeys unavailable: {e}");
                None
            }
        };
        // Handle events on a dedicated thread so they work while the window is hidden or unfocused.
        let ids2 = ids.clone();
        let _ = std::thread::Builder::new().name("hotkeys".into()).spawn(move || {
            let rx = GlobalHotKeyEvent::receiver();
            while !shared.quit.load(Ordering::SeqCst) {
                if let Ok(ev) = rx.recv_timeout(Duration::from_millis(200)) {
                    if ev.state() != HotKeyState::Pressed {
                        continue;
                    }
                    if ev.id() == ids2.0.load(Ordering::SeqCst) {
                        shared.request_recenter();
                    } else if ev.id() == ids2.1.load(Ordering::SeqCst) {
                        shared.toggle_pause();
                    }
                }
            }
        });
        let error = manager.is_none().then(|| "Global hotkeys are not available on this system".to_string());
        Self { manager, registered: Vec::new(), current: (String::new(), String::new()), ids, error }
    }

    /// Register the configured hotkeys if they changed.
    pub fn update(&mut self, recenter: &str, pause: &str) {
        let wanted = (recenter.to_string(), pause.to_string());
        if wanted == self.current {
            return;
        }
        self.current = wanted;
        let Some(m) = &self.manager else { return };
        for hk in self.registered.drain(..) {
            let _ = m.unregister(hk);
        }
        let mut errors = Vec::new();
        for (i, s) in [recenter, pause].iter().enumerate() {
            if s.trim().is_empty() {
                continue;
            }
            match HotKey::from_str(s.trim()) {
                Ok(hk) => match m.register(hk) {
                    Ok(()) => {
                        if i == 0 {
                            self.ids.0.store(hk.id(), Ordering::SeqCst);
                        } else {
                            self.ids.1.store(hk.id(), Ordering::SeqCst);
                        }
                        self.registered.push(hk);
                    }
                    Err(e) => errors.push(format!("\"{s}\": {e}")),
                },
                Err(e) => errors.push(format!("\"{s}\" is not a valid hotkey ({e})")),
            }
        }
        self.error = (!errors.is_empty()).then(|| errors.join("; "));
    }
}

/// Joystick state shared with the UI (for "press a button to bind").
#[derive(Default)]
pub struct JoyState {
    pub devices: Vec<String>,
    /// Set by the UI: the next button press is stored in `learned` instead of triggering actions.
    pub learning: bool,
    pub learned: Option<JoyBinding>,
    pub error: Option<String>,
}

pub fn spawn_joystick(shared: Arc<Shared>, joy: Arc<Mutex<JoyState>>) {
    let _ = std::thread::Builder::new().name("joystick".into()).spawn(move || {
        let mut gilrs = match gilrs::Gilrs::new() {
            Ok(g) => g,
            Err(gilrs::Error::NotImplemented(g)) => g,
            Err(e) => {
                joy.lock().error = Some(format!("joystick support unavailable: {e}"));
                return;
            }
        };
        let mut refresh = std::time::Instant::now() - Duration::from_secs(10);
        while !shared.quit.load(Ordering::SeqCst) {
            if refresh.elapsed() > Duration::from_secs(2) {
                joy.lock().devices = gilrs.gamepads().map(|(_, g)| g.name().to_string()).collect();
                refresh = std::time::Instant::now();
            }
            let Some(ev) = gilrs.next_event_blocking(Some(Duration::from_millis(200))) else { continue };
            let gilrs::EventType::ButtonPressed(_, code) = ev.event else { continue };
            let device = gilrs.gamepad(ev.id).name().to_string();
            let pressed = JoyBinding { device, code: code.into_u32() };
            let mut j = joy.lock();
            if j.learning {
                j.learned = Some(pressed);
                j.learning = false;
                continue;
            }
            drop(j);
            let cfg = shared.config.read();
            let matches = |b: &Option<JoyBinding>| b.as_ref().is_some_and(|b| b.code == pressed.code && b.device == pressed.device);
            if matches(&cfg.joy_recenter) {
                shared.request_recenter();
            } else if matches(&cfg.joy_pause) {
                shared.toggle_pause();
            }
        }
    });
}
