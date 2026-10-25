//! FreeTrack shared-memory protocol (Windows), the interface opentrack's NPClient64.dll reads.
//! DCS loads NPClient64.dll from the folder named in the TrackIR registry key, and that DLL
//! reads the pose we write here. Layout and game-table handshake follow opentrack's
//! `proto-ft` (ISC licence).

use crate::config::FreetrackConfig;

static GAMES_CSV: &str = include_str!("../../models/freetrack_games.csv");

/// Look up a game's 8-byte encryption table by its TrackIR id. Returns (name, table).
pub fn game_table(id: i32) -> Option<(String, [u8; 8])> {
    if id == 0 {
        return None;
    }
    let id_s = id.to_string();
    for line in GAMES_CSV.lines() {
        let cols: Vec<&str> = line.trim_end_matches('\r').split(';').collect();
        if cols.len() != 8 || cols[6] != id_s {
            continue;
        }
        let mut table = [0u8; 8];
        let ftn = cols[7];
        if cols[3] != "V160" && ftn.len() == 22 {
            let b: Option<Vec<u8>> = (0..11).map(|i| u8::from_str_radix(&ftn[2 * i..2 * i + 2], 16).ok()).collect();
            if let Some(b) = b {
                // Byte order as in opentrack's csv.cpp sscanf: fuzz, fuzz, t3 t2 t1 t0, t7 t6 t5 t4, fuzz.
                table = [b[5], b[4], b[3], b[2], b[9], b[8], b[7], b[6]];
            }
        }
        return Some((cols[1].to_string(), table));
    }
    None
}

/// FreeTrack angles/positions as written to shared memory, from mapped opentrack-convention values.
/// Returns [yaw, pitch, roll (rad), x, y, z (mm)].
pub fn ft_values(v: &[f64; 6]) -> [f32; 6] {
    let d2r = std::f64::consts::PI / 180.0;
    // Falcon BMS makes a "bump" at exactly 90° pitch (same workaround as opentrack).
    let pitch = if (v[1] - 90.0).abs() < 0.15 { 89.86 } else { v[1] };
    [
        (-v[0] * d2r) as f32,
        (-pitch * d2r) as f32,
        (v[2] * d2r) as f32,
        (v[3] * 10.0) as f32,
        (v[4] * 10.0) as f32,
        (v[5] * 10.0) as f32,
    ]
}

#[cfg(windows)]
pub use imp::FreeTrack;

#[cfg(not(windows))]
#[derive(Default)]
pub struct FreeTrack;

#[cfg(not(windows))]
impl FreeTrack {
    pub fn send(&mut self, _mapped: &[f64; 6], _raw: &[f64; 6], _cfg: &FreetrackConfig) -> String {
        "FreeTrack/TrackIR output is only available on Windows (use UDP here).".into()
    }
    pub fn disable(&mut self) -> String {
        "Off".into()
    }
}

/// Folders that may contain an FT-compatible NPClient64.dll, best first.
pub fn npclient_candidates() -> Vec<std::path::PathBuf> {
    let mut v = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(d) = exe.parent() {
            v.push(d.to_path_buf());
            v.push(d.join("npclient"));
        }
    }
    for var in ["ProgramFiles", "ProgramFiles(x86)", "ProgramW6432"] {
        if let Ok(p) = std::env::var(var) {
            v.push(std::path::Path::new(&p).join("opentrack").join("modules"));
        }
    }
    if let Ok(p) = std::env::var("LOCALAPPDATA") {
        v.push(std::path::Path::new(&p).join("Programs").join("opentrack").join("modules"));
    }
    v
}

pub fn find_npclient_dir(configured: &str) -> Option<std::path::PathBuf> {
    if !configured.trim().is_empty() {
        let p = std::path::PathBuf::from(configured.trim());
        return p.join("NPClient64.dll").exists().then_some(p);
    }
    npclient_candidates().into_iter().find(|d| d.join("NPClient64.dll").exists())
}

#[cfg(windows)]
mod imp {
    use super::*;
    use std::sync::atomic::{AtomicI32, AtomicU32, Ordering};
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Memory::{
        CreateFileMappingW, MapViewOfFile, UnmapViewOfFile, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, PAGE_READWRITE,
    };
    use windows_sys::Win32::System::Threading::CreateMutexW;

    // FTHeap layout (all fields 4 bytes).
    const OFF_DATA_ID: usize = 0;
    const OFF_CAM_W: usize = 4;
    const OFF_CAM_H: usize = 8;
    const OFF_YAW: usize = 12; // yaw, pitch, roll, x, y, z
    const OFF_RAW: usize = 36; // raw yaw, pitch, roll, x, y, z
    const OFF_GAME_ID: usize = 92;
    const OFF_TABLE: usize = 96;
    const OFF_GAME_ID2: usize = 104;
    const HEAP_SIZE: usize = 108;

    fn wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(std::iter::once(0)).collect()
    }

    struct Shm {
        mapping: HANDLE,
        mutex: HANDLE,
        view: MEMORY_MAPPED_VIEW_ADDRESS,
    }

    // The mapping is process-global OS memory; access goes through atomics.
    unsafe impl Send for Shm {}

    impl Shm {
        fn open() -> Result<Self, String> {
            unsafe {
                let mutex = CreateMutexW(std::ptr::null(), 0, wide("FT_Mutext").as_ptr());
                let mapping = CreateFileMappingW(
                    INVALID_HANDLE_VALUE,
                    std::ptr::null(),
                    PAGE_READWRITE,
                    0,
                    HEAP_SIZE as u32,
                    wide("FT_SharedMem").as_ptr(),
                );
                if mapping.is_null() {
                    if !mutex.is_null() {
                        CloseHandle(mutex);
                    }
                    return Err(format!("CreateFileMapping failed ({})", std::io::Error::last_os_error()));
                }
                let view = MapViewOfFile(mapping, FILE_MAP_WRITE, 0, 0, HEAP_SIZE);
                if view.Value.is_null() {
                    CloseHandle(mapping);
                    if !mutex.is_null() {
                        CloseHandle(mutex);
                    }
                    return Err(format!("MapViewOfFile failed ({})", std::io::Error::last_os_error()));
                }
                Ok(Self { mapping, mutex, view })
            }
        }

        fn u32_at(&self, off: usize) -> &AtomicU32 {
            debug_assert!(off + 4 <= HEAP_SIZE && off.is_multiple_of(4));
            unsafe { &*((self.view.Value as *mut u8).add(off) as *const AtomicU32) }
        }

        fn i32_at(&self, off: usize) -> &AtomicI32 {
            unsafe { &*((self.view.Value as *mut u8).add(off) as *const AtomicI32) }
        }

        fn store_f32(&self, off: usize, v: f32) {
            self.u32_at(off).store(v.to_bits(), Ordering::SeqCst);
        }
    }

    impl Drop for Shm {
        fn drop(&mut self) {
            unsafe {
                UnmapViewOfFile(self.view);
                CloseHandle(self.mapping);
                if !self.mutex.is_null() {
                    CloseHandle(self.mutex);
                }
            }
        }
    }

    #[derive(Default)]
    pub struct FreeTrack {
        shm: Option<Shm>,
        game_id: i32,
        game_name: String,
        registry_done_for: Option<String>,
        npclient: Option<std::path::PathBuf>,
        error: Option<String>,
        last_open_attempt: Option<std::time::Instant>,
    }

    impl FreeTrack {
        pub fn disable(&mut self) -> String {
            self.shm = None;
            self.registry_done_for = None;
            "Off".into()
        }

        pub fn send(&mut self, mapped: &[f64; 6], raw: &[f64; 6], cfg: &FreetrackConfig) -> String {
            if self.shm.is_none() {
                let retry = self.last_open_attempt.map(|t| t.elapsed().as_secs() >= 2).unwrap_or(true);
                if !retry {
                    return self.error.clone().unwrap_or_default();
                }
                self.last_open_attempt = Some(std::time::Instant::now());
                match Shm::open() {
                    Ok(s) => {
                        s.u32_at(OFF_DATA_ID).store(1, Ordering::SeqCst);
                        s.i32_at(OFF_CAM_W).store(100, Ordering::SeqCst);
                        s.i32_at(OFF_CAM_H).store(250, Ordering::SeqCst);
                        self.shm = Some(s);
                        self.game_id = -1;
                        self.error = None;
                    }
                    Err(e) => {
                        self.error = Some(format!("FreeTrack shared memory error: {e}"));
                        return self.error.clone().unwrap_or_default();
                    }
                }
            }
            if cfg.set_registry && self.registry_done_for.as_deref() != Some(cfg.npclient_dir.as_str()) {
                self.npclient = find_npclient_dir(&cfg.npclient_dir);
                if let Some(dir) = &self.npclient {
                    if let Err(e) = set_registry(dir) {
                        log::warn!("freetrack: registry: {e}");
                    }
                }
                self.registry_done_for = Some(cfg.npclient_dir.clone());
            }
            let shm = self.shm.as_ref().expect("shm open");
            let v = ft_values(mapped);
            let r = ft_values(raw);
            for i in 0..6 {
                shm.store_f32(OFF_YAW + 4 * i, v[i]);
                shm.store_f32(OFF_RAW + 4 * i, r[i]);
            }
            // opentrack writes RawPitch with the opposite sign of Pitch; keep that quirk.
            shm.store_f32(OFF_RAW + 4, -r[1]);

            let id = shm.i32_at(OFF_GAME_ID).load(Ordering::SeqCst);
            if id != self.game_id {
                let (name, table) = game_table(id).unwrap_or_else(|| (if id == 0 { String::new() } else { format!("unknown game {id}") }, [0; 8]));
                shm.u32_at(OFF_TABLE).store(u32::from_le_bytes([table[0], table[1], table[2], table[3]]), Ordering::SeqCst);
                shm.u32_at(OFF_TABLE + 4).store(u32::from_le_bytes([table[4], table[5], table[6], table[7]]), Ordering::SeqCst);
                shm.i32_at(OFF_GAME_ID2).store(id, Ordering::SeqCst);
                shm.u32_at(OFF_DATA_ID).store(0, Ordering::SeqCst);
                self.game_id = id;
                self.game_name = name;
            } else {
                shm.u32_at(OFF_DATA_ID).fetch_add(1, Ordering::SeqCst);
            }

            let dll = match (&self.npclient, cfg.set_registry) {
                (Some(d), true) => format!("NPClient: {}", d.display()),
                (None, true) => "NPClient64.dll not found: install opentrack (it doesn't need to run) or set the folder".into(),
                (_, false) => "registry not managed".into(),
            };
            if self.game_name.is_empty() {
                format!("Active, waiting for a game · {dll}")
            } else {
                format!("Connected: {} · {dll}", self.game_name)
            }
        }
    }

    fn set_registry(dir: &std::path::Path) -> Result<(), String> {
        use winreg::enums::HKEY_CURRENT_USER;
        use winreg::RegKey;
        let mut path = dir.display().to_string();
        if !path.ends_with('\\') {
            path.push('\\');
        }
        let hkcu = RegKey::predef(HKEY_CURRENT_USER);
        for key in ["Software\\NaturalPoint\\NATURALPOINT\\NPClient Location", "Software\\Freetrack\\FreeTrackClient"] {
            let (k, _) = hkcu.create_subkey(key).map_err(|e| format!("{key}: {e}"))?;
            k.set_value("Path", &path).map_err(|e| format!("{key}: {e}"))?;
        }
        log::info!("freetrack: registry points to {path}");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dcs_table_matches_opentrack_parsing() {
        // 88;DCS;FreeTrack20;V170;V;EmBeES;1006;0058F688FC9B0556868F00
        let (name, t) = game_table(1006).expect("DCS in table");
        assert_eq!(name, "DCS");
        assert_eq!(t, [0x9B, 0xFC, 0x88, 0xF6, 0x8F, 0x86, 0x56, 0x05]);
        assert!(game_table(0).is_none());
        assert!(game_table(987654).is_none());
    }

    #[test]
    fn ft_conventions() {
        let v = ft_values(&[90.0, 10.0, -5.0, 1.0, 2.0, 3.0]);
        assert!((v[0] + std::f32::consts::FRAC_PI_2).abs() < 1e-6);
        assert!(v[1] < 0.0 && v[2] < 0.0);
        assert_eq!(&v[3..], &[10.0, 20.0, 30.0]);
    }
}
