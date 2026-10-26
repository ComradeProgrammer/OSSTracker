# MyTracker (Rust)


一个开箱即用的头瞄软件 适配dcs和opentracker

本tracker接受笔记本前置摄像头，不会对视频帧进行裁剪，在捕捉人脸动作之前先识别人脸,大角度扭头不会闪退，rust写的也不会内存泄漏。

使用方法：去release页面下载windows的压缩包 解压缩之后 里面就一个x86 exe，开箱即用点击就行。**必须在启动dcs之前启动本软件**

English below

<img width="1440" height="1080" alt="v2-e7b8568184825dba85e019ba47f871b7_r" src="https://github.com/user-attachments/assets/2cbe0359-11c4-4c70-9ad3-401cc975478f" />

## 前言

Deepseek圆了我的飞行梦<br/>
一个VR的价格是，2千5百块 (quest3s)<br/>
一个头瞄的价格是，1千5百块<br/>
而一个摄像头 30块；DCS的苏25T 不要钱<br/>
我想玩DCS 但是没有头瞄也没有VR<br/>
于是跟万能的deepseek说 我该怎么办?<br/>
大肥鱼让我把摄像头打开,然后她去huggingface拉了俩模型，一个识别图片哪部分有脸，一个锁我脸. 她又用rust写了个纯视觉方案<br/>
3个小时之后，火车从埃因霍温开回了阿姆斯特丹<br/>
我回到家就开上了苏25T<br/>
**玩的蛮开心 但是对CS这行的信心玩没了<br/>**
**早晚得失业回国送外卖吧<br/>**

*代码真是deepseek写的 contributor有claude是我让他后来搞定一下release pipeline*


# MyTracker (Rust)

A webcam head tracker for DCS World and other flight sims, for Windows and macOS.
It outputs to **opentrack (UDP)** and, on Windows, directly to **FreeTrack/TrackIR** (what DCS reads).

## Features

- **Never quits on its own.** No face, no camera, an unplugged camera, a driver error or a bad
  frame all show as a status message, and the app keeps retrying. Camera and tracker threads
  restart themselves after an internal error.
- **Works close to the camera** (camera on the monitor bezel). Face crops read black outside the
  frame, so a face larger than the frame or partly out of it still tracks. The detector retries
  zoomed out for very close faces, and pose is solved with a perspective-correct PnP using your
  camera's field of view.
- **Large head turns.** The face region is tracked from the previous frame's 478 landmarks
  (MediaPipe Face Mesh V2), so tracking doesn't depend on a frontal face detector. If the face is
  lost, the view holds for 2 s, eases back to centre, and tracking resumes when the face is found
  again, preferring the face nearest to where it was lost.
- **Smooth motion.** Pose comes from about 45 rigid landmarks (forehead, nose bridge, eye corners,
  cheekbones), never the mouth or eyelids. An adaptive One-Euro filter keeps the view steady when
  still and responsive when moving, and a separate 120 Hz output stage interpolates between camera
  frames, so a 30 fps camera doesn't produce stair steps.
- **Full camera field of view.** The capture mode is chosen to keep the sensor's full field of view
  at the highest frame rate (or pick a mode yourself). Frames are letterboxed, never cropped.

## Download

Prebuilt packages are on the [Releases page](https://github.com/ComradeProgrammer/OSSTracker/releases):

- **Windows x64:** unzip and run `MyTracker.exe`.
- **macOS (Apple Silicon):** unzip and move `MyTracker.app` to Applications. The app isn't notarized,
  so the first time, right-click it and choose *Open*, or run
  `xattr -dr com.apple.quarantine MyTracker.app`.

Releases are manual: on GitHub open **Actions → release → Run workflow** and enter a tag such
as `v0.2.0` (or run `gh workflow run release.yml -f version=v0.2.0`). It builds and tests both
platforms, creates the tag and publishes the release. **Actions → build** is a manual test build.
Nothing runs automatically on push.

## Build from source

### Windows (main target)

1. Install [Rust](https://rustup.rs) (the default `x86_64-pc-windows-msvc` toolchain) and the
   **Visual Studio Build Tools** with "Desktop development with C++".
2. In this folder:
   ```
   cargo build --release
   ```
3. Run `target\release\mytracker.exe`. It is a single self-contained exe: ONNX Runtime and the
   face models are built in.

The first build downloads a prebuilt ONNX Runtime library, so it needs internet access.

### macOS

`cargo build --release`, then run `target/release/mytracker`. The first launch asks for camera
permission; allow it for the terminal or app that starts MyTracker.

## Set up DCS

**Option A: direct, no opentrack running (Windows)**
1. Install [opentrack](https://github.com/opentrack/opentrack/releases). It only provides
   `NPClient64.dll`; it does not need to run.
2. In MyTracker, under *Output*, enable **FreeTrack / TrackIR**. It finds opentrack's `modules`
   folder and points the TrackIR registry key at it. If it can't find the folder, enter it yourself.
3. Start DCS. The status line shows `Connected: DCS`.
4. Tune the view in *Response curves*. The defaults turn about 40° of head yaw into 180° of view.

**Option B: through opentrack**
1. In opentrack set **Input = UDP over network** (port 4242) and **Output = freetrack 2.0 Enhanced**.
2. In MyTracker keep **opentrack UDP** enabled and **Send with my response curves** off, then
   tune the curves in opentrack instead.

If an axis moves the wrong way in your setup, tick **Invert** for that axis.

## Use

- **Recenter**: look straight at your screen and press `F12` (configurable), the button, or a bound
  joystick button. The first pose found after start becomes the centre.
- **Pause**: `Shift+F12` freezes the view, like TrackIR's pause.
- **Joystick/HOTAS**: under *Hotkeys & buttons*, click **Bind** and press the button.
- **Camera FOV**: set it to your webcam's horizontal field of view (usually 60–80°). This matters
  most when you sit close.
- *Diagnostics* shows camera fps and per-stage timings (detection, landmarks, pose).

Settings save automatically to `%APPDATA%\MyTracker\config\config.json` on Windows
(`~/Library/Application Support/MyTracker/` on macOS). A `mytracker.json` next to the exe takes
priority, for portable use. The log is `mytracker.log` in the same folder.

## Development

```
cargo test --release                                            # unit tests
cargo run --release --example replay  -- path/to/face_photo.jpg # stress test: roll, very close, out of frame, lost/back
cargo run --release --example e2e_udp -- path/to/face_photo.jpg # threaded pipeline -> UDP: rate, smoothness, lag, lost-face behaviour
```

Layout: `camera.rs` (capture thread, mode choice), `models.rs` (ONNX Runtime: BlazeFace +
Face Mesh V2), `tracker.rs` (ROI tracking, PnP, filtering), `pose.rs` (Umeyama + robust LM PnP,
Euler conventions), `filter.rs` (One-Euro, curves), `output/` (120 Hz output, UDP, FreeTrack),
`hotkeys.rs`, `app.rs` (egui UI).

## Licences and credits

- Face models: Google MediaPipe BlazeFace (short range) and Face Mesh V2, Apache-2.0. ONNX
  conversions from `unity/inference-engine-blaze-face` and `fernandotonon/QtMeshEditor-facemesh-onnx`
  on Hugging Face. The canonical face model is from MediaPipe, Apache-2.0.
- `models/freetrack_games.csv` and the FreeTrack memory layout follow opentrack (ISC).
