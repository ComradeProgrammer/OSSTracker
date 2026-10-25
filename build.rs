// The prebuilt ONNX Runtime static library for macOS calls `__isPlatformVersionAtLeast`
// (from `@available` checks), which lives in clang's compiler-rt and isn't linked by rustc.
fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        if let Ok(out) = std::process::Command::new("clang").arg("--print-resource-dir").output() {
            let dir = String::from_utf8_lossy(&out.stdout).trim().to_string();
            let lib = std::path::Path::new(&dir).join("lib/darwin");
            if lib.join("libclang_rt.osx.a").exists() {
                // A plain link arg goes after all rlibs, so the symbols resolve for ort-sys.
                println!("cargo:rustc-link-arg={}", lib.join("libclang_rt.osx.a").display());
            }
        }
    }
    println!("cargo:rerun-if-changed=build.rs");
}
