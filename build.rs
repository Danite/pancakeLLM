use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=setup_env.sh");

    let output = Command::new("sh")
        .arg("-c")
        .arg("source ./setup_env.sh && env")
        .output()
        .expect("Failed to execute setup_env.sh");

    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        if line.starts_with("LIBTORCH=")
            || line.starts_with("LD_LIBRARY_PATH=")
            || line.starts_with("DYLD_LIBRARY_PATH=")
            || line.starts_with("LIBTORCH_USE_PYTORCH=")
            || line.starts_with("LIBTORCH_BYPASS_VERSION_CHECK=")
        {
            println!("cargo:rustc-env={}", line);
        }
    }
}
