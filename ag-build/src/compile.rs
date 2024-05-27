//! Convience function to generate a kernel/source based on a source builder.
///
/// When the `cuda` feature is enabled it will compile a CUDA fatbin. The
/// path to the file is stored in the `_EC_GPU_CUDA_KERNEL_FATBIN`
/// environment variable, that will automatically be used by the
/// `ec-gpu-gen` functionality that needs a kernel.
///
///
/// When the `opencl` feature is enabled it will generate the source code
/// for OpenCL. The path to the source file is stored in the
/// `_EC_GPU_OPENCL_KERNEL_SOURCE` environment variable, that will
/// automatically be used by the `ec-gpu-gen` functionality that needs a
/// kernel. OpenCL compiles the source at run time).
pub use super::source::SourceBuilder;

pub use std::{env, fs, path::PathBuf};

fn in_build_script() -> bool { std::env::var("OUT_DIR").is_ok() }

fn working_dir() -> String {
    if let Ok(dir) = env::var("ARK_GPU_BUILD_DIR") {
        dir
    } else if let Ok(dir) = env::var("OUT_DIR") {
        dir
    } else {
        tempfile::tempdir()
            .unwrap()
            .path()
            .to_str()
            .unwrap()
            .to_owned()
    }
}

macro_rules! bprintln {
    ($($arg:tt)*) => {
        if in_build_script() {
            println!($($arg)*);
        }
    };
}

#[cfg(feature = "cuda")]
pub fn generate_cuda(source_builder: &SourceBuilder) -> PathBuf {
    use sha2::{Digest, Sha256};

    // This is a hack when no properly compiled kernel is needed. That's the
    // case when the documentation is built on docs.rs and when Clippy is
    // run. We can use arbitrary bytes as input then.
    if std::env::var("DOCS_RS").is_ok() || cfg!(feature = "cargo-clippy") {
        bprintln!("cargo:rustc-env=_EC_GPU_CUDA_KERNEL_FATBIN=../build.rs");
        return PathBuf::from("../build.rs");
    }

    let kernel_source = source_builder.build_32_bit_limbs();
    let out_dir = working_dir();

    // Make it possible to override the default options. Though the source and
    // output file is always set automatically.
    let mut nvcc = match env::var("EC_GPU_CUDA_NVCC_ARGS") {
        Ok(args) => execute::command(format!("nvcc {}", args)),
        Err(_) => {
            let mut command = std::process::Command::new("nvcc");
            command
                .arg("--optimize=6")
                // Compile with as many threads as CPUs are available.
                .arg("--threads=0")
                .arg("--fatbin")
                .arg("--gpu-architecture=sm_86")
                .arg("--generate-code=arch=compute_86,code=sm_86")
                .arg("--generate-code=arch=compute_80,code=sm_80")
                .arg("--generate-code=arch=compute_75,code=sm_75");
            command
        }
    };

    // Hash the source and the compile flags. Use that as the filename, so that
    // the kernel is only rebuilt if any of them change.
    let mut hasher = Sha256::new();
    hasher.update(kernel_source.as_bytes());
    hasher.update(&format!("{:?}", &nvcc));
    let kernel_digest = hex::encode(hasher.finalize());

    let source_path: PathBuf = [&out_dir, &format!("{}.cu", &kernel_digest)]
        .iter()
        .collect();
    let fatbin_path: PathBuf =
        [&out_dir, &format!("{}.fatbin", &kernel_digest)]
            .iter()
            .collect();

    fs::write(&source_path, &kernel_source).unwrap_or_else(|_| {
        panic!(
            "Cannot write kernel source at {}.",
            source_path.to_str().unwrap()
        )
    });

    // Only compile if the output doesn't exist yet.
    if !fatbin_path.as_path().exists() {
        let status = nvcc
            .arg("--output-file")
            .arg(&fatbin_path)
            .arg(&source_path)
            .status()
            .expect("Cannot run nvcc. Install the NVIDIA toolkit or disable the `cuda` feature.");

        if !status.success() {
            panic!(
                "nvcc failed. See the kernel source at {}",
                source_path.to_str().unwrap()
            );
        }
    }

    // The idea to put the path to the farbin into a compile-time env variable
    // is from https://github.com/LutzCle/fast-interconnects-demo/blob/b80ea8e04825167f486ab8ac1b5d67cf7dd51d2c/rust-demo/build.rs
    bprintln!(
        "cargo:rustc-env=_EC_GPU_CUDA_KERNEL_FATBIN={}",
        fatbin_path.to_str().unwrap()
    );
    if !in_build_script() {
        env::set_var(
            "_EC_GPU_CUDA_KERNEL_FATBIN",
            fatbin_path.to_str().unwrap(),
        );
    }

    fatbin_path
}

#[cfg(feature = "opencl")]
pub fn generate_opencl(source_builder: &SourceBuilder) -> PathBuf {
    let kernel_source = source_builder.build_64_bit_limbs();
    let out_dir = working_dir();

    // Generating the kernel source is cheap, hence use a fixed name and
    // override it on every build.
    let source_path: PathBuf = [&out_dir, "kernel.cl"].iter().collect();

    fs::write(&source_path, &kernel_source).unwrap_or_else(|_| {
        panic!(
            "Cannot write kernel source at {}.",
            source_path.to_str().unwrap()
        )
    });

    // For OpenCL we only need the kernel source, it is compiled at runtime.
    bprintln!(
        "cargo:rustc-env=_EC_GPU_OPENCL_KERNEL_SOURCE={}",
        source_path.to_str().unwrap()
    );
    if !in_build_script() {
        env::set_var(
            "_EC_GPU_OPENCL_KERNEL_SOURCE",
            source_path.to_str().unwrap(),
        );
    }

    source_path
}
