use super::types::{Base, G1Affine, GpuScalar, Scalar};
use crate::compile::*;
use lazy_static::lazy_static;
use rust_gpu_tools::{Device, GPUError, Program};
use std::sync::Mutex;

#[cfg(feature = "cuda")]
use rust_gpu_tools::cuda;
#[cfg(feature = "opencl")]
use rust_gpu_tools::opencl;

fn test_source() -> SourceBuilder {
    SourceBuilder::new().add_test::<G1Affine, Base>()
}

/// The `run` call needs to return a result, use this struct as placeholder.
#[derive(Debug)]
struct NoError;
impl From<GPUError> for NoError {
    fn from(_error: GPUError) -> Self { Self }
}

#[cfg(feature = "cuda")]
lazy_static! {
    pub static ref CUDA_PROGRAM: Mutex<Program> = {
        use std::ffi::CString;

        let source = test_source();
        let fatbin_path = generate_cuda(&source);

        let device =
            *Device::all().first().expect("Cannot get a default device.");
        let cuda_device = device.cuda_device().unwrap();
        let fatbin_path_cstring = CString::new(
            fatbin_path.to_str().expect("path is not valid UTF-8."),
        )
        .expect("path contains NULL byte.");
        let program = cuda::Program::from_binary(
            cuda_device,
            fatbin_path_cstring.as_c_str(),
        )
        .unwrap();
        Mutex::new(Program::Cuda(program))
    };
}

#[cfg(feature = "opencl")]
lazy_static! {
    pub static ref OPENCL_PROGRAM: Mutex<(Program, Program)> = {
        let device =
            *Device::all().first().expect("Cannot get a default device");
        let opencl_device = device.opencl_device().unwrap();
        let source_32 = test_source().build_32_bit_limbs();
        let program_32 =
            opencl::Program::from_opencl(opencl_device, &source_32).unwrap();
        let source_64 = test_source().build_64_bit_limbs();
        let program_64 =
            opencl::Program::from_opencl(opencl_device, &source_64).unwrap();
        Mutex::new((Program::Opencl(program_32), Program::Opencl(program_64)))
    };
}

use rust_gpu_tools::program_closures;

pub fn call_kernel(name: &str, scalars: &[GpuScalar], uints: &[u32]) -> Scalar {
    let closures =
        program_closures!(|program, _args| -> Result<Scalar, NoError> {
            let mut cpu_buffer = vec![GpuScalar::default()];
            let buffer = program.create_buffer_from_slice(&cpu_buffer).unwrap();

            let mut kernel = program.create_kernel(name, 1, 64).unwrap();
            for scalar in scalars {
                kernel = kernel.arg(scalar);
            }
            for uint in uints {
                kernel = kernel.arg(uint);
            }
            kernel.arg(&buffer).run().unwrap();

            program.read_into_buffer(&buffer, &mut cpu_buffer).unwrap();
            Ok(cpu_buffer[0].0)
        });

    // For CUDA we only test 32-bit limbs.
    #[cfg(all(feature = "cuda", not(feature = "opencl")))]
    return CUDA_PROGRAM.lock().unwrap().run(closures, ()).unwrap();

    // For OpenCL we test for 32 and 64-bi limbs.
    #[cfg(all(feature = "opencl", not(feature = "cuda")))]
    {
        let result_32 =
            OPENCL_PROGRAM.lock().unwrap().0.run(closures, ()).unwrap();
        let result_64 =
            OPENCL_PROGRAM.lock().unwrap().1.run(closures, ()).unwrap();
        assert_eq!(
            result_32, result_64,
            "Results for 32-bit and 64-bit limbs must be the same."
        );
        result_32
    }

    // When both features are enabled, check if the results are the same
    #[cfg(all(feature = "cuda", feature = "opencl"))]
    {
        let cuda_result =
            CUDA_PROGRAM.lock().unwrap().run(closures, ()).unwrap();
        let opencl_32_result =
            OPENCL_PROGRAM.lock().unwrap().0.run(closures, ()).unwrap();
        let opencl_64_result =
            OPENCL_PROGRAM.lock().unwrap().1.run(closures, ()).unwrap();
        assert_eq!(
            opencl_32_result, opencl_64_result,
            "Results for 32-bit and 64-bit limbs on OpenCL must be the same."
        );
        assert_eq!(
            cuda_result, opencl_32_result,
            "Results for CUDA and OpenCL must be the same."
        );
        cuda_result
    }
}
