use crate::pairing_suite::{Affine, Curve, Scalar};
use ag_cuda_proxy::{ActiveWorkspace, DeviceParam, KernelConfig};
use ag_cuda_workspace_macro::auto_workspace;
use ag_types::GpuName;
use ark_ff::Field;
use ark_std::Zero;
use rustacuda::error::CudaResult;

use crate::{GLOBAL, LOCAL};

#[auto_workspace]
pub fn radix_ec_fft(
    workspace: &ActiveWorkspace, input: &mut Vec<Curve>, omegas: &[Scalar],
) -> CudaResult<()> {
    radix_fft(workspace, input, omegas, Affine::name(), false)
}

#[auto_workspace]
pub fn radix_scalar_fft(
    workspace: &ActiveWorkspace, input: &mut Vec<Scalar>, omegas: &[Scalar],
) -> CudaResult<()> {
    radix_fft(workspace, input, omegas, Scalar::name(), true)
}

fn radix_fft<T: Zero + Clone + Copy>(
    workspace: &ActiveWorkspace, input: &mut Vec<T>, omegas: &[Scalar],
    name: String, twiddle_list: bool,
) -> CudaResult<()> {
    const MAX_LOG2_RADIX: u32 = 8;

    let n = input.len();
    let log_n = n.ilog2();
    assert_eq!(n, 1 << log_n);

    let mut output = vec![T::zero(); n];

    let max_deg = std::cmp::min(MAX_LOG2_RADIX, log_n);

    // let twiddle = omegas[0].pow([(n >> max_deg) as u64]);

    let twiddle = {
        let p = omegas[(log_n - max_deg) as usize];
        if !twiddle_list {
            vec![p]
        } else {
            (0..(1 << (max_deg - 1))).map(|i| p.pow([i])).collect()
        }
    };

    let mut input_gpu = DeviceParam::new(input)?;
    let mut output_gpu = DeviceParam::new(&mut output)?;

    let stream = workspace.stream()?;
    input_gpu.to_device(&stream)?;

    let mut kernel = workspace.create_kernel()?;

    // Specifies log2 of `p`, (http://www.bealto.com/gpu-fft_group-1.html)
    let mut log_p = 0u32;
    // Each iteration performs a FFT round
    while log_p < log_n {
        // 1=>radix2, 2=>radix4, 3=>radix8, ...
        let deg = std::cmp::min(max_deg, log_n - log_p);

        let n = 1u32 << log_n;

        let virtual_local_work_size = 1 << (deg - 1);

        // The algorithm may require a small local_network_size. However, too
        // small local_network_size will undermine the performance. So we
        // allocate a larger local_network_size, but translate the global
        // parameter before execution.
        let physical_local_work_size = if virtual_local_work_size >= 32 {
            virtual_local_work_size
        } else if n <= 64 {
            virtual_local_work_size
        } else {
            32
        };
        let global_work_size = n / 2 / physical_local_work_size;

        let config = KernelConfig {
            global_work_size: global_work_size as usize,
            local_work_size: physical_local_work_size as usize,
            shared_mem: std::mem::size_of::<T>()
                * 2
                * physical_local_work_size as usize,
        };

        let kernel_name = format!("{}_radix_fft", name);

        kernel = kernel
            .func(&kernel_name)?
            .dev_arg(&input_gpu)?
            .dev_arg(&output_gpu)?
            .in_ref_slice(&twiddle[..])?
            .in_ref_slice(&omegas[..])?
            .empty()?
            .val(n)?
            .val(log_p)?
            .val(deg)?
            .val(virtual_local_work_size)?
            .val(max_deg)?
            .launch(config)?
            .complete()?;

        log_p += deg;
        DeviceParam::swap_device_pointer(&mut input_gpu, &mut output_gpu);
    }

    input_gpu.to_host(&stream)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::pairing_suite::Scalar;
    use ark_ff::{FftField, Field};
    use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
    use ark_std::{rand::thread_rng, Zero};

    use super::*;
    use crate::test_tools::random_input;

    #[test]
    fn test_ec_fft() {
        let mut rng = thread_rng();

        for degree in 4..8 {
            let n = 1 << degree;

            println!("Testing FFTg for {} elements...", n);

            let mut omegas = vec![Scalar::zero(); 32];
            omegas[0] = Scalar::get_root_of_unity(n as u64).unwrap();
            for i in 1..32 {
                omegas[i] = omegas[i - 1].square();
            }

            let mut v1_coeffs = random_input(n, &mut rng);
            let mut v2_coeffs = v1_coeffs.clone();

            // Evaluate with GPU
            radix_ec_fft_mt(&mut v1_coeffs, &omegas[..]).unwrap();

            // Evaluate with CPU
            let fft_domain =
                Radix2EvaluationDomain::<Scalar>::new(v2_coeffs.len()).unwrap();

            v2_coeffs = fft_domain.fft(&v2_coeffs);

            if v1_coeffs != v2_coeffs {
                panic!("wrong answer");
            }
        }
    }

    #[test]
    fn test_scalar_fft() {
        let mut rng = thread_rng();

        for degree in 4..8 {
            let n = 1 << degree;

            println!("Testing FFT for {} elements...", n);

            let mut omegas = vec![Scalar::zero(); 32];
            omegas[0] = Scalar::get_root_of_unity(n as u64).unwrap();
            for i in 1..32 {
                omegas[i] = omegas[i - 1].square();
            }

            let mut v1_coeffs = random_input(n, &mut rng);
            let mut v2_coeffs = v1_coeffs.clone();

            // Evaluate with GPU
            radix_scalar_fft_mt(&mut v1_coeffs, &omegas[..]).unwrap();

            // Evaluate with CPU
            let fft_domain =
                Radix2EvaluationDomain::<Scalar>::new(v2_coeffs.len()).unwrap();

            v2_coeffs = fft_domain.fft(&v2_coeffs);
            // dbg!(&v1_coeffs);
            // dbg!(&v2_coeffs);

            if v1_coeffs != v2_coeffs {
                panic!("wrong answer");
            }
        }
    }
}
