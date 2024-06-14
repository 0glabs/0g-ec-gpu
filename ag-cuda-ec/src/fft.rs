#[allow(unused)]
use crate::pairing_suite::{
    Affine, Affine2, Curve, G1Config, G2Config, Scalar,
};
use ag_cuda_proxy::{ActiveWorkspace, DeviceParam, KernelConfig};
use ag_cuda_workspace_macro::auto_workspace;
use ark_ff::Field;
use ark_std::{One, Zero};
use rustacuda::error::CudaResult;

use crate::{GLOBAL, LOCAL};

pub trait GpuFftField: Zero + Clone + Copy {
    const TWIDDLE_LIST: bool;
    fn name() -> String;
}
#[cfg(feature = "fr-fft")]
impl GpuFftField for Scalar {
    const TWIDDLE_LIST: bool = true;

    fn name() -> String { <Scalar as ag_types::GpuName>::name() }
}
#[cfg(feature = "g1-fft")]
impl GpuFftField for ark_ec::short_weierstrass::Projective<G1Config> {
    const TWIDDLE_LIST: bool = false;

    fn name() -> String { <Affine as ag_types::GpuName>::name() }
}
#[cfg(feature = "g2-fft")]
impl GpuFftField for ark_ec::short_weierstrass::Projective<G2Config> {
    const TWIDDLE_LIST: bool = false;

    fn name() -> String { <Affine2 as ag_types::GpuName>::name() }
}

#[auto_workspace]
pub fn radix_fft<T: GpuFftField>(
    workspace: &ActiveWorkspace, input: &mut [T], omegas: &[Scalar],
    offset: Option<Scalar>,
) -> CudaResult<()> {
    let offset = offset.unwrap_or(Scalar::one());
    radix_fft_inner(workspace, input, omegas, offset, Scalar::one(), false)
}

#[auto_workspace]
pub fn radix_ifft<T: GpuFftField>(
    workspace: &ActiveWorkspace, input: &mut [T], omegas: &[Scalar],
    offset_inv: Option<Scalar>, size_inv: Scalar,
) -> CudaResult<()> {
    let offset = offset_inv.unwrap_or(Scalar::one());
    radix_fft_inner(workspace, input, omegas, offset, size_inv, true)
}

fn radix_fft_inner<T: GpuFftField>(
    workspace: &ActiveWorkspace, input: &mut [T], omegas: &[Scalar], g: Scalar,
    c: Scalar, after: bool,
) -> CudaResult<()> {
    const MAX_LOG2_RADIX: u32 = 8;

    if g.is_zero() || c.is_zero() {
        input.iter_mut().for_each(|x| *x = T::zero());
        return Ok(());
    }

    let n = input.len();
    let log_n = n.ilog2();
    assert_eq!(n, 1 << log_n);

    let mut output = vec![T::zero(); n];

    let max_deg = std::cmp::min(MAX_LOG2_RADIX, log_n);

    // let twiddle = omegas[0].pow([(n >> max_deg) as u64]);

    let twiddle = {
        let p = omegas[(log_n - max_deg) as usize];
        if !T::TWIDDLE_LIST {
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

    let gc_flag = {
        let mut ans = 0;
        if !g.is_one() {
            ans |= 0x2;
        }
        if !c.is_one() {
            ans |= 0x1;
        }
        ans
    };

    if gc_flag != 0 && !after {
        let local_work_size = std::cmp::min(64, n);
        let global_work_size = n / local_work_size;
        let config = KernelConfig {
            global_work_size,
            local_work_size,
            shared_mem: 0,
        };
        kernel = kernel
            .func(&format!("{}_mul_by_field", T::name()))?
            .dev_arg(&input_gpu)?
            .val(n)?
            .val(g)?
            .val(c)?
            .val(gc_flag)?
            .launch(config)?
            .complete()?;
    }

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
        let physical_local_work_size =
            if virtual_local_work_size >= 32 || n <= 64 {
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

        let kernel_name = format!("{}_radix_fft", T::name());

        kernel = kernel
            .func(&kernel_name)?
            .dev_arg(&input_gpu)?
            .dev_arg(&output_gpu)?
            .in_ref_slice(&twiddle)?
            .in_ref_slice(omegas)?
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

    if gc_flag != 0 && after {
        let local_work_size = std::cmp::min(64, n);
        let global_work_size = n / local_work_size;
        let config = KernelConfig {
            global_work_size,
            local_work_size,
            shared_mem: 0,
        };
        kernel
            .func(&format!("{}_mul_by_field", T::name()))?
            .dev_arg(&input_gpu)?
            .val(n)?
            .val(g)?
            .val(c)?
            .val(gc_flag)?
            .launch(config)?
            .complete()?;
    }

    input_gpu.to_host(&stream)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(unused)]

    use core::prelude::v1;
    use std::any::Any;

    use crate::pairing_suite::{Curve, Curve2, Scalar};
    use ark_ff::{FftField, Field};
    use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
    use ark_std::{
        rand::{
            distributions::{Distribution, Standard},
            thread_rng,
        },
        Zero,
    };
    use once_cell::sync::Lazy;

    use super::*;
    use crate::test_tools::random_input;

    fn omegas(n: usize) -> [Scalar; 32] {
        let mut omegas = [Scalar::zero(); 32];
        omegas[0] = Scalar::get_root_of_unity(n as u64).unwrap();
        for i in 1..32 {
            omegas[i] = omegas[i - 1].square();
        }
        omegas
    }

    fn omegas_inv(n: usize) -> [Scalar; 32] {
        let mut omegas = [Scalar::zero(); 32];
        omegas[0] = Scalar::get_root_of_unity(n as u64)
            .unwrap()
            .inverse()
            .unwrap();
        for i in 1..32 {
            omegas[i] = omegas[i - 1].square();
        }
        omegas
    }

    fn test_fft_inner<T: GpuFftField + Any + Eq>(
        gpu: impl Fn(&mut [T]), cpu: impl Fn(&[T]) -> Vec<T>,
    ) where Standard: Distribution<T> {
        let mut rng = thread_rng();
        for degree in 4..8 {
            let n = 1 << degree;
            println!(
                "Testing FFT for {} elements type {}",
                n,
                std::any::type_name::<T>()
            );

            let mut v1_coeffs = random_input(n, &mut rng);
            let mut v2_coeffs = v1_coeffs.clone();

            gpu(&mut v1_coeffs);
            let v2_coeffs = cpu(&mut v2_coeffs);

            if v1_coeffs != v2_coeffs {
                panic!("wrong answer");
            }
        }
    }

    type Domain = Radix2EvaluationDomain<Scalar>;

    #[cfg(feature = "fr-fft")]
    #[test]
    fn test_scalar_fft() {
        test_fft_inner(
            |v| radix_fft_mt::<Scalar>(v, &omegas(v.len()), None).unwrap(),
            |v| Domain::new(v.len()).unwrap().fft(&v),
        );
    }

    #[cfg(feature = "fr-fft")]
    #[test]
    fn test_scalar_ifft() {
        let domain = |n| Domain::new(n).unwrap();
        test_fft_inner(
            |v| {
                radix_ifft_mt::<Scalar>(
                    v,
                    &omegas_inv(v.len()),
                    None,
                    domain(v.len()).size_inv,
                )
                .unwrap()
            },
            |v| domain(v.len()).ifft(&v),
        );
    }

    #[cfg(feature = "fr-fft")]
    #[test]
    fn test_scalar_fft_coset() {
        let domain =
            |n| Domain::new(n).unwrap().get_coset(Scalar::from(4)).unwrap();
        test_fft_inner(
            |v| {
                radix_fft_mt::<Scalar>(
                    v,
                    &omegas(v.len()),
                    Some(domain(v.len()).coset_offset()),
                )
                .unwrap()
            },
            |v| domain(v.len()).fft(&v),
        );
    }

    #[cfg(feature = "fr-fft")]
    #[test]
    fn test_scalar_ifft_coset() {
        let domain =
            |n| Domain::new(n).unwrap().get_coset(Scalar::from(4)).unwrap();
        test_fft_inner(
            |v| {
                let d = domain(v.len());
                radix_ifft_mt::<Scalar>(
                    v,
                    &omegas_inv(v.len()),
                    Some(d.coset_offset_inv()),
                    d.size_inv,
                )
                .unwrap()
            },
            |v| domain(v.len()).ifft(&v),
        );
    }

    #[cfg(feature = "g1-fft")]
    #[test]
    fn test_ec_fft() {
        test_fft_inner(
            |v| radix_fft_mt::<Curve>(v, &omegas(v.len()), None).unwrap(),
            |v| Domain::new(v.len()).unwrap().fft(&v),
        );
    }

    #[cfg(feature = "g1-fft")]
    #[test]
    fn test_ec_ifft_coset() {
        let domain =
            |n| Domain::new(n).unwrap().get_coset(Scalar::from(4)).unwrap();
        test_fft_inner(
            |v| {
                let d = domain(v.len());
                radix_ifft_mt::<Curve>(
                    v,
                    &omegas_inv(v.len()),
                    Some(d.coset_offset_inv()),
                    d.size_inv,
                )
                .unwrap()
            },
            |v| domain(v.len()).ifft(&v),
        );
    }

    #[cfg(feature = "g2-fft")]
    #[test]
    fn test_ec_g2_fft() {
        test_fft_inner(
            |v| radix_fft_mt::<Curve2>(v, &omegas(v.len()), None).unwrap(),
            |v| Domain::new(v.len()).unwrap().fft(&v),
        );
    }

    #[cfg(feature = "g2-fft")]
    #[test]
    fn test_ec_g2_ifft_coset() {
        let domain =
            |n| Domain::new(n).unwrap().get_coset(Scalar::from(4)).unwrap();
        test_fft_inner(
            |v| {
                let d = domain(v.len());
                radix_ifft_mt::<Curve2>(
                    v,
                    &omegas_inv(v.len()),
                    Some(d.coset_offset_inv()),
                    d.size_inv,
                )
                .unwrap()
            },
            |v| domain(v.len()).ifft(&v),
        );
    }
}
