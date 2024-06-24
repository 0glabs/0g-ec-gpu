#[allow(unused)]
use crate::pairing_suite::{
    Affine, Affine2, Curve, G1Config, G2Config, Scalar,
};
use ag_cuda_proxy::{ActiveWorkspace, DeviceParam, Kernel, KernelConfig};
use ag_cuda_workspace_macro::auto_workspace;
use ark_ff::Field;
use ark_std::{One, Zero};
use rustacuda::error::CudaResult;

use crate::{GLOBAL, LOCAL};

mod toggle {
    #![allow(unused)]

    use crate::pairing_suite::{Affine, Affine2, G1Config, G2Config, Scalar};
    use ag_types::GpuName;
    use ark_std::Zero;

    pub trait GpuFftField: Zero + Clone + Copy {
        const TWIDDLE_LIST: bool;
        fn name() -> String;
    }

    #[cfg(feature = "fr-fft")]
    impl GpuFftField for Scalar {
        const TWIDDLE_LIST: bool = true;

        fn name() -> String { <Scalar as GpuName>::name() }
    }

    #[cfg(feature = "g1-fft")]
    impl GpuFftField for ark_ec::short_weierstrass::Projective<G1Config> {
        const TWIDDLE_LIST: bool = false;

        fn name() -> String { <Affine as GpuName>::name() }
    }

    #[cfg(feature = "g2-fft")]
    impl GpuFftField for ark_ec::short_weierstrass::Projective<G2Config> {
        const TWIDDLE_LIST: bool = false;

        fn name() -> String { <Affine2 as GpuName>::name() }
    }
}

use toggle::GpuFftField;

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

fn mul_by_field<'a, T: GpuFftField>(
    kernel: Kernel<'a>, device: &DeviceParam<'_, T>, n: usize, g: Scalar,
    c: Scalar,
) -> CudaResult<Kernel<'a>> {
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

    if gc_flag == 0 {
        return Ok(kernel);
    }

    let local_work_size = std::cmp::min(64, n);
    let global_work_size = n / local_work_size;
    let config = KernelConfig {
        global_work_size,
        local_work_size,
        shared_mem: 0,
    };
    kernel
        .func(&format!("{}_mul_by_field", T::name()))?
        .dev_arg(device)?
        .val(n)?
        .val(g)?
        .val(c)?
        .val(gc_flag)?
        .launch(config)?
        .complete()
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

    if !after {
        kernel = mul_by_field(kernel, &input_gpu, n, g, c)?;
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

    if after {
        kernel = mul_by_field(kernel, &input_gpu, n, g, c)?;
    }

    input_gpu.to_host(&stream)?;
    std::mem::drop(kernel);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{pairing_suite::Scalar, test_tools::random_input};
    use ark_ff::{FftField, Field};
    use ark_poly::{
        domain::DomainCoeff, EvaluationDomain, Radix2EvaluationDomain,
    };
    use ark_std::{
        rand::{
            distributions::{Distribution, Standard},
            thread_rng,
        },
        Zero,
    };

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

    trait FftTestTask {
        fn gpu<T: GpuFftField>(input: &mut [T]);
        fn cpu<T: DomainCoeff<Scalar>>(input: &[T]) -> Vec<T>;
        fn test<T: GpuFftField + DomainCoeff<Scalar>>()
        where Standard: Distribution<T> {
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

                Self::gpu::<T>(&mut v1_coeffs);
                let v2_coeffs = Self::cpu::<T>(&mut v2_coeffs);

                if v1_coeffs != v2_coeffs {
                    panic!("wrong answer");
                }
            }
        }

        fn test_all() {
            #[cfg(feature = "fr-fft")]
            Self::test::<Scalar>();
            #[cfg(feature = "g1-fft")]
            Self::test::<crate::pairing_suite::Curve>();
            #[cfg(feature = "g2-fft")]
            Self::test::<crate::pairing_suite::Curve2>();
        }
    }

    type Domain = Radix2EvaluationDomain<Scalar>;

    #[test]
    fn test_fft() {
        struct Fft;
        Fft::test_all();

        impl FftTestTask for Fft {
            fn gpu<T: GpuFftField>(v: &mut [T]) {
                radix_fft_mt::<T>(v, &omegas(v.len()), None).unwrap()
            }

            fn cpu<T: DomainCoeff<Scalar>>(v: &[T]) -> Vec<T> {
                Domain::new(v.len()).unwrap().fft(v)
            }
        }
    }

    #[test]
    fn test_ifft() {
        struct Ifft;
        Ifft::test_all();

        impl FftTestTask for Ifft {
            fn gpu<T: GpuFftField>(v: &mut [T]) {
                let size_inv = Scalar::from(v.len() as u64).inverse().unwrap();
                radix_ifft_mt::<T>(v, &omegas_inv(v.len()), None, size_inv)
                    .unwrap()
            }

            fn cpu<T: DomainCoeff<Scalar>>(v: &[T]) -> Vec<T> {
                Domain::new(v.len()).unwrap().ifft(v)
            }
        }
    }

    #[test]
    fn test_fft_coset() {
        struct FftCoset;
        FftCoset::test_all();

        impl FftTestTask for FftCoset {
            fn gpu<T: GpuFftField>(v: &mut [T]) {
                let coset = Scalar::from(4);
                radix_fft_mt::<T>(v, &omegas(v.len()), Some(coset)).unwrap()
            }

            fn cpu<T: DomainCoeff<Scalar>>(v: &[T]) -> Vec<T> {
                Domain::new(v.len())
                    .unwrap()
                    .get_coset(4.into())
                    .unwrap()
                    .fft(v)
            }
        }
    }

    #[test]
    fn test_ifft_coset() {
        struct IfftCoset;
        IfftCoset::test_all();

        impl FftTestTask for IfftCoset {
            fn gpu<T: GpuFftField>(v: &mut [T]) {
                let coset_inv = Scalar::from(4).inverse().unwrap();
                let size_inv = Scalar::from(v.len() as u64).inverse().unwrap();
                radix_ifft_mt::<T>(
                    v,
                    &omegas_inv(v.len()),
                    Some(coset_inv),
                    size_inv,
                )
                .unwrap()
            }

            fn cpu<T: DomainCoeff<Scalar>>(v: &[T]) -> Vec<T> {
                Domain::new(v.len())
                    .unwrap()
                    .get_coset(4.into())
                    .unwrap()
                    .ifft(v)
            }
        }
    }
}
