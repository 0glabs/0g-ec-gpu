use std::marker::PhantomData;

use crate::pairing_suite::Scalar;
use ag_cuda_proxy::{ActiveWorkspace, DeviceData, KernelConfig};
use ag_cuda_workspace_macro::auto_workspace;
use ag_types::{GpuCurveAffine, GpuRepr, PrimeFieldRepr};
use ark_ec::AffineRepr;
use ark_ff::{BigInteger, Zero};
use rustacuda::error::CudaResult;

use crate::{GLOBAL, LOCAL};

pub struct DeviceAffineSlice<T: GpuCurveAffine>(DeviceData, PhantomData<T>);

fn filter_out_zero<T: Copy + AffineRepr, U: Copy + BigInteger>(
    bases: &[T], exps: &[U], num_chunks: usize, default_base: T,
) -> (Vec<T>, Vec<U>) {
    let zipped: Vec<_> = bases
        .iter()
        .zip(exps.iter())
        .filter(|(x, y)| !x.is_zero() && !y.is_zero())
        .collect();

    let length = zipped.len();

    let mut actual_bases: Vec<_> =
        zipped.iter().map(|(base, _)| **base).collect();
    let mut actual_exps: Vec<_> = zipped.iter().map(|(_, exp)| **exp).collect();
    if length % num_chunks != 0 {
        let padding_len = num_chunks - length % num_chunks;

        actual_bases.extend(std::iter::repeat(default_base).take(padding_len));

        actual_exps.extend(std::iter::repeat(U::from(0u32)).take(padding_len));
    };
    (actual_bases, actual_exps)
}

#[auto_workspace]
pub fn multiexp<T: GpuCurveAffine<Scalar = Scalar>>(
    workspace: &ActiveWorkspace, bases: &[T],
    exponents: &[<Scalar as PrimeFieldRepr>::Repr], num_chunks: usize,
    window_size: usize, neg_is_cheap: bool,
) -> CudaResult<Vec<<T as AffineRepr>::Group>> {
    let (bases, exponents) =
        filter_out_zero(bases, exponents, num_chunks, AffineRepr::generator());

    let device = upload_multiexp_bases(workspace, &bases[..])?;
    multiple_multiexp::<T>(
        workspace,
        &device,
        &exponents[..],
        num_chunks,
        window_size,
        neg_is_cheap,
    )
}

#[auto_workspace]
pub fn upload_multiexp_bases<T: GpuCurveAffine<Scalar = Scalar>>(
    workspace: &ActiveWorkspace, bases: &[T],
) -> CudaResult<DeviceAffineSlice<T>> {
    let bases_gpu_repr: Vec<_> =
        bases.iter().map(GpuRepr::to_gpu_repr).collect();
    let stream = workspace.stream()?;
    Ok(DeviceAffineSlice(
        DeviceData::upload(&bases_gpu_repr, &stream)?,
        PhantomData,
    ))
}

#[auto_workspace]
pub fn multiple_multiexp<T: GpuCurveAffine<Scalar = Scalar>>(
    workspace: &ActiveWorkspace, bases_gpu: &DeviceAffineSlice<T>,
    exponents: &[<Scalar as PrimeFieldRepr>::Repr], num_chunks: usize,
    window_size: usize, neg_is_cheap: bool,
) -> CudaResult<Vec<<T as AffineRepr>::Group>> {
    let bases_gpu = &bases_gpu.0;
    let num_windows = (256 + window_size - 1) / window_size;
    let num_bases =
        bases_gpu.size() / std::mem::size_of::<<T as GpuRepr>::Repr>();
    let num_lines = num_bases / exponents.len();
    let work_units = num_windows * num_chunks * num_lines;
    let input_len = exponents.len();

    let bucket_len = if neg_is_cheap {
        1 << (window_size - 1)
    } else {
        (1 << window_size) - 1
    };

    let mut output =
        vec![<T as AffineRepr>::Group::zero(); num_chunks * num_lines];

    let buckets = DeviceData::uninitialized(
        work_units
            * bucket_len
            * std::mem::size_of::<<T as AffineRepr>::Group>(),
    )?;

    let kernel = workspace.create_kernel()?;

    let local_work_size = num_windows; // most efficient: 32 - 128
    let global_work_size = work_units / local_work_size;

    let config = KernelConfig {
        global_work_size,
        local_work_size,
        shared_mem: 0,
    };

    let kernel_name = format!("{}_multiexp", T::name());

    kernel
        .func(&kernel_name)?
        .dev_data(bases_gpu)?
        .out_slice(&mut output)?
        .in_ref_slice(&exponents)?
        .dev_data(&buckets)?
        .val(input_len as u32)?
        .val(num_lines as u32)?
        .val(num_chunks as u32)?
        .val(num_windows as u32)?
        .val(window_size as u32)?
        .val(neg_is_cheap)?
        .launch(config)?
        .complete()?;

    Ok(output)
}

#[cfg(test)]
mod tests {
    use crate::pairing_suite::{Affine, Affine2, Curve, Curve2, Scalar};
    use ag_types::PrimeFieldRepr;
    use ark_ec::VariableBaseMSM;
    use ark_std::rand::thread_rng;

    use super::*;
    use crate::test_tools::random_input;

    #[test]
    fn test_multiexp() {
        let mut rng = thread_rng();

        let mut bases = random_input(1773, &mut rng);
        let mut exponents = random_input::<Scalar, _>(2579, &mut rng);
        exponents[0] = Scalar::zero();
        bases[1] = Affine::zero();
        let exponents_repr: Vec<_> =
            exponents.iter().map(|x| x.to_bigint()).collect();

        let output =
            multiexp_mt(&bases, &exponents_repr[..], 32, 6, false).unwrap();
        let res: Curve = output.iter().cloned().sum();

        let expected: Curve =
            VariableBaseMSM::msm_bigint(&bases, &exponents_repr);
        assert_eq!(res, expected);
    }

    #[test]
    fn test_multiexp_g2() {
        let mut rng = thread_rng();

        let mut bases = random_input(1773, &mut rng);
        let mut exponents = random_input::<Scalar, _>(2579, &mut rng);
        exponents[0] = Scalar::zero();
        bases[1] = Affine2::zero();
        let exponents_repr: Vec<_> =
            exponents.iter().map(|x| x.to_bigint()).collect();

        let output =
            multiexp_mt::<Affine2>(&bases, &exponents_repr[..], 32, 6, false)
                .unwrap();
        let res: Curve2 = output.iter().cloned().sum();

        let expected: Curve2 =
            VariableBaseMSM::msm_bigint(&bases, &exponents_repr);
        assert_eq!(res, expected);
    }

    #[test]
    fn test_multiexp_batch() {
        let mut rng = thread_rng();

        const CHUNK_SIZE: usize = 64;
        const CHUNK_NUM: usize = 32;
        const LINES: usize = 2;
        const INPUT_LEN: usize = CHUNK_SIZE * CHUNK_NUM;

        let bases = random_input(INPUT_LEN * LINES, &mut rng);
        let exponents = random_input::<Scalar, _>(INPUT_LEN, &mut rng);

        let bases_gpu = upload_multiexp_bases_mt(&bases).unwrap();
        let exponents_repr: Vec<_> =
            exponents.iter().map(|x| x.to_bigint()).collect();

        let cpu_output: Vec<_> = bases
            .chunks(CHUNK_SIZE)
            .zip(exponents_repr.chunks(CHUNK_SIZE).cycle())
            .map(|(bs, er)| Curve::msm_bigint(bs, er))
            .collect();

        for window_size in 1..=9 {
            let gpu_output: Vec<_> = multiple_multiexp_mt::<Affine>(
                &bases_gpu,
                &exponents_repr,
                CHUNK_NUM,
                window_size,
                true,
            )
            .unwrap();

            assert_eq!(gpu_output.len(), cpu_output.len());

            if gpu_output != cpu_output {
                panic!("Result inconsistent");
            }

            let gpu_output: Vec<_> = multiple_multiexp_mt::<Affine>(
                &bases_gpu,
                &exponents_repr,
                CHUNK_NUM,
                window_size,
                false,
            )
            .unwrap();

            if gpu_output != cpu_output {
                panic!("Result inconsistent");
            }
        }
    }
}
