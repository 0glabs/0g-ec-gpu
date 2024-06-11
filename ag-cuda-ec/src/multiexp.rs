use crate::pairing_suite::{Affine, Curve, Scalar};
use ag_cuda_proxy::{ActiveWorkspace, DeviceData, KernelConfig};
use ag_cuda_workspace_macro::auto_workspace;
use ag_types::{GpuName, GpuRepr, PrimeFieldRepr};
use ark_std::Zero;
use rustacuda::error::CudaResult;

use crate::{GLOBAL, LOCAL};

#[auto_workspace]
pub fn upload_multiexp_bases(
    workspace: &ActiveWorkspace, bases: &[Affine],
) -> CudaResult<DeviceData> {
    let bases_gpu_repr: Vec<_> =
        bases.iter().map(GpuRepr::to_gpu_repr).collect();
    let stream = workspace.stream()?;
    DeviceData::upload(&bases_gpu_repr, &stream)
}

#[auto_workspace]
pub fn multiple_multiexp(
    workspace: &ActiveWorkspace, bases_gpu: &DeviceData,
    exponents: &[<Scalar as PrimeFieldRepr>::Repr], num_chunks: usize,
    window_size: usize, neg_is_cheap: bool,
) -> CudaResult<Vec<Curve>> {
    let num_windows = (256 + window_size - 1) / window_size;
    let num_bases =
        bases_gpu.size() / std::mem::size_of::<<Affine as GpuRepr>::Repr>();
    let num_lines = num_bases / exponents.len();
    let work_units = num_windows * num_chunks * num_lines;
    let input_len = exponents.len();

    let bucket_len = if neg_is_cheap {
        1 << (window_size - 1)
    } else {
        (1 << window_size) - 1
    };

    let mut output = vec![Curve::zero(); num_chunks * num_lines];

    let buckets = DeviceData::uninitialized(
        work_units * bucket_len * std::mem::size_of::<Curve>(),
    )?;

    let kernel = workspace.create_kernel()?;

    let local_work_size = num_windows; // most efficient: 32 - 128
    let global_work_size = work_units / local_work_size;

    let config = KernelConfig {
        global_work_size,
        local_work_size,
        shared_mem: 0,
    };

    let kernel_name = format!("{}_multiexp", Affine::name());

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
    use crate::pairing_suite::{Curve, Scalar};
    use ag_types::PrimeFieldRepr;
    use ark_ec::VariableBaseMSM;
    use ark_std::rand::thread_rng;

    use super::*;
    use crate::test_tools::random_input;

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
            let gpu_output: Vec<_> = multiple_multiexp_mt(
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

            let gpu_output: Vec<_> = multiple_multiexp_mt(
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
