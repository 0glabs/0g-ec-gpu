use ag_cuda_ec::{
    init_global_workspace,
    multiexp::*,
    pairing_suite::{Affine, Scalar},
    test_tools::random_input_by_cycle,
};
use ag_types::PrimeFieldRepr;
use ark_std::rand::thread_rng;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::time::Instant;

fn main() { bench_large_amt(); }

fn bench_large_amt() {
    let mut rng = thread_rng();
    init_global_workspace();

    const LOG_N: usize = 10;
    const LENGTH: usize = 1 << (2 * LOG_N + 1);

    let bases =
        random_input_by_cycle::<Affine, _>(LENGTH * LOG_N, 97, &mut rng);
    let exps = random_input_by_cycle::<Scalar, _>(LENGTH, 73, &mut rng);

    let now = Instant::now();
    let exp_reprs: Vec<_> = exps.par_iter().map(|x| x.to_bigint()).collect();
    let unmont_dur = now.elapsed().as_millis();
    println!(
        "unmont_dur CPU took {}ms for {} scalars.",
        unmont_dur,
        exp_reprs.len()
    );

    let bases_gpu = upload_multiexp_bases_st(&bases).unwrap();

    // Evaluate with GPU
    for group_degree in 7..=11 {
        let num_groups = 1 << group_degree;
        for window_size in 4..=9 {
            let now = Instant::now();
            let _acc_gpu: Vec<_> = multiple_multiexp_st(
                &bases_gpu,
                &exp_reprs,
                num_groups,
                window_size,
                true,
            )
            .unwrap();
            let gpu_dur = now.elapsed().as_millis();
            println!(
                "GPU took {}ms with: group size {}, window size {}.",
                gpu_dur, num_groups, window_size
            );
            println!("----------------------------");
        }
        println!("============================");
    }
}
