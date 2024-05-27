use ag_cuda_ec::{
    ec_fft::*,
    init_global_workspace, init_local_workspace,
    pairing_suite::{Affine, Scalar},
    test_tools::random_input,
};
use ark_ec::AffineRepr;
use ark_ff::{FftField, Field, UniformRand};
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use ark_std::{rand::thread_rng, Zero};
use rayon::iter::ParallelIterator;
use std::time::Instant;

fn main() {
    bench_ec_fft_sequential();
    // bench_ec_fft_parallel();
}

#[allow(unused)]
fn bench_ec_fft_sequential() {
    let mut rng = thread_rng();
    init_global_workspace();

    for degree in 0..12usize {
        let n: usize = 1 << degree;

        println!("Testing FFTg for {} elements...", n);

        let mut omegas = vec![Scalar::zero(); 32];
        omegas[0] = Scalar::get_root_of_unity(n as u64).unwrap();
        for i in 1..32 {
            omegas[i] = omegas[i - 1].square();
        }

        let mut v1_coeffs = random_input(n, &mut rng);
        let mut v2_coeffs = v1_coeffs.clone();

        // Evaluate with GPU
        let now = Instant::now();
        radix_ec_fft_st(&mut v1_coeffs, &omegas[..]).unwrap();
        let gpu_dur = now.elapsed().as_millis();
        println!("GPU took {}ms.", gpu_dur);

        // Evaluate with CPU
        let fft_domain =
            Radix2EvaluationDomain::<Scalar>::new(v2_coeffs.len()).unwrap();
        let now = Instant::now();
        fft_domain.fft_in_place(&mut v2_coeffs);
        let cpu_dur = now.elapsed().as_millis();
        println!("CPU took {}ms.", cpu_dur);

        // Evaluate with CPU
        println!("Speedup: x{}", cpu_dur as f32 / gpu_dur as f32);

        assert_eq!(v1_coeffs, v2_coeffs);

        println!("============================");
    }
}

#[allow(unused)]
fn bench_ec_fft_parallel() {
    use ark_std::One;
    use rayon::iter::IntoParallelIterator;

    (0..32u64).into_par_iter().for_each(|_| {
        let mut rng = thread_rng();
        let degree = 6;

        let n = 1 << degree;

        let mut omegas = vec![Scalar::zero(); 32];
        omegas[0] = Scalar::get_root_of_unity(n).unwrap();
        for i in 1..32 {
            omegas[i] = omegas[i - 1].square();
        }

        init_local_workspace();

        let mut v1_coeffs = (0..n)
            .map(|_| Affine::rand(&mut rng).into_group())
            .collect::<Vec<_>>();
        let mut v2_coeffs = v1_coeffs.clone();

        let now = Instant::now();
        // Evaluate with GPU
        radix_ec_fft_mt(&mut v1_coeffs, &omegas[..]).unwrap();
        let gpu_dur = now.elapsed().as_millis();
        println!("GPU took {}ms.", gpu_dur);

        let mut omegas2 = vec![Scalar::zero(); 32];
        omegas2[0] = Scalar::one() / Scalar::get_root_of_unity(n).unwrap();
        for i in 1..32 {
            omegas2[i] = omegas2[i - 1].square();
        }

        let now = Instant::now();
        // Evaluate with GPU
        radix_ec_fft_mt(&mut v1_coeffs, &omegas2[..]).unwrap();
        let gpu_dur = now.elapsed().as_millis();
        println!("GPU took {}ms.", gpu_dur);

        // // Evaluate with CPU
        // let fft_domain =
        //     Radix2EvaluationDomain::<Scalar>::new(v2_coeffs.len()).unwrap();
        // fft_domain.fft_in_place(&mut v2_coeffs);

        let sn: Scalar = Scalar::from(n as u64);
        v2_coeffs.iter_mut().for_each(|x| *x *= sn);

        assert_eq!(v1_coeffs, v2_coeffs);
    });
}
