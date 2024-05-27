use rand::{thread_rng, Rng};

use super::{
    program::call_kernel,
    types::{GpuScalar, Scalar},
};

use ag_types::PrimeFieldRepr;
use ark_ff::{Field, UniformRand};

#[test]
fn test_add() {
    let mut rng = thread_rng();
    for _ in 0..10 {
        let a = Scalar::rand(&mut rng);
        let b = Scalar::rand(&mut rng);
        let c = a + b;

        assert_eq!(
            call_kernel("test_add", &[GpuScalar(a), GpuScalar(b)], &[]),
            c
        );
    }
}

#[test]
fn test_sub() {
    let mut rng = thread_rng();
    for _ in 0..10 {
        let a = Scalar::rand(&mut rng);
        let b = Scalar::rand(&mut rng);
        let c = a - b;
        assert_eq!(
            call_kernel("test_sub", &[GpuScalar(a), GpuScalar(b)], &[]),
            c
        );
    }
}

#[test]
fn test_mul() {
    let mut rng = thread_rng();
    for _ in 0..10 {
        let a = Scalar::rand(&mut rng);
        let b = Scalar::rand(&mut rng);
        let c = a * b;

        assert_eq!(
            call_kernel("test_mul", &[GpuScalar(a), GpuScalar(b)], &[]),
            c
        );
    }
}

#[test]
fn test_pow() {
    let mut rng = thread_rng();
    for _ in 0..10 {
        let a = Scalar::rand(&mut rng);
        let b = rng.gen::<u32>();
        let c = a.pow(&[b as u64]);
        assert_eq!(call_kernel("test_pow", &[GpuScalar(a)], &[b]), c);
    }
}

#[test]
fn test_sqr() {
    let mut rng = thread_rng();
    for _ in 0..10 {
        let a = Scalar::rand(&mut rng);
        let b = a.square();

        assert_eq!(call_kernel("test_sqr", &[GpuScalar(a)], &[]), b);
    }
}

#[test]
fn test_double() {
    let mut rng = thread_rng();
    for _ in 0..10 {
        let a = Scalar::rand(&mut rng);
        let b = a.double();

        assert_eq!(call_kernel("test_double", &[GpuScalar(a)], &[]), b);
    }
}

#[test]
fn test_unmont() {
    let mut rng = thread_rng();
    for _ in 0..10 {
        let a = Scalar::rand(&mut rng);
        let b: Scalar = unsafe { std::mem::transmute(a.to_bigint()) };
        assert_eq!(call_kernel("test_unmont", &[GpuScalar(a)], &[]), b);
    }
}

#[test]
fn test_mont() {
    let mut rng = thread_rng();
    for _ in 0..10 {
        let a_repr = Scalar::rand(&mut rng).to_bigint();
        let a: Scalar = unsafe { std::mem::transmute(a_repr) };
        let b = Scalar::from_bigint(a_repr).unwrap();
        assert_eq!(call_kernel("test_mont", &[GpuScalar(a)], &[]), b);
    }
}
