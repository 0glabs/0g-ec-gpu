#![allow(unused, unused_mut)]

#[cfg(feature = "bn254")]
use ark_bn254::{Fr, G1Affine, G2Affine};
#[cfg(feature = "bls12-381")]
use ark_bn254::{Fr, G1Affine, G2Affine};

#[cfg(not(any(feature = "bls12-381", feature = "bn254")))]
compile_error!("One of the pairing suite must be specified");

fn main() {
    use ag_build::{generate, SourceBuilder};
    let mut source = SourceBuilder::new();

    if cfg!(feature = "fr-fft") {
        source = source.add_fft::<Fr>();
    }
    if cfg!(feature = "g1-fft") {
        source = source.add_ec_fft::<G1Affine>();
    }
    if cfg!(feature = "g2-fft") {
        source = source.add_ec_fft::<G2Affine>();
    }
    if cfg!(feature = "g1-msm") {
        source = source.add_multiexp::<G1Affine>();
    }
    if cfg!(feature = "g2-msm") {
        source = source.add_multiexp::<G2Affine>();
    }

    generate(&source);
}
