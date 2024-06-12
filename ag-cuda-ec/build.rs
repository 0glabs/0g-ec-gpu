fn main() {
    use ag_build::{generate, SourceBuilder};

    let source = if cfg!(feature = "bn254") {
        SourceBuilder::new()
            .add_fft::<ark_bn254::Fr>()
            .add_ec_fft::<ark_bn254::G1Affine>()
            .add_multiexp::<ark_bn254::G1Affine>()
            .add_multiexp::<ark_bn254::G2Affine>()
    } else if cfg!(feature = "bls12-381") {
        SourceBuilder::new()
            .add_fft::<ark_bls12_381::Fr>()
            .add_ec_fft::<ark_bls12_381::G1Affine>()
            .add_multiexp::<ark_bls12_381::G1Affine>()
            .add_multiexp::<ark_bls12_381::G2Affine>()
    } else {
        panic!("One of the pairing suite must be specified");
    };

    generate(&source);
}
