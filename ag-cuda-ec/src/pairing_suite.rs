#[cfg(feature = "bn254")]
pub use ark_bn254::{
    Bn254 as PE, Fr as Scalar, G1Affine as Affine, G1Projective as Curve,
};

#[cfg(feature = "bls12-381")]
pub use ark_bls12_381::{
    Bls12_381 as PE, Fr as Scalar, G1Affine as Affine, G1Projective as Curve,
};

#[cfg(not(any(feature = "bn254", feature = "bls12-381")))]
compile_error!("One of the pairing suite must be specified");
