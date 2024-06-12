use super::*;

use ark_ec::{
    models::short_weierstrass::Affine, short_weierstrass::SWCurveConfig,
};

impl<T: MontConfig<N>, const N: usize> PrimeFieldRepr
    for ark_ff::Fp<MontBackend<T, N>, N>
where Self: PrimeField
{
    type Repr = BigInt<N>;

    fn to_bigint(&self) -> Self::Repr { MontConfig::into_bigint(*self) }

    fn from_bigint(repr: Self::Repr) -> Option<Self> {
        MontConfig::from_bigint(repr)
    }
}

fn u64_to_u32(limbs: &[u64]) -> Vec<u32> {
    let split_u64 =
        |limb: &u64| [(limb & u32::MAX as u64) as u32, (limb >> 32) as u32];
    limbs.iter().flat_map(split_u64).collect()
}

impl<T: MontConfig<N>, const N: usize> GpuField
    for ark_ff::Fp<MontBackend<T, N>, N>
{
    fn one() -> Vec<u32> { u64_to_u32(&Self::R.0[..]) }

    fn r2() -> Vec<u32> { u64_to_u32(&Self::R2.0[..]) }

    fn modulus() -> Vec<u32> { u64_to_u32(&Self::MODULUS.0[..]) }
}

impl<P: Fp2Config> GpuField for ark_ff::Fp2<P>
where P::Fp: GpuField
{
    fn one() -> Vec<u32> { <P::Fp as GpuField>::one() }

    fn r2() -> Vec<u32> { <P::Fp as GpuField>::r2() }

    fn modulus() -> Vec<u32> { <P::Fp as GpuField>::modulus() }

    fn sub_field_name() -> Option<String> { Some(<P::Fp as GpuName>::name()) }
}

impl<P: SWCurveConfig> GpuRepr for Affine<P> {
    type Repr = [P::BaseField; 2];

    fn to_gpu_repr(&self) -> Self::Repr {
        if self.is_zero() {
            [P::BaseField::zero(); 2]
        } else {
            [self.x, self.y]
        }
    }
}

impl<P: SWCurveConfig> GpuCurveAffine for Affine<P>
where
    <Affine<P> as ark_ec::AffineRepr>::ScalarField: GpuField + PrimeFieldRepr,
    <Affine<P> as ark_ec::AffineRepr>::BaseField: GpuField,
{
    type Base = <Affine<P> as ark_ec::AffineRepr>::BaseField;
    type Curve = <Affine<P> as ark_ec::AffineRepr>::Group;
    type Scalar = <Affine<P> as ark_ec::AffineRepr>::ScalarField;

    fn is_identity(&self) -> bool { Affine::is_zero(self) }
}

impl<T: GpuCurveAffine> GpuCurveName for T {
    type Affine = Self;
    type Base = <Self as GpuCurveAffine>::Base;
    type Scalar = <Self as GpuCurveAffine>::Scalar;
}

impl<T: Any> GpuName for T {
    fn name() -> String { name!() }
}
