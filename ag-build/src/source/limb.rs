use ag_types::GpuField;
use std::mem;

#[derive(Clone, Copy)]
pub enum Limb32Or64 {
    Limb32,
    Limb64,
}

/// Trait to implement limbs of different underlying bit sizes.
pub trait Limb: Sized + Clone + Copy {
    /// The underlying size of the limb, e.g. `u32`
    type LimbType: Clone + std::fmt::Display;
    /// Returns the value representing zero.
    fn zero() -> Self;
    /// Returns a new limb.
    fn new(val: Self::LimbType) -> Self;
    /// Returns the raw value of the limb.
    fn value(&self) -> Self::LimbType;
    /// Returns the bit size of the limb.
    fn bits() -> usize { mem::size_of::<Self::LimbType>() * 8 }
    /// Returns a tuple with the strings that PTX is using to describe the type
    /// and the register.
    fn ptx_info() -> (&'static str, &'static str);
    /// Returns the type that OpenCL is using to represent the limb.
    fn opencl_type() -> &'static str;
    /// Returns the limbs that represent the multiplicative identity of the
    /// given field.
    fn one_limbs<F: GpuField>() -> Vec<Self>;
    /// Returns the field modulus in non-Montgomery form as a vector of
    /// `Self::LimbType` (least significant limb first).
    fn modulus_limbs<F: GpuField>() -> Vec<Self>;
    /// Calculate the `INV` parameter of Montgomery reduction algorithm for
    /// 32/64bit limbs
    /// * `a` - Is the first limb of modulus.
    fn calc_inv(a: Self) -> Self;
    /// Returns the limbs that represent `R ^ 2 mod P`.
    fn calculate_r2<F: GpuField>() -> Vec<Self>;
}

/// A 32-bit limb.
#[derive(Clone, Copy)]
pub struct Limb32(u32);
impl Limb for Limb32 {
    type LimbType = u32;

    fn zero() -> Self { Self(0) }

    fn new(val: Self::LimbType) -> Self { Self(val) }

    fn value(&self) -> Self::LimbType { self.0 }

    fn ptx_info() -> (&'static str, &'static str) { ("u32", "r") }

    fn opencl_type() -> &'static str { "uint" }

    fn one_limbs<F: GpuField>() -> Vec<Self> {
        F::one().into_iter().map(Self::new).collect()
    }

    fn modulus_limbs<F: GpuField>() -> Vec<Self> {
        F::modulus().into_iter().map(Self::new).collect()
    }

    fn calc_inv(a: Self) -> Self {
        let mut inv = 1u32;
        for _ in 0..31 {
            inv = inv.wrapping_mul(inv);
            inv = inv.wrapping_mul(a.value());
        }
        Self(inv.wrapping_neg())
    }

    fn calculate_r2<F: GpuField>() -> Vec<Self> {
        F::r2().into_iter().map(Self::new).collect()
    }
}

/// A 64-bit limb.
#[derive(Clone, Copy)]
pub struct Limb64(u64);
impl Limb for Limb64 {
    type LimbType = u64;

    fn zero() -> Self { Self(0) }

    fn new(val: Self::LimbType) -> Self { Self(val) }

    fn value(&self) -> Self::LimbType { self.0 }

    fn ptx_info() -> (&'static str, &'static str) { ("u64", "l") }

    fn opencl_type() -> &'static str { "ulong" }

    fn one_limbs<F: GpuField>() -> Vec<Self> {
        F::one()
            .chunks(2)
            .map(|chunk| {
                Self::new(((chunk[1] as u64) << 32) + (chunk[0] as u64))
            })
            .collect()
    }

    fn modulus_limbs<F: GpuField>() -> Vec<Self> {
        F::modulus()
            .chunks(2)
            .map(|chunk| {
                Self::new(((chunk[1] as u64) << 32) + (chunk[0] as u64))
            })
            .collect()
    }

    fn calc_inv(a: Self) -> Self {
        let mut inv = 1u64;
        for _ in 0..63 {
            inv = inv.wrapping_mul(inv);
            inv = inv.wrapping_mul(a.value());
        }
        Self(inv.wrapping_neg())
    }

    fn calculate_r2<F: GpuField>() -> Vec<Self> {
        F::r2()
            .chunks(2)
            .map(|chunk| {
                Self::new(((chunk[1] as u64) << 32) + (chunk[0] as u64))
            })
            .collect()
    }
}
