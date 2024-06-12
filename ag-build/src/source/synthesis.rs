use ag_types::{GpuCurveName, GpuField, GpuName};
use std::{
    fmt,
    hash::{Hash, Hasher},
    marker::PhantomData,
};

use super::{limb::Limb32Or64, template::*};

/// This trait is used to uniquely identify items by some identifier (`name`)
/// and to return the GPU source code they produce.
pub trait NameAndSource {
    /// The name to identify the item.
    fn name(&self) -> String;
    /// The GPU source code that is generated.
    fn source(&self, limb: Limb32Or64) -> String;
}

impl PartialEq for dyn NameAndSource {
    fn eq(&self, other: &Self) -> bool { self.name() == other.name() }
}

impl Eq for dyn NameAndSource {}

impl PartialOrd for dyn NameAndSource {
    #[allow(clippy::non_canonical_partial_ord_impl)]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.name().partial_cmp(&other.name())
    }
}

impl Ord for dyn NameAndSource {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.name().cmp(&other.name())
    }
}

impl Hash for dyn NameAndSource {
    fn hash<H: Hasher>(&self, state: &mut H) { self.name().hash(state) }
}

/// Prints the name by default, the source code of the 32-bit limb in the
/// alternate mode via `{:#?}`.
impl fmt::Debug for dyn NameAndSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            f.debug_map()
                .entries(vec![
                    ("name", self.name()),
                    ("source", self.source(Limb32Or64::Limb32)),
                ])
                .finish()
        } else {
            write!(f, "{:?}", self.name())
        }
    }
}

/// A field that might also be an extension field.
///
/// When the field is an extension field, we also add its sub-field to the list
/// of fields. This enum is used to indicate that it's a sub-field that has a
/// corresponding extension field. This way we can make sure that when the
/// source is generated, that also the source for the sub-field is generated,
/// while not having duplicated field definitions.
// Storing the sub-field as a string is a bit of a hack around Rust's type
// system. If we would store the generic type, then the enum would need to be
// generic over two fields, even in the case when no extension field is used.
// This would make the API harder to use.
#[derive(Debug)]
pub enum Field<F: GpuField> {
    /// A field, might be an extension field.
    Field(PhantomData<F>),
    /// A sub-field with the given name that has a corresponding extension
    /// field.
    SubField(String),
}

impl<F: GpuField> Field<F> {
    /// Create a new field for the given generic type.
    pub fn new() -> Self {
        // By default it's added as a field. If it's an extension field, then
        // the `add_field()` function will create a copy of it, as
        // `SubField` variant.
        Self::Field(PhantomData)
    }
}

impl<F: GpuField> Default for Field<F> {
    fn default() -> Self { Self::new() }
}

impl<F: GpuField> NameAndSource for Field<F> {
    fn name(&self) -> String {
        match self {
            Self::Field(_) => F::name(),
            Self::SubField(name) => name.to_string(),
        }
    }

    fn source(&self, limb: Limb32Or64) -> String {
        match self {
            Self::Field(_) => {
                // If it's an extension field.
                if let Some(sub_field_name) = F::sub_field_name() {
                    String::from(FIELD2_SRC)
                        .replace("FIELD2", &F::name())
                        .replace("FIELD", &sub_field_name)
                } else {
                    field_source::<F>(limb).replace("FIELD", &F::name())
                }
            }
            Self::SubField(sub_field_name) => {
                // The `GpuField` implementation of the extension field contains
                // the constants of the sub-field. Hence we can
                // just forward the `F`. It's important that those
                // functions do *not* use the name of the field, else we might
                // generate the sub-field named like the
                // extension field.
                field_source::<F>(limb).replace("FIELD", sub_field_name)
            }
        }
    }
}

/// Struct that generates FFT GPU source code.
pub struct Fft<F: GpuName>(PhantomData<F>);

impl<F: GpuName> Fft<F> {
    pub fn new() -> Self { Self(PhantomData) }
}

impl<F: GpuName> NameAndSource for Fft<F> {
    fn name(&self) -> String { F::name() }

    fn source(&self, _limb: Limb32Or64) -> String {
        String::from(FFT_SRC).replace("FIELD", &F::name())
    }
}

/// Struct that generates FFT for G1 GPU source code.
pub struct Ec<C: GpuCurveName>(PhantomData<C>);

impl<C: GpuCurveName> Ec<C> {
    pub fn new() -> Self { Self(PhantomData) }
}

impl<C: GpuCurveName> NameAndSource for Ec<C> {
    fn name(&self) -> String { C::Affine::name() }

    fn source(&self, _limb: Limb32Or64) -> String {
        String::from(EC_SRC)
            .replace("BASE", &C::Base::name())
            .replace("POINT", &C::Affine::name())
            .replace("SCALAR", &C::Scalar::name())
    }
}

/// Struct that generates FFT for G1 GPU source code.
pub struct EcFft<C: GpuCurveName>(PhantomData<C>);

impl<C: GpuCurveName> EcFft<C> {
    pub fn new() -> Self { Self(PhantomData) }
}

impl<C: GpuCurveName> NameAndSource for EcFft<C> {
    fn name(&self) -> String { C::Affine::name() }

    fn source(&self, _limb: Limb32Or64) -> String {
        String::from(EC_FFT_SRC)
            .replace("POINT", &C::Affine::name())
            .replace("SCALAR", &C::Scalar::name())
    }
}

/// Struct that generates multiexp GPU source code.
#[derive(Default)]
pub struct Multiexp<C: GpuCurveName>(PhantomData<C>);

impl<C: GpuCurveName> Multiexp<C> {
    pub fn new() -> Self { Self(PhantomData) }
}

impl<C: GpuCurveName> NameAndSource for Multiexp<C> {
    fn name(&self) -> String { C::Affine::name() }

    fn source(&self, _limb: Limb32Or64) -> String {
        String::from(MULTIEXP_SRC)
            .replace("POINT", &C::Affine::name())
            .replace("SCALAR", &C::Scalar::name())
    }
}

#[cfg(test)]
/// Struct that generates multiexp GPU source code.
pub struct Test<C: GpuCurveName>(PhantomData<C>);

#[cfg(test)]
impl<C: GpuCurveName> Test<C> {
    pub fn new() -> Self { Self(PhantomData) }
}

#[cfg(test)]
impl<C: GpuCurveName> NameAndSource for Test<C> {
    fn name(&self) -> String { C::Affine::name() }

    fn source(&self, _limb: Limb32Or64) -> String {
        String::from(TEST_SRC)
            .replace("FIELD", &C::Base::name())
            .replace("POINT", &C::Affine::name())
            .replace("SCALAR", &C::Scalar::name())
    }
}
