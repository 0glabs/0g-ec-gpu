//! Builder to create the source code of a GPU kernel.

use std::{collections::BTreeSet, fmt::Write};

use super::{
    limb::Limb32Or64,
    synthesis::{Ec, EcFft, Fft, Field, Multiexp, NameAndSource},
    template::*,
};
use ag_types::{GpuCurveAffine, GpuField};

// In the `HashSet`s the concrete types cannot be used, as each item of the set
// should be able to have its own (different) generic type.
// We distinguish between extension fields and other fields as sub-fields need
// to be defined first in the source code (due to being C, where the order of
// declaration matters).
#[derive(Default)]
pub struct SourceBuilder {
    /// The [`Field`]s that are used in this kernel.
    fields: BTreeSet<Box<dyn NameAndSource>>,
    /// The extension [`Field`]s that are used in this kernel.
    extension_fields: BTreeSet<Box<dyn NameAndSource>>,
    /// The [`Fft`]s that are used in this kernel.
    ffts: BTreeSet<Box<dyn NameAndSource>>,
    ec: BTreeSet<Box<dyn NameAndSource>>,
    /// The [`Fftg`]s that are used in this kernel.
    ec_ffts: BTreeSet<Box<dyn NameAndSource>>,
    /// The [`Multiexp`]s that are used in this kernel.
    multiexps: BTreeSet<Box<dyn NameAndSource>>,
    others: BTreeSet<Box<dyn NameAndSource>>,
    /// Additional source that is appended at the end of the generated source.
    extra_sources: Vec<String>,
}

impl SourceBuilder {
    /// Create a new configuration to generation a GPU kernel.
    pub fn new() -> Self { Self::default() }

    /// Add a field to the configuration.
    ///
    /// If it is an extension field, then the extension field *and* the
    /// sub-field is added.
    pub fn add_field<F>(mut self) -> Self
    where F: GpuField + 'static {
        let field = Field::<F>::new();
        // If it's an extension field, also add the corresponding sub-field.
        if let Some(sub_field_name) = F::sub_field_name() {
            self.extension_fields.insert(Box::new(field));
            let sub_field = Field::<F>::SubField(sub_field_name);
            self.fields.insert(Box::new(sub_field));
        } else {
            self.fields.insert(Box::new(field));
        }
        self
    }

    /// Add an FFT kernel function to the configuration.
    pub fn add_fft<F>(self) -> Self
    where F: GpuField + 'static {
        let mut config = self.add_field::<F>();
        let fft = Fft::<F>::new();
        config.ffts.insert(Box::new(fft));
        config
    }

    pub fn add_ec<C>(self) -> Self
    where C: GpuCurveAffine + 'static {
        let mut config = self.add_field::<C::Base>().add_field::<C::Scalar>();
        let ec = Ec::<C>::new();
        config.ec.insert(Box::new(ec));
        config
    }

    /// Add an FFTg kernel function to the configuration.
    ///
    /// The field must be given explicitly as currently it cannot derived from
    /// the curve point directly.
    pub fn add_ec_fft<C>(self) -> Self
    where C: GpuCurveAffine + 'static {
        let mut config = self.add_ec::<C>();
        let ec_fft = EcFft::<C>::new();
        config.ec_ffts.insert(Box::new(ec_fft));
        config
    }

    /// Add an Multiexp kernel function to the configuration.
    ///
    /// The field must be given explicitly as currently it cannot derived from
    /// the curve point directly.
    pub fn add_multiexp<C>(self) -> Self
    where C: GpuCurveAffine + 'static {
        if cfg!(feature = "opencl") {
            panic!("The source code has not been tested on opencl");
        }
        let mut config = self.add_ec::<C>();
        let multiexp = Multiexp::<C>::new();
        config.multiexps.insert(Box::new(multiexp));
        config
    }

    #[cfg(test)]
    pub fn add_test<C, F>(self) -> Self
    where C: GpuCurveAffine + 'static {
        use super::synthesis::Test;

        let mut config = self.add_ec::<C>();
        let test = Test::<C>::new();
        config.others.insert(Box::new(test));
        config
    }

    /// Appends some given source at the end of the generated source.
    ///
    /// This is useful for cases where you use this library as building block,
    /// but have your own kernel implementation. If this function is is
    /// called several times, then those sources are appended in that call
    /// order.
    pub fn append_source(mut self, source: String) -> Self {
        self.extra_sources.push(source);
        self
    }

    /// Generate the GPU kernel source code based on the current configuration
    /// with 32-bit limbs.
    ///
    /// On CUDA 32-bit limbs are recommended.
    pub fn build_32_bit_limbs(&self) -> String {
        self.build(Limb32Or64::Limb32)
    }

    /// Generate the GPU kernel source code based on the current configuration
    /// with 64-bit limbs.
    ///
    /// On OpenCL 32-bit limbs are recommended.
    pub fn build_64_bit_limbs(&self) -> String {
        self.build(Limb32Or64::Limb64)
    }

    /// Generate the GPU kernel source code based on the current configuration.
    fn build(&self, limb_size: Limb32Or64) -> String {
        let mut answer = COMMON_SRC.into();
        write_field(&mut answer, limb_size, &self.fields);
        write_field(&mut answer, limb_size, &self.extension_fields);
        write_field(&mut answer, limb_size, &self.ec);
        write_field(&mut answer, limb_size, &self.ffts);
        write_field(&mut answer, limb_size, &self.ec_ffts);
        write_field(&mut answer, limb_size, &self.multiexps);
        write_field(&mut answer, limb_size, &self.others);
        write!(answer, "{}", self.extra_sources.join("\n")).unwrap();
        answer
    }
}

fn write_field(
    result: &mut String, limb_size: Limb32Or64,
    field: &BTreeSet<Box<dyn NameAndSource>>,
) {
    for item in field {
        writeln!(result, "{}", item.source(limb_size)).unwrap();
    }
    write!(result, "\n\n").unwrap();
}
