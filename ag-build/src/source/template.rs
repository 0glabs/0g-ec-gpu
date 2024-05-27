use super::limb::{Limb, Limb32, Limb32Or64, Limb64};
use ag_types::GpuField;
use std::fmt::Write;

macro_rules! include_cl {
    ($file:literal) => {
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/cl/", $file))
    };
}

pub static COMMON_SRC: &str = include_cl!("common.cl");
pub static FIELD_SRC: &str = include_cl!("field.cl");
pub static FIELD2_SRC: &str = include_cl!("field2.cl");
pub static EC_SRC: &str = include_cl!("ec.cl");
pub static FFT_SRC: &str = include_cl!("fft.cl");
pub static EC_FFT_SRC: &str = include_cl!("ec-fft.cl");
pub static MULTIEXP_SRC: &str = include_cl!("multiexp.cl");

#[cfg(test)]
pub static TEST_SRC: &str = include_cl!("test.cl");

pub fn const_field<L: Limb>(name: &str, limbs: Vec<L>) -> String {
    format!(
        "CONSTANT FIELD {} = {{ {{ {} }} }};",
        name,
        limbs
            .iter()
            .map(|l| l.value().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// Generates CUDA/OpenCL constants and type definitions of prime-field `F`
pub fn params<F, L>() -> String
where
    F: GpuField,
    L: Limb,
{
    let one = L::one_limbs::<F>(); // Get Montgomery form of F::one()
    let p = L::modulus_limbs::<F>(); // Get field modulus in non-Montgomery form
    let r2 = L::calculate_r2::<F>();
    let limbs = one.len(); // Number of limbs
    let inv = L::calc_inv(p[0]);
    let limb_def = format!("#define FIELD_limb {}", L::opencl_type());
    let limbs_def = format!("#define FIELD_LIMBS {}", limbs);
    let limb_bits_def = format!("#define FIELD_LIMB_BITS {}", L::bits());
    let p_def = const_field("FIELD_P", p);
    let r2_def = const_field("FIELD_R2", r2);
    let one_def = const_field("FIELD_ONE", one);
    let zero_def = const_field("FIELD_ZERO", vec![L::zero(); limbs]);
    let inv_def = format!("#define FIELD_INV {}", inv.value());
    let type_def =
        "typedef struct { FIELD_limb val[FIELD_LIMBS]; } FIELD;".to_string();
    let type_repr_def =
        "typedef struct { FIELD_limb val[FIELD_LIMBS]; } FIELD_repr;"
            .to_string();
    [
        limb_def,
        limbs_def,
        limb_bits_def,
        inv_def,
        type_def,
        type_repr_def,
        one_def,
        p_def,
        r2_def,
        zero_def,
    ]
    .join("\n")
}

pub fn field_source<F: GpuField>(limb: Limb32Or64) -> String {
    match limb {
        Limb32Or64::Limb32 => [
            params::<F, Limb32>(),
            field_add_sub_nvidia::<F, Limb32>().expect("preallocated"),
            String::from(FIELD_SRC),
        ]
        .join("\n"),
        Limb32Or64::Limb64 => [
            params::<F, Limb64>(),
            field_add_sub_nvidia::<F, Limb64>().expect("preallocated"),
            String::from(FIELD_SRC),
        ]
        .join("\n"),
    }
}

/// Generates PTX-Assembly implementation of FIELD_add_/FIELD_sub_
fn field_add_sub_nvidia<F, L>() -> Result<String, std::fmt::Error>
where
    F: GpuField,
    L: Limb,
{
    let mut result = String::new();
    let (ptx_type, ptx_reg) = L::ptx_info();

    writeln!(result, "#if defined(OPENCL_NVIDIA) || defined(CUDA)\n")?;
    for op in &["sub", "add"] {
        let len = L::one_limbs::<F>().len();

        writeln!(
            result,
            "DEVICE FIELD FIELD_{}_nvidia(FIELD a, FIELD b) {{",
            op
        )?;
        if len > 1 {
            write!(result, "asm(")?;
            writeln!(
                result,
                "\"{}.cc.{} %0, %0, %{};\\r\\n\"",
                op, ptx_type, len
            )?;

            for i in 1..len - 1 {
                writeln!(
                    result,
                    "\"{}c.cc.{} %{}, %{}, %{};\\r\\n\"",
                    op,
                    ptx_type,
                    i,
                    i,
                    len + i
                )?;
            }
            writeln!(
                result,
                "\"{}c.{} %{}, %{}, %{};\\r\\n\"",
                op,
                ptx_type,
                len - 1,
                len - 1,
                2 * len - 1
            )?;

            write!(result, ":")?;
            for n in 0..len {
                write!(result, "\"+{}\"(a.val[{}])", ptx_reg, n)?;
                if n != len - 1 {
                    write!(result, ", ")?;
                }
            }

            write!(result, "\n:")?;
            for n in 0..len {
                write!(result, "\"{}\"(b.val[{}])", ptx_reg, n)?;
                if n != len - 1 {
                    write!(result, ", ")?;
                }
            }
            writeln!(result, ");")?;
        }
        writeln!(result, "return a;\n}}")?;
    }
    writeln!(result, "#endif")?;

    Ok(result)
}
