extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ExprClosure, FnArg, Ident, ItemFn, PatType};

#[proc_macro_attribute]
pub fn auto_workspace(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(item as ItemFn);

    if input_fn.sig.inputs.len() < 1 {
        return syn::Error::new_spanned(
            &input_fn.sig.inputs,
            "Function must have at least one parameter",
        )
        .to_compile_error()
        .into();
    }

    let fn_name = &input_fn.sig.ident;
    let generics = &input_fn.sig.generics; 
    let generic_args_call = if generics.params.is_empty() {
        quote! {}
    } else {
        let generic_args = generics
            .params
            .iter()
            .map(|x| match x {
                syn::GenericParam::Lifetime(syn::LifetimeParam {
                    lifetime,
                    ..
                }) => quote! {#lifetime },
                syn::GenericParam::Type(syn::TypeParam { ident, .. }) => {
                    quote! { #ident}
                }
                syn::GenericParam::Const(syn::ConstParam { ident, .. }) => {
                    quote! { #ident}
                }
            })
            .collect::<Vec<_>>();

        quote! { :: <#(#generic_args),*> }
    };
    let st_fn_name = Ident::new(&format!("{}_st", fn_name), fn_name.span());
    let mt_fn_name = Ident::new(&format!("{}_mt", fn_name), fn_name.span());

    let fn_args: Vec<_> = input_fn.sig.inputs.iter().skip(1).collect(); // 跳过第一个参数
    let fn_args_names: Vec<_> = input_fn
        .sig
        .inputs
        .iter()
        .skip(1)
        .map(|arg| match arg {
            FnArg::Typed(PatType { pat, .. }) => quote! { #pat },
            _ => quote! {},
        })
        .collect();
    let fn_return_type = &input_fn.sig.output;

    let output_fn = quote! {
        #input_fn

        pub fn #mt_fn_name #generics (#(#fn_args),*) #fn_return_type {
            LOCAL.with(|auto_workspace_macro_inner__| #fn_return_type {
                let workspace = auto_workspace_macro_inner__.activate()?;
                #fn_name #generic_args_call (&workspace, #(#fn_args_names),*)
            })
        }

        pub fn #st_fn_name #generics (#(#fn_args),*) #fn_return_type {
            let workspace = GLOBAL.activate()?;
            #fn_name #generic_args_call (&workspace, #(#fn_args_names),*)
        }
    };

    output_fn.into()
}

#[proc_macro]
pub fn construct_workspace(item: TokenStream) -> TokenStream {
    let closure = parse_macro_input!(item as ExprClosure);

    let output = quote! {
        static GLOBAL: once_cell::sync::Lazy<CudaWorkspace> = once_cell::sync::Lazy::new(#closure);

        std::thread_local! {
            static LOCAL: once_cell::unsync::Lazy<CudaWorkspace> = once_cell::unsync::Lazy::new(#closure);
        }

        pub fn init_global_workspace() {
            let _x = &*GLOBAL;
        }

        pub fn init_local_workspace() {
            let _y = LOCAL.with(|x| { once_cell::unsync::Lazy::force(x); });
        }
    };

    output.into()
}
