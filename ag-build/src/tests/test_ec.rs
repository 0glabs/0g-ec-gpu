use rand::thread_rng;

use super::{program::CUDA_PROGRAM, types::*};

use ark_ff::UniformRand;

#[test]
fn test_ec() {
    use rust_gpu_tools::{program_closures, GPUError};
    let mut rng = thread_rng();
    for _ in 0..100 {
        let a = Curve::rand(&mut rng);
        let b = Scalar::rand(&mut rng);
        let target = a * b;
        let closures = program_closures!(|program,
                                          _args|
         -> Result<Curve, GPUError> {
            let mut cpu_buffer = vec![Curve::default()];

            let buffer = program.create_buffer_from_slice(&cpu_buffer).unwrap();

            let kernel = program.create_kernel("test_ec", 1, 1).unwrap();
            kernel
                .arg(&GpuCurve(a))
                .arg(&GpuScalar(b))
                .arg(&buffer)
                .run()
                .unwrap();

            program.read_into_buffer(&buffer, &mut cpu_buffer).unwrap();
            Ok(cpu_buffer[0])
        });

        let answer = CUDA_PROGRAM.lock().unwrap().run(closures, ()).unwrap();
        assert_eq!(answer, target);
    }
}
