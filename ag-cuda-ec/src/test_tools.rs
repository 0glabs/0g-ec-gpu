use ark_ff::UniformRand;
use ark_std::rand::Rng;

pub fn random_input<T: UniformRand, R: Rng>(
    length: usize, rng: &mut R,
) -> Vec<T> {
    (0..length).map(|_| T::rand(rng)).collect::<Vec<_>>()
}

pub fn random_input_by_cycle<T: UniformRand + Clone, R: Rng>(
    length: usize, period: usize, rng: &mut R,
) -> Vec<T> {
    let meta = random_input(period, rng);
    meta.iter().cycle().cloned().take(length).collect()
}
