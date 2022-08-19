#![no_std]

extern crate alloc;

use alloc::vec::Vec;
use alloc::vec;

use num_traits::real::Real;
use num_traits::float::Float;
use num_complex::Complex;

pub struct SlidingDFT<F> {
    // circular sample buffer
    time_domain: Vec<F>,

    // unwindowed output
    freq_domain: Vec<Complex<F>>,

    // windowed output
    output: Vec<Complex<F>>,

    // twiddle factors
    twiddle: Vec<Complex<F>>,

    // size of the buffers
    size: usize,

    // progress into the sample buffer
    i: usize,

    // whether enough samples have been seen
    valid: bool,
}

impl<F: Real + Float + Copy> SlidingDFT<F> {
    pub fn new(size: usize) -> Self {
        let c_zero = Complex::new(F::zero(), F::zero());
        let ref_twiddle = Complex::new(F::zero(), F::one());
        let tau = Float::to_radians(F::from(360).unwrap());

        let f_size = F::from(size).unwrap();
        let mut twiddle = vec![c_zero; size];

        for k in 0..size {
            let factor = tau * F::from(k).unwrap() / f_size;
            twiddle[k] = (ref_twiddle * factor).exp();
        }

        Self {
            time_domain: vec![F::zero(); size],
            freq_domain: vec![c_zero; size],
            output: vec![c_zero; size],
            twiddle,
            i: 0,
            size,
            valid: false,
        }
    }

    pub fn output(&self) -> Option<&[Complex<F>]> {
        match self.valid {
            true => Some(&self.output),
            false => None,
        }
    }

    pub fn update(&mut self, sample: F) -> Option<&[Complex<F>]> {
        // Update the storage of the time domain values
        let prev_value = self.time_domain[self.i];
        self.time_domain[self.i] = sample;

        let r = F::one() - <F as Real>::epsilon();
        let r_pow_size = Real::powi(r, self.size as i32);

        // Update the DFT
        for k in 0..self.size {
            let pre_twiddle = self.freq_domain[k] * r - r_pow_size * prev_value + sample;
            self.freq_domain[k] = self.twiddle[k] * pre_twiddle;
        }

        let half = F::from(0.5_f32).unwrap();
        let fourth = F::from(0.25_f32).unwrap();

        // Apply the Hanning window
        for k in 0..self.size {
            let k_m1 = k.checked_sub(1).unwrap_or(self.size - 1);
            let k_p1 = (k + 1) % self.size;

            let prev = self.freq_domain[k_m1];
            let this = self.freq_domain[k];
            let next = self.freq_domain[k_p1];

            self.output[k] = this * half - (prev + next) * fourth;
        }

        // Increment the counter
        self.i += 1;
        if self.i >= self.size {
            self.valid = true;
            self.i = 0;
        }

        // Done.
        return self.output();
    }
}
