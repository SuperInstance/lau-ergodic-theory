//! # lau-ergodic-theory
//!
//! Ergodic theory library implementing the mathematics of long-term statistical
//! behavior of dynamical systems. Covers measure-preserving transformations,
//! ergodicity, Birkhoff's ergodic theorem, mixing, Kolmogorov-Sinai entropy,
//! Lyapunov exponents, Perron-Frobenius operators, and agent behavior prediction.

pub mod measure;
pub mod ergodicity;
pub mod birkhoff;
pub mod mixing;
pub mod entropy;
pub mod shannon_mcmillan;
pub mod markov;
pub mod lyapunov;
pub mod perron_frobenius;
pub mod agent_prediction;

pub use measure::{Measure, MeasureSpace, Transformation};
pub use ergodicity::ErgodicChecker;
pub use birkhoff::BirkhoffAverage;
pub use mixing::MixingChecker;
pub use entropy::KolmogorovSinaiEntropy;
pub use shannon_mcmillan::ShannonMcMillanBreiman;
pub use markov::MarkovChain;
pub use lyapunov::LyapunovExponent;
pub use perron_frobenius::PerronFrobeniusOperator;
pub use agent_prediction::AgentPredictor;
