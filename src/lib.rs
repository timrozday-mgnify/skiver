pub mod analyze;
pub mod cmdline;
pub mod constants;
pub mod dump;
pub mod huber;
pub mod inference;
pub mod kvmer;
pub mod mapping;
pub mod seeding;
pub mod sketch;
pub mod summary;
pub mod types;
pub mod utils;

#[cfg(target_arch = "x86_64")]
pub mod avx2_seeding;

#[cfg(test)]
mod tests {
    use crate::types::EditOperation::*;
    use crate::utils::{_get_neighbors, _kmer_to_string};

    /// Decoding round-trips the 2-bit encoding, and no substitution neighbour of
    /// a value equals the value itself.
    #[test]
    fn test_kmer_neighbors() {
        let value = 0b11_00_00_01_11_01_11; // 7 bases
        assert_eq!(_kmer_to_string(value, 7), "TAACTCT");

        let neighbors = _get_neighbors(value, 7);
        assert!(!neighbors.is_empty());
        for (&nbr, info) in &neighbors {
            let is_substitution = matches!(
                info.op,
                AC | AG | AT | CA | CG | CT | GA | GC | GT | TA | TC | TG
            );
            if is_substitution {
                assert_ne!(nbr, value);
            }
        }
    }
}
