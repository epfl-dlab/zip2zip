use bumpalo::{collections::Vec as BumpVec, Bump};
use fastset::Set;
use itertools::Itertools;
use pyo3::prelude::*;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use rustc_hash::FxHashMap;
use std::collections::HashMap;

////////////////////////////////////////////////////////////
// LZW Compression
////////////////////////////////////////////////////////////
#[inline(always)]
fn codebook_contains(
    codebook: &FxHashMap<Vec<usize>, usize>,
    ids: &Vec<usize>,
    initial_vocab_size: usize,
) -> bool {
    if ids.len() == 1 {
        ids[0] < initial_vocab_size
    } else {
        codebook.contains_key(ids)
    }
}

#[inline(always)]
fn get_usize_from_codebook(codebook: &FxHashMap<Vec<usize>, usize>, ids: &Vec<usize>) -> usize {
    if ids.len() == 1 {
        ids[0]
    } else {
        codebook.get(ids).unwrap().clone()
    }
}

#[inline(always)]
fn disabled_ids_to_set(disabled_ids: Option<Vec<usize>>) -> Set {
    disabled_ids.map_or_else(
        || Set::with_capacity(0),
        |d_ids| {
            let mut set = Set::with_capacity(d_ids.iter().max().unwrap() + 1);
            for id in d_ids {
                set.insert(id);
            }
            set
        },
    )
}

fn encode(
    ids: &Vec<usize>,
    initial_vocab_size: usize,
    max_codebook_size: usize,
    max_subtokens: usize,
    disabled_ids: &Set,
) -> (Vec<usize>, Vec<usize>) {
    let mut compressed_ids: Vec<usize> = Vec::new();
    let mut codebook: FxHashMap<Vec<usize>, usize> = FxHashMap::default();

    let mut next_id: usize = initial_vocab_size;
    let mut ids_to_merge: Vec<usize> = Vec::with_capacity(max_subtokens);

    for &id in ids {
        if disabled_ids.contains(&id) {
            if ids_to_merge.len() > 0 {
                compressed_ids.push(get_usize_from_codebook(&codebook, &ids_to_merge));
                ids_to_merge.clear();
            }
            compressed_ids.push(id);
            continue;
        }

        ids_to_merge.push(id);

        let is_in_codebook = codebook_contains(&codebook, &ids_to_merge, initial_vocab_size);
        if !is_in_codebook {
            if next_id < initial_vocab_size + max_codebook_size {
                codebook.insert(ids_to_merge.clone(), next_id);
                next_id += 1;
            }

            ids_to_merge.pop();
            compressed_ids.push(get_usize_from_codebook(&codebook, &ids_to_merge));
            ids_to_merge.clear();
            ids_to_merge.push(id);
        }

        if ids_to_merge.len() == max_subtokens {
            compressed_ids.push(get_usize_from_codebook(&codebook, &ids_to_merge));
            ids_to_merge.clear();
        }
    }

    if ids_to_merge.len() > max_subtokens {
        let last_id = ids_to_merge.pop().unwrap();
        compressed_ids.push(get_usize_from_codebook(&codebook, &ids_to_merge));
        ids_to_merge.clear();
        ids_to_merge.push(last_id);
    }

    if ids_to_merge.len() > 0 {
        compressed_ids.push(get_usize_from_codebook(&codebook, &ids_to_merge));
    }

    let attention_mask = vec![1; compressed_ids.len()];

    (compressed_ids, attention_mask)
}

fn decode(
    compressed_ids: &Vec<usize>,
    initial_vocab_size: usize,
    max_codebook_size: usize,
    max_subtokens: usize,
    disabled_ids: &Set,
) -> Vec<usize> {
    let bump = Bump::new();

    let mut ids: Vec<usize> = Vec::with_capacity(compressed_ids.len());
    let mut codebook: FxHashMap<usize, &[usize]> = FxHashMap::default();

    let mut next_id: usize = initial_vocab_size;
    let mut previous_ids: &[usize] = &[];

    for &id in compressed_ids {
        if disabled_ids.contains(&id) {
            previous_ids = &[];
            ids.push(id);
            continue;
        }

        let current_ids: &[usize];

        if id < initial_vocab_size {
            current_ids = bump.alloc_slice_copy(&[id]);
        } else if let Some(slice) = codebook.get(&id) {
            current_ids = slice;
        } else if previous_ids.len() == max_subtokens {
            current_ids = previous_ids;
        } else {
            let mut inferred_vec = BumpVec::with_capacity_in(previous_ids.len() + 1, &bump);
            inferred_vec.extend_from_slice(previous_ids);
            inferred_vec.push(previous_ids[0]);

            current_ids = inferred_vec.into_bump_slice();

            codebook.insert(id, current_ids);
        }

        ids.extend_from_slice(current_ids);

        if !previous_ids.is_empty()
            && next_id < initial_vocab_size + max_codebook_size
            && previous_ids.len() < max_subtokens
        {
            let mut new_entry_vec = BumpVec::with_capacity_in(previous_ids.len() + 1, &bump);
            new_entry_vec.extend_from_slice(previous_ids);
            new_entry_vec.push(current_ids[0]);

            let new_entry_slice = new_entry_vec.into_bump_slice();

            codebook.insert(next_id, new_entry_slice);
            next_id += 1;
        }

        previous_ids = current_ids;
    }

    ids
}

#[pyclass]
pub struct CodebookManager {
    initial_vocab_size: usize,
    max_codebook_size: usize,
    max_subtokens: usize,
    pad_token_id: usize,
    disabled_ids: Set,
    codebook: HashMap<usize, Vec<usize>>,
    previous_ids: Vec<usize>,
    next_id: usize,
}

#[pymethods]
impl CodebookManager {
    #[new]
    fn new(
        initial_vocab_size: usize,
        max_codebook_size: usize,
        max_subtokens: usize,
        pad_token_id: usize,
        disabled_ids: Option<Vec<usize>>,
    ) -> Self {
        Self {
            initial_vocab_size,
            max_codebook_size,
            max_subtokens,
            pad_token_id,
            disabled_ids: disabled_ids_to_set(disabled_ids),
            codebook: HashMap::with_capacity(max_codebook_size),
            previous_ids: Vec::new(),
            next_id: initial_vocab_size,
        }
    }

    fn get_subtokens(&self, id: usize) -> Vec<usize> {
        self.codebook.get(&id).cloned().unwrap_or_else(|| vec![id])
    }

    fn update_codebook(&mut self, ids: Vec<usize>, prefill: bool) -> (Vec<Vec<usize>>, usize) {
        let mut updates_set = Set::new(self.max_codebook_size);

        for id in ids {
            if self.disabled_ids.contains(&id) {
                self.previous_ids.clear();
                continue;
            }

            let mut current_ids: Vec<usize>;

            if id < self.initial_vocab_size {
                current_ids = vec![id];
            } else if let Some(entry) = self.codebook.get(&id) {
                current_ids = entry.clone();
            } else if self.previous_ids.len() == self.max_subtokens {
                current_ids = self.previous_ids.clone();
            } else {
                current_ids = self.previous_ids.clone();
                current_ids.push(self.previous_ids[0]);

                self.codebook.insert(id, current_ids.clone());
                updates_set.insert(id);
            }

            if !self.previous_ids.is_empty()
                && self.next_id < self.initial_vocab_size + self.max_codebook_size
                && self.previous_ids.len() < self.max_subtokens
            {
                self.previous_ids.push(current_ids[0]);

                self.codebook
                    .insert(self.next_id, self.previous_ids.clone());
                updates_set.insert(self.next_id);
                self.next_id += 1;
            }

            self.previous_ids = current_ids;
        }

        let mut updates: Vec<Vec<usize>> = self
            .codebook
            .iter()
            .filter(|(key, _)| updates_set.contains(*key))
            .sorted_by_key(|(key, _)| *key)
            .map(|(_, value)| {
                let mut v = value.clone();
                v.resize(self.max_subtokens, self.pad_token_id);
                v
            })
            .collect();

        if prefill {
            updates.resize(
                self.max_codebook_size,
                vec![self.pad_token_id; self.max_subtokens],
            );
        }

        (updates, updates_set.len())
    }

    fn reset(&mut self) {
        self.codebook.clear();
        self.previous_ids.clear();
        self.next_id = self.initial_vocab_size;
    }
}

////////////////////////////////////////////////////////////
// Legacy LZW Compression
////////////////////////////////////////////////////////////
fn vocab_contains(
    extra_vocab: &HashMap<Vec<usize>, usize>,
    v: &Vec<usize>,
    initial_vocab_size: usize,
) -> bool {
    if v.len() == 1 {
        v[0] < initial_vocab_size
    } else {
        extra_vocab.contains_key(v)
    }
}

fn get_from_vocab(extra_vocab: &HashMap<Vec<usize>, usize>, v: &Vec<usize>) -> Option<usize> {
    if v.len() == 1 {
        Some(v[0])
    } else {
        extra_vocab.get(v).cloned()
    }
}

fn is_disabled(disabled_ids: Option<&Vec<usize>>, id: usize) -> bool {
    if let Some(disabled_ids) = disabled_ids {
        disabled_ids.contains(&id)
    } else {
        false
    }
}

/// Apply LZW compression to input ids with a target output length, truncating at out_seq_length if necessary
fn lzw(
    ids: Vec<usize>,
    initial_vocab_size: usize,
    extra_vocab_size: usize,
    max_out_seq_length: usize,
    max_subtokens: usize,
    disabled_ids: Option<Vec<usize>>,
) -> (Vec<usize>, Vec<usize>, HashMap<Vec<usize>, usize>) {
    let mut compressed_ids: Vec<usize> = Vec::new();
    let mut extra_merges = HashMap::with_capacity(extra_vocab_size);

    let mut i = 0;
    let mut next_compressed_id = initial_vocab_size;
    let mut ids_to_merge: Vec<usize> = Vec::new();

    while i < ids.len() && compressed_ids.len() < max_out_seq_length - 1 {
        let id = ids[i];

        let disabled = is_disabled(disabled_ids.as_ref(), id);
        if disabled {
            if ids_to_merge.len() > 0 {
                compressed_ids.push(get_from_vocab(&extra_merges, &ids_to_merge).unwrap());
                ids_to_merge = vec![];
            }
            compressed_ids.push(id);
            i += 1;
            continue;
        }

        ids_to_merge.push(id);

        let vocab_contains = vocab_contains(&extra_merges, &ids_to_merge, initial_vocab_size);
        if !vocab_contains || ids_to_merge.len() == max_subtokens {
            if !vocab_contains && next_compressed_id < initial_vocab_size + extra_vocab_size {
                extra_merges.insert(ids_to_merge.clone(), next_compressed_id);
                next_compressed_id += 1;
            }

            // based on LZW algorithm, the last token should be kept to the next iteration
            ids_to_merge.pop();
            compressed_ids.push(get_from_vocab(&extra_merges, &ids_to_merge).unwrap());
            ids_to_merge = vec![id];
        }

        i += 1;
    }

    if ids_to_merge.len() > max_subtokens {
        let last_token = ids_to_merge.pop().unwrap();
        compressed_ids.push(get_from_vocab(&extra_merges, &ids_to_merge).unwrap());
        ids_to_merge = vec![last_token];
    }

    if ids_to_merge.len() > 0 {
        compressed_ids.push(get_from_vocab(&extra_merges, &ids_to_merge).unwrap());
    }

    let remaining_ids = ids[i..].to_vec();

    (compressed_ids, remaining_ids, extra_merges)
}

fn chunk_lzw(
    ids: Vec<usize>,
    initial_vocab_size: usize,
    extra_vocab_size: usize,
    chunk_length: usize,
    max_subtokens: usize,
    disabled_ids: Option<Vec<usize>>,
) -> Vec<(Vec<usize>, HashMap<String, usize>)> {
    let mut remaining_ids = ids;
    let mut compressed_chunks = Vec::new();

    while !remaining_ids.is_empty() {
        let (compressed_ids, new_remaining_ids, merges) = lzw(
            remaining_ids,
            initial_vocab_size,
            extra_vocab_size,
            chunk_length,
            max_subtokens,
            disabled_ids.clone(),
        );

        let hashable_merges = merges
            .iter()
            .map(|(k, v)| {
                (
                    k.iter()
                        .map(|i| i.to_string())
                        .collect::<Vec<String>>()
                        .join(","),
                    *v,
                )
            })
            .collect::<HashMap<String, usize>>();
        compressed_chunks.push((compressed_ids, hashable_merges));
        remaining_ids = new_remaining_ids;
    }

    compressed_chunks
}

#[pyfunction]
#[pyo3(name = "encode")]
fn py_encode(
    ids: Vec<usize>,
    initial_vocab_size: usize,
    max_codebook_size: usize,
    max_subtokens: usize,
    disabled_ids: Option<Vec<usize>>,
) -> PyResult<(Vec<usize>, Vec<usize>)> {
    Ok(encode(
        &ids,
        initial_vocab_size,
        max_codebook_size,
        max_subtokens,
        &disabled_ids_to_set(disabled_ids),
    ))
}

#[pyfunction]
#[pyo3(name = "batch_encode")]
fn py_batch_encode(
    ids: Vec<Vec<usize>>,
    initial_vocab_size: usize,
    max_codebook_size: usize,
    max_subtokens: usize,
    disabled_ids: Option<Vec<usize>>,
) -> PyResult<(Vec<Vec<usize>>, Vec<Vec<usize>>)> {
    let disabled_ids = disabled_ids_to_set(disabled_ids);

    Ok(ids
        .par_iter()
        .map(|sequence| {
            encode(
                sequence,
                initial_vocab_size,
                max_codebook_size,
                max_subtokens,
                &disabled_ids,
            )
        })
        .unzip())
}

#[pyfunction]
#[pyo3(name = "decode")]
fn py_decode(
    compressed_ids: Vec<usize>,
    initial_vocab_size: usize,
    max_codebook_size: usize,
    max_subtokens: usize,
    disabled_ids: Option<Vec<usize>>,
) -> PyResult<Vec<usize>> {
    Ok(decode(
        &compressed_ids,
        initial_vocab_size,
        max_codebook_size,
        max_subtokens,
        &disabled_ids_to_set(disabled_ids),
    ))
}

#[pyfunction]
#[pyo3(name = "batch_decode")]
fn py_batch_decode(
    compressed_ids: Vec<Vec<usize>>,
    initial_vocab_size: usize,
    max_codebook_size: usize,
    max_subtokens: usize,
    disabled_ids: Option<Vec<usize>>,
) -> PyResult<Vec<Vec<usize>>> {
    let disabled_ids = disabled_ids_to_set(disabled_ids);

    Ok(compressed_ids
        .par_iter()
        .map(|sequence| {
            decode(
                sequence,
                initial_vocab_size,
                max_codebook_size,
                max_subtokens,
                &disabled_ids,
            )
        })
        .collect())
}

/// Apply LZW compression to a sequence of ids with chunking
#[pyfunction]
fn lzw_compress(
    ids: Vec<usize>,
    initial_vocab_size: usize,
    extra_vocab_size: usize,
    max_out_seq_length: usize,
    max_subtokens: usize,
    disabled_ids: Option<Vec<usize>>,
) -> PyResult<Vec<(Vec<usize>, HashMap<String, usize>)>> {
    // Call the core compression function
    let compressed_chunks = chunk_lzw(
        ids,
        initial_vocab_size,
        extra_vocab_size,
        max_out_seq_length,
        max_subtokens,
        disabled_ids,
    );

    Ok(compressed_chunks)
}

#[pyfunction]
fn batch_lzw_compress(
    ids: Vec<Vec<usize>>,
    initial_vocab_size: usize,
    extra_vocab_size: usize,
    max_out_seq_length: usize,
    max_subtokens: usize,
    disabled_ids: Option<Vec<usize>>,
) -> PyResult<Vec<(Vec<usize>, HashMap<String, usize>)>> {
    // Compress each sequence in parallel and flatten the results
    let results: Vec<_> = ids
        .par_iter()
        .map(|sequence| {
            chunk_lzw(
                sequence.clone(),
                initial_vocab_size,
                extra_vocab_size,
                max_out_seq_length,
                max_subtokens,
                disabled_ids.clone(),
            )
        })
        .flatten()
        .collect();

    Ok(results)
}

#[pymodule]
fn fast_compression(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lzw_compress, m)?)?;
    m.add_function(wrap_pyfunction!(batch_lzw_compress, m)?)?;
    m.add_function(wrap_pyfunction!(py_encode, m)?)?;
    m.add_function(wrap_pyfunction!(py_batch_encode, m)?)?;
    m.add_function(wrap_pyfunction!(py_decode, m)?)?;
    m.add_function(wrap_pyfunction!(py_batch_decode, m)?)?;
    m.add_class::<CodebookManager>()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::distr::Uniform;
    use rand::{rng, Rng};
    use std::time::Instant;

    #[test]
    fn benchmark_encode() {
        let ids_len = 1024 * 1024;
        let initial_vocab_size = 32011;
        let extra_vocab_size = 2048;
        let max_subtokens = 4;
        let disabled_ids = vec![
            32000, 32001, 32002, 32003, 32004, 32005, 32006, 32007, 32008, 32009, 32010, 11, 12,
            13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        ];

        let rng = rng();
        let range = Uniform::new(0, initial_vocab_size).unwrap();
        let ids: Vec<usize> = rng.sample_iter(range).take(ids_len).collect();

        let start = Instant::now();
        let (compressed_ids, _) = encode(
            &ids,
            initial_vocab_size,
            extra_vocab_size,
            max_subtokens,
            &disabled_ids_to_set(Some(disabled_ids.clone())),
        );
        let duration = start.elapsed();
        println!(
            "encode time taken: {:?}, speed: {:.2} mega tokens/s",
            duration,
            ids_len as f64 / duration.as_secs_f64() / 1_000_000.0
        );

        let start = Instant::now();
        let decoded_ids = decode(
            &compressed_ids,
            initial_vocab_size,
            extra_vocab_size,
            max_subtokens,
            &disabled_ids_to_set(Some(disabled_ids.clone())),
        );
        let duration = start.elapsed();
        println!(
            "decode time taken: {:?}, speed: {:.2} mega tokens/s",
            duration,
            ids_len as f64 / duration.as_secs_f64() / 1_000_000.0
        );

        for (i, (&id, decoded_id)) in ids.iter().zip(decoded_ids.clone()).enumerate() {
            if id != decoded_id {
                panic!("i: {}, id: {}, decoded_id: {}", i, id, decoded_id);
            }
        }
    }

    #[test]
    fn test_encode_decode() {
        let initial_vocab_size = 32011;
        let extra_vocab_size = 2048;
        let max_subtokens = 4;

        let ids_list = vec![
            vec![
                32010, 1815, 366, 10683, 445, 365, 29999, 29956, 2094, 6119, 515, 390, 504, 304,
                5132, 29973, 13, 13, 28956, 23575, 13, 1509, 289, 3427, 7003, 1057, 29912, 29027,
                1057, 25987, 408, 350, 3427, 25987, 29892, 350, 3427, 3400, 13, 1509, 5172, 842,
                1057, 2697, 29936, 13, 1509, 282, 9029, 29941, 1057, 1457, 29880, 1151, 1057,
                29930, 29936, 13, 1509, 15570, 265, 1057, 1524, 1057, 29912, 797, 517, 2177, 6553,
                5620, 20277, 29892, 1459, 6553, 20277, 3400, 13, 1509, 21580, 29883, 29918, 8568,
                1057, 29943, 29916, 27824, 29936, 13, 1509, 3659, 1057, 29027, 1057, 27824, 29936,
                13, 13, 19511, 19511, 19511, 10797, 6165, 13, 458, 365, 29999, 29956, 422, 2590,
                13, 19511, 19511, 19511, 10797, 6165, 13, 29937, 29961, 14764, 29898, 21936, 4638,
                13, 9144, 775, 2909, 29918, 11516, 29898, 13, 1678, 775, 2909, 29901, 669, 29943,
                29916, 27824, 29966, 25987, 29966, 375, 675, 10202, 502, 675, 10202, 13, 1678,
                18999, 29901, 669, 25987, 29966, 375, 675, 10202, 13, 1678, 2847, 29918, 29894,
                542, 370, 29918, 2311, 29901, 502, 675, 29892, 13, 29897, 1599, 6120, 426, 13,
                1678, 565, 18999, 13, 28956, 32007, 32001,
            ],
            (0..2048).map(|_| 34).collect(),
        ];

        for ids in ids_list {
            let (compressed_ids, _) = encode(
                &ids,
                initial_vocab_size,
                extra_vocab_size,
                max_subtokens,
                &disabled_ids_to_set(Some(vec![
                    0, 1, 2, 32000, 32001, 32002, 32003, 32004, 32005, 32006, 32007, 32008, 32009,
                    32010,
                ])),
            );

            let decoded_ids = decode(
                &compressed_ids,
                initial_vocab_size,
                extra_vocab_size,
                max_subtokens,
                &disabled_ids_to_set(Some(vec![
                    0, 1, 2, 32000, 32001, 32002, 32003, 32004, 32005, 32006, 32007, 32008, 32009,
                    32010,
                ])),
            );

            assert_eq!(ids, decoded_ids);
        }
    }

    #[test]
    fn test_encode() {
        let initial_vocab_size = 32011;
        let extra_vocab_size = 2048;
        let max_subtokens = 4;

        let ids = vec![
            671, 289, 3427, 7003, 1057, 29912, 29027, 1057, 25987, 408, 350, 3427, 25987, 29892,
            350, 3427, 3400, 13, 1509, 5172, 842, 1057, 2697, 29936, 13, 1509, 4256, 8504, 1057,
            13463, 8504, 29936, 13, 1509, 282, 9029, 29941, 1057, 1457, 29880, 1151, 1057, 29930,
            29936, 13, 1509, 15570, 265, 1057, 1524, 1057, 29912, 797, 517, 2177, 6553, 5620,
            20277, 29892, 1459, 6553, 20277, 3400, 13, 1509, 21580, 29883, 29918, 8568, 1057,
            29943, 29916, 27824, 29936, 13, 1509, 3659, 1057, 29027, 1057, 27824, 29936, 13, 13,
            19511, 19511, 19511, 10797, 6165, 13, 458, 365, 29999, 29956, 422, 2590, 13, 19511,
            19511, 19511, 10797, 6165, 13, 29937, 29961, 14764, 29898, 21936, 4638, 13, 9144, 775,
            2909, 29918, 11516, 29898, 13, 1678, 775, 2909, 29901, 669, 29943, 29916, 27824, 29966,
            25987, 29966, 375, 675, 10202, 502, 675, 10202, 13, 1678, 18999, 29901, 669, 25987,
            29966, 375, 675, 10202, 13, 1678, 2847, 29918, 29894, 542, 370, 29918, 2311, 29901,
            502, 675, 29892, 13, 29897, 1599, 6120, 426, 13, 1678, 565, 18999, 29889, 2435, 580,
            1275, 29871, 29896, 426, 13, 4706, 18999, 29961, 29900, 29962, 529, 2847, 29918, 29894,
            542, 370, 29918, 2311, 13, 1678, 500, 1683, 426, 13, 4706, 775, 2909, 29889, 11516,
            29918, 1989, 29898, 4841, 29897, 13, 1678, 500, 13, 29913, 13, 13, 29937, 29961, 14764,
            29898, 21936, 4638, 13, 9144, 679, 29918, 375, 675, 29918, 3166, 29918, 401, 2909,
            29898, 401, 2909, 29901, 669, 29943, 29916, 27824, 29966, 25987, 29966, 375, 675,
            10202, 502, 675, 10202, 18999, 29901, 669, 25987, 29966, 375, 675, 12948, 1599, 502,
            675, 426, 13, 1678, 565, 18999, 29889, 2435, 580, 1275, 29871, 29896, 426, 13, 4706,
            18999, 29961, 29900, 29962, 13, 1678, 500, 1683, 426, 13, 4706, 775, 2909, 29889, 657,
            29898, 4841, 467, 26238, 2141, 16513, 580, 13, 1678, 500, 13, 29913, 13, 13, 29937,
            29961, 14764, 29898, 21936, 4638, 13, 9144, 12708, 29918, 4841, 29918, 517, 29918, 842,
            29898, 18279, 29918, 4841, 29901, 10831, 29966, 25987, 29966, 375, 675, 6778, 29897,
            1599, 3789, 426, 13, 1678, 12708, 29918, 4841, 29889, 1958, 29918, 272, 29918, 2870,
            29898, 13, 4706, 3830, 3789, 1057, 2541, 29918, 5030, 5946, 29898, 29900, 511, 13,
            4706, 891, 29881, 29918, 4841, 29989, 426, 13, 9651, 1235, 5478, 731, 353, 3789, 1057,
            2541, 29918, 5030, 5946, 29898, 29881, 29918, 4841, 29889, 1524, 2141, 3317, 2141,
            26238, 580, 718, 29871, 29896, 416, 13, 9651, 363, 1178, 297, 270, 29918, 4841, 426,
            13, 18884, 731, 29889, 7851, 29898, 333, 416, 13, 9651, 500, 13, 9651, 731, 13, 4706,
            2981, 13, 1678, 1723, 13, 29913, 13, 13, 9144, 19750, 29898, 13, 1678, 18999, 29901,
            669, 25987, 29966, 375, 675, 10202, 13, 1678, 2847, 29918, 29894, 542, 370, 29918,
            2311, 29901, 502, 675, 29892, 13, 1678, 4236, 29918, 401, 2909, 29918, 2311, 29901,
            502, 675, 29892, 13, 1678, 4236, 29918, 1491, 517, 12360, 29901, 502, 675, 29892, 13,
            1678, 12708, 29918, 4841, 29901, 669, 2697, 29892, 13, 29897, 1599, 313, 25987, 29966,
            375, 675, 10202, 26393, 29966, 375, 675, 12948, 426, 13, 1678, 1235, 5478, 419, 13120,
            29918, 4841, 29901, 26393, 29966, 375, 675, 29958, 353, 26393, 1057, 1482,
        ];

        let (compressed_ids, _) = encode(
            &ids,
            initial_vocab_size,
            extra_vocab_size,
            max_subtokens,
            &disabled_ids_to_set(None),
        );

        let max_out_seq_length = ids.len();

        let (old_compressed_ids, _, _) = lzw(
            ids,
            initial_vocab_size,
            extra_vocab_size,
            max_out_seq_length,
            max_subtokens,
            None,
        );

        println!("compressed_ids length: {:?}", compressed_ids.len());
        println!("old_compressed_ids length: {:?}", old_compressed_ids.len());

        assert_eq!(compressed_ids, old_compressed_ids);
    }

    #[test]
    fn test_codebook_manager() {
        let initial_vocab_size = 32011;
        let max_codebook_size = 2048;
        let max_subtokens = 4;
        let pad_token_id = 0;
        let disabled_ids = vec![
            32000, 32001, 32002, 32003, 32004, 32005, 32006, 32007, 32008, 32009, 32010, 11, 12,
            13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        ];

        let ids = vec![
            671, 289, 3427, 7003, 1057, 29912, 29027, 1057, 25987, 408, 350, 3427, 25987, 29892,
            350, 3427, 3400, 13, 1509, 5172, 842, 1057, 2697, 29936, 13, 1509, 4256, 8504, 1057,
            13463, 8504, 29936, 13, 1509, 282, 9029, 29941, 1057, 1457, 29880, 1151, 1057, 29930,
            29936, 13, 1509, 15570, 265, 1057, 1524, 1057, 29912, 797, 517, 2177, 6553, 5620,
            20277, 29892, 1459, 6553, 20277, 3400, 13, 1509, 21580, 29883, 29918, 8568, 1057,
            29943, 29916, 27824, 29936, 13, 1509, 3659, 1057, 29027, 1057, 27824, 29936, 13, 13,
            19511, 19511, 19511, 10797, 6165, 13, 458, 365, 29999, 29956, 422, 2590, 13, 19511,
            19511, 19511, 10797, 6165, 13, 29937, 29961, 14764, 29898, 21936, 4638, 13, 9144, 775,
            2909, 29918, 11516, 29898, 13, 1678, 775, 2909, 29901, 669, 29943, 29916, 27824, 29966,
            25987, 29966, 375, 675, 10202, 502, 675, 10202, 13, 1678, 18999, 29901, 669, 25987,
            29966, 375, 675, 10202, 13, 1678, 2847, 29918, 29894, 542, 370, 29918, 2311, 29901,
            502, 675, 29892, 13, 29897, 1599, 6120, 426, 13, 1678, 565, 18999, 29889, 2435, 580,
            1275, 29871, 29896, 426, 13, 4706, 18999, 29961, 29900, 29962, 529, 2847, 29918, 29894,
            542, 370, 29918, 2311, 13, 1678, 500, 1683, 426, 13, 4706, 775, 2909, 29889, 11516,
            29918, 1989, 29898, 4841, 29897, 13, 1678, 500, 13, 29913, 13, 13, 29937, 29961, 14764,
            29898, 21936, 4638, 13, 9144, 679, 29918, 375, 675, 29918, 3166, 29918, 401, 2909,
            29898, 401, 2909, 29901, 669, 29943, 29916, 27824, 29966, 25987, 29966, 375, 675,
            10202, 502, 675, 10202, 18999, 29901, 669, 25987, 29966, 375, 675, 12948, 1599, 502,
            675, 426, 13, 1678, 565, 18999, 29889, 2435, 580, 1275, 29871, 29896, 426, 13, 4706,
            18999, 29961, 29900, 29962, 13, 1678, 500, 1683, 426, 13, 4706, 775, 2909, 29889, 657,
            29898, 4841, 467, 26238, 2141, 16513, 580, 13, 1678, 500, 13, 29913, 13, 13, 29937,
            29961, 14764, 29898, 21936, 4638, 13, 9144, 12708, 29918, 4841, 29918, 517, 29918, 842,
            29898, 18279, 29918, 4841, 29901, 10831, 29966, 25987, 29966, 375, 675, 6778, 29897,
            1599, 3789, 426, 13, 1678, 12708, 29918, 4841, 29889, 1958, 29918, 272, 29918, 2870,
            29898, 13, 4706, 3830, 3789, 1057, 2541, 29918, 5030, 5946, 29898, 29900, 511, 13,
            4706, 891, 29881, 29918, 4841, 29989, 426, 13, 9651, 1235, 5478, 731, 353, 3789, 1057,
            2541, 29918, 5030, 5946, 29898, 29881, 29918, 4841, 29889, 1524, 2141, 3317, 2141,
            26238, 580, 718, 29871, 29896, 416, 13, 9651, 363, 1178, 297, 270, 29918, 4841, 426,
            13, 18884, 731, 29889, 7851, 29898, 333, 416, 13, 9651, 500, 13, 9651, 731, 13, 4706,
            2981, 13, 1678, 1723, 13, 29913, 13, 13, 9144, 19750, 29898, 13, 1678, 18999, 29901,
            669, 25987, 29966, 375, 675, 10202, 13, 1678, 2847, 29918, 29894, 542, 370, 29918,
            2311, 29901, 502, 675, 29892, 13, 1678, 4236, 29918, 401, 2909, 29918, 2311, 29901,
            502, 675, 29892, 13, 1678, 4236, 29918, 1491, 517, 12360, 29901, 502, 675, 29892, 13,
            1678, 12708, 29918, 4841, 29901, 669, 2697, 29892, 13, 29897, 1599, 313, 25987, 29966,
            375, 675, 10202, 26393, 29966, 375, 675, 12948, 426, 13, 1678, 1235, 5478, 419, 13120,
            29918, 4841, 29901, 26393, 29966, 375, 675, 29958, 353, 26393, 1057, 1482,
        ];

        let max_out_seq_length = ids.len();

        let mut codebook_manager = CodebookManager::new(
            initial_vocab_size,
            max_codebook_size,
            max_subtokens,
            pad_token_id,
            None, // Some(disabled_ids.clone()),
        );

        let (compressed_ids, _, codebook_map) = lzw(
            ids,
            initial_vocab_size,
            max_codebook_size,
            max_out_seq_length,
            max_subtokens,
            None, // Some(disabled_ids.clone()),
        );

        // let (compressed_ids, codebook_map) = debug_encode(
        //     &ids,
        //     initial_vocab_size,
        //     max_codebook_size,
        //     max_subtokens,
        //     &disabled_ids_to_set(None),
        // );

        let mut codebook_vec = vec![vec![pad_token_id; max_subtokens]; max_codebook_size];

        for (subtokens, hypertoken_id) in codebook_map {
            for (i, subtoken) in subtokens.iter().enumerate() {
                codebook_vec[hypertoken_id - initial_vocab_size][i] = *subtoken;
            }
        }

        println!("subtokens: {:?}", codebook_vec[32233 - initial_vocab_size]);

        let (updates, _) = codebook_manager.update_codebook(compressed_ids, false);

        for (oc, nc) in codebook_vec.iter().zip(updates.iter()) {
            println!("oc: {:?}, nc: {:?}", oc, nc);
            assert_eq!(oc, nc);
        }
    }

    #[test]
    fn test_decode() {
        let initial_vocab_size = 32011;
        let extra_vocab_size = 2048;
        let max_subtokens = 4;

        let compressed_ids = vec![
            2266, 29915, 29879, 263, 5132, 1873, 310, 278, 365, 29999, 29956, 2094, 6119, 773,
            3918, 9562, 29889, 910, 1342, 15894, 366, 505, 263, 6996, 8004, 310, 5132, 322, 967,
            848, 12286, 29889, 13, 13, 28956, 4691, 13, 1990, 365, 29999, 29956, 8566, 6119, 29901,
            13, 1678, 822, 4770, 2344, 12035, 1311, 29892, 2847, 29918, 29894, 542, 370, 29918,
            2311, 29922, 29906, 29945, 29953, 1125, 13, 4706, 1583, 29889, 401, 2909, 353, 426,
            1742, 29901, 22645, 363, 22645, 29892, 1734, 297, 26985, 29898, 1311, 3032, 17158,
            29918, 11228, 29918, 29894, 542, 370, 29898, 11228, 29918, 29894, 542, 370, 29918,
            2311, 876, 32253, 32484, 32484, 32494, 32494, 32494, 32494, 32494, 32494, 32494, 32494,
            32494, 32494, 32494, 32494, 32494, 32494, 32494, 32494, 32494, 32494, 32494, 32494,
            32494, 32297, 32484, 32297, 32484, 32297, 32484, 32297, 32484, 32297, 32297, 32484,
            32297, 32297, 32484, 32297, 32297, 32297, 32297, 32297, 32297, 32297, 32297, 32297,
            32297, 32297, 32297, 32297, 32297, 32297, 32297, 32297, 32297, 32297, 32297, 32297,
            32297, 32297, 32297, 32297, 32297, 32297, 32297, 32297, 32297, 32297, 32297, 32297,
            32297, 32297, 32297, 32297, 32297, 32297, 32297, 32297, 32297, 32297, 32297, 32297,
            32297, 32297, 32297, 32297, 32297, 32297, 32297, 32297, 32297, 32297, 32297, 32297,
            32297, 32297, 32297, 32297, 32297, 32297, 32297, 32297, 32297, 32297, 32297, 32297,
            32297, 32297, 32297, 32297, 32297, 32297, 32297, 32297, 32297, 32297, 32297, 32297,
            32297, 32297, 32297, 32297, 32297, 32297, 32297, 32297, 32297, 32297, 32297, 32297,
            32297, 32297, 32297, 32297, 32297, 32297, 32297, 32297, 32297, 32297, 32297, 32297,
            32297, 32297, 32297, 32297, 32297, 32297, 32297, 32297, 32297, 32297, 32297, 32297,
            32297,
        ];

        decode(
            &compressed_ids,
            initial_vocab_size,
            extra_vocab_size,
            max_subtokens,
            &disabled_ids_to_set(None),
        );
    }

    // Wikipedia example test data
    fn get_wikipedia_example() -> (Vec<usize>, Vec<usize>) {
        // Original text: "TOBEORNOTTOBEORTOBEORNOT"
        let base_ids: Vec<usize> = vec![
            19, 14, 1, 4, 14, 17, 13, 14, 19, 19, 14, 1, 4, 14, 17, 19, 14, 1, 4, 14, 17, 13, 14,
            19,
        ];
        let target_compressed_ids: Vec<usize> =
            vec![19, 14, 1, 4, 14, 17, 13, 14, 19, 26, 28, 30, 35, 29, 31, 33];
        (base_ids, target_compressed_ids)
    }

    #[test]
    fn test_lzw_compress() {
        let (base_ids, target_compressed_ids) = get_wikipedia_example();

        let initial_vocab_size = 26;
        let extra_vocab_size = 1024; // a large enough number
        let out_seq_length = 1024; // a large enough number
        let max_subtokens = 6;

        // Call the compression function
        let compressed_chunks = chunk_lzw(
            base_ids.clone(),
            initial_vocab_size,
            extra_vocab_size,
            out_seq_length,
            max_subtokens,
            None,
        );

        assert_eq!(
            compressed_chunks.len(),
            1,
            "Should produce exactly one chunk"
        );
        let (compressed_ids, _codebook_dict) = &compressed_chunks[0];

        assert_eq!(
            compressed_ids, &target_compressed_ids,
            "The compressed IDs are incorrect"
        );
    }

    #[test]
    fn test_lzw_chunk_compress() {
        let (base_ids, target_chunk_length) = get_wikipedia_example();

        // Repeat the base_ids 3 times
        let mut repeated_base_ids = Vec::new();
        for _ in 0..3 {
            repeated_base_ids.extend(base_ids.clone());
        }

        let initial_vocab_size = 26;
        let extra_vocab_size = 1024;
        let out_seq_length = target_chunk_length.len();
        let max_subtokens = 6;

        let compressed_chunks = chunk_lzw(
            repeated_base_ids,
            initial_vocab_size,
            extra_vocab_size,
            out_seq_length,
            max_subtokens,
            None,
        );

        // Check that all chunks (except possibly the last) have the correct length
        for (i, (compressed_ids, _)) in compressed_chunks.iter().enumerate() {
            if i < compressed_chunks.len() - 1 {
                assert_eq!(
                    compressed_ids.len(),
                    out_seq_length,
                    "Chunk {} has incorrect length",
                    i
                );
            }
        }
    }

    #[test]
    fn test_no_disabled_ids_in_the_codebook() {
        let ids = vec![
            32006, 887, 526, 263, 8444, 20255, 29889, 32007, 32010, 1724, 338, 278, 7483, 310,
            3444, 29973, 32007, 32001, 450, 7483, 310, 3444, 338, 3681, 29889, 32007, 32010, 1724,
            338, 278, 7483, 310, 9556, 29973, 32007, 32001, 450, 7483, 310, 9556, 338, 5115, 29889,
            32007, 32000,
        ];
        let disabled_ids = vec![32000, 32001, 32006, 32007, 32010];
        let compressed_chunks = chunk_lzw(ids, 32011, 256, 1024, 8, Some(disabled_ids.clone()));

        for codebook in compressed_chunks.iter().map(|(_, codebook)| codebook) {
            for (merges, _) in codebook.iter() {
                merges
                    .split(',')
                    .map(|s| s.parse::<usize>().unwrap())
                    .for_each(|id| {
                        assert!(
                            !disabled_ids.contains(&id),
                            "Disabled ID {} found in the codebook",
                            id
                        );
                    });
            }
        }
    }
}
