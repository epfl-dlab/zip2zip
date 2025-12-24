use bumpalo::{collections::Vec as BumpVec, Bump};
use fastset::Set;
use pyo3::prelude::*;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use rustc_hash::FxHashMap;

#[pyclass]
#[pyo3(get_all)]
#[derive(Debug, FromPyObject)]
struct Codebook {
    codebook: Vec<usize>,
    codebook_size: usize,
    initial_vocab_size: usize,
    max_codebook_size: usize,
    max_subtokens: usize,
    pad_token_id: usize,
    updates: Vec<usize>,
}

impl Codebook {
    fn new(compressor: &LZWCompressor, codebook: &FxHashMap<Vec<usize>, usize>) -> Self {
        let mut codebook_vec: Vec<usize> =
            vec![compressor.pad_token_id; compressor.max_codebook_size * compressor.max_subtokens];
        let codebook_size = codebook.len();

        for (k, &v) in codebook {
            let ov = v - compressor.initial_vocab_size;

            let start_index = ov * compressor.max_subtokens;
            let end_index = start_index + k.len();

            codebook_vec[start_index..end_index].copy_from_slice(k);
        }

        Self {
            codebook: codebook_vec,
            codebook_size,
            initial_vocab_size: compressor.initial_vocab_size,
            max_codebook_size: compressor.max_codebook_size,
            max_subtokens: compressor.max_subtokens,
            pad_token_id: compressor.pad_token_id,
            updates: Vec::new(),
        }
    }

    fn get(&self, id: usize) -> Option<Vec<usize>> {
        let index = id - self.initial_vocab_size;

        if 0 < index && index < self.codebook_size {
            Some(
                self.codebook[index * self.max_subtokens..(index + 1) * self.max_subtokens]
                    .to_vec(),
            )
        } else {
            None
        }
    }

    fn insert(&mut self, id: usize, ids: Vec<usize>) {
        let index = id - self.initial_vocab_size;

        if 0 < index && index < self.codebook_size {
            self.codebook[index * self.max_subtokens..(index + 1) * self.max_subtokens]
                .copy_from_slice(&ids);

            let mut padded_ids = ids.clone();
            padded_ids.resize(self.max_subtokens, self.pad_token_id);
            self.updates.extend_from_slice(&padded_ids);
        }
    }

    fn get_updates(&mut self, first_update: bool) -> (Vec<usize>, usize) {
        let (updates, num_updates) = if first_update {
            (self.codebook.clone(), self.codebook_size)
        } else {
            (
                self.updates.clone(),
                self.updates.len() / self.max_subtokens,
            )
        };

        self.updates.clear();
        (updates, num_updates)
    }
}

#[pymethods]
impl Codebook {
    fn to_list(&self, use_padding: bool) -> Vec<Vec<usize>> {
        let mut result = Vec::with_capacity(self.codebook_size);
        let size = if use_padding {
            self.codebook.len() / self.max_subtokens
        } else {
            self.codebook_size
        };

        for i in 0..size {
            let start_index = i * self.max_subtokens;
            let end_index = start_index + self.max_subtokens;
            let mut entry_vec: Vec<usize> = self.codebook[start_index..end_index].to_vec();

            if !use_padding {
                while entry_vec.last() == Some(&self.pad_token_id) {
                    entry_vec.pop();
                }
            }

            result.push(entry_vec);
        }
        result
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

#[derive(FromPyObject)]
enum PaddingType {
    Str(String),
    Bool(bool),
}

#[derive(Clone, Copy)]
enum PaddingStrategy {
    Longest,
    MaxLength,
    DoNotPad,
}

#[pyclass]
struct LZWCompressor {
    initial_vocab_size: usize,
    max_codebook_size: usize,
    max_subtokens: usize,
    pad_token_id: usize,
    disabled_ids: Set,
}

impl LZWCompressor {
    #[inline(always)]
    fn codebook_contains(&self, codebook: &FxHashMap<Vec<usize>, usize>, ids: &Vec<usize>) -> bool {
        if ids.len() == 1 {
            ids[0] < self.initial_vocab_size
        } else {
            codebook.contains_key(ids)
        }
    }

    #[inline(always)]
    fn push_to_compressed_ids(
        &self,
        compressed_ids: &mut Vec<usize>,
        id: usize,
        truncation: bool,
        max_length: Option<usize>,
    ) {
        if !truncation || compressed_ids.len() < max_length.unwrap() {
            compressed_ids.push(id);
        }
    }

    fn inner_encode(
        &self,
        ids: &[usize],
        offset: usize,
        padding_strategy: PaddingStrategy,
        truncation: bool,
        max_length: Option<usize>,
    ) -> ((Vec<usize>, FxHashMap<Vec<usize>, usize>), usize) {
        let mut compressed_ids: Vec<usize> = Vec::new();
        let mut codebook: FxHashMap<Vec<usize>, usize> = FxHashMap::default();

        let mut next_id: usize = self.initial_vocab_size;
        let mut ids_to_merge: Vec<usize> = Vec::with_capacity(self.max_subtokens);

        let mut i = offset;
        while i < ids.len() && (!truncation || compressed_ids.len() < max_length.unwrap()) {
            let id = ids[i];
            i += 1;

            if self.disabled_ids.contains(&id) {
                if ids_to_merge.len() > 0 {
                    self.push_to_compressed_ids(
                        &mut compressed_ids,
                        get_usize_from_codebook(&codebook, &ids_to_merge),
                        truncation,
                        max_length,
                    );
                    ids_to_merge.clear();
                }
                self.push_to_compressed_ids(&mut compressed_ids, id, truncation, max_length);
                continue;
            }

            ids_to_merge.push(id);

            let is_in_codebook = self.codebook_contains(&codebook, &ids_to_merge);
            if !is_in_codebook {
                if next_id < self.initial_vocab_size + self.max_codebook_size {
                    codebook.insert(ids_to_merge.clone(), next_id);
                    next_id += 1;
                }

                ids_to_merge.pop();
                self.push_to_compressed_ids(
                    &mut compressed_ids,
                    get_usize_from_codebook(&codebook, &ids_to_merge),
                    truncation,
                    max_length,
                );
                ids_to_merge.clear();
                ids_to_merge.push(id);
            }

            if ids_to_merge.len() == self.max_subtokens {
                self.push_to_compressed_ids(
                    &mut compressed_ids,
                    get_usize_from_codebook(&codebook, &ids_to_merge),
                    truncation,
                    max_length,
                );
                ids_to_merge.clear();
            }
        }

        if ids_to_merge.len() > self.max_subtokens {
            let last_id = ids_to_merge.pop().unwrap();
            self.push_to_compressed_ids(
                &mut compressed_ids,
                get_usize_from_codebook(&codebook, &ids_to_merge),
                truncation,
                max_length,
            );
            ids_to_merge.clear();
            ids_to_merge.push(last_id);
        }

        if ids_to_merge.len() > 0 {
            self.push_to_compressed_ids(
                &mut compressed_ids,
                get_usize_from_codebook(&codebook, &ids_to_merge),
                truncation,
                max_length,
            );
        }

        match padding_strategy {
            PaddingStrategy::MaxLength => {
                if max_length.is_some() && compressed_ids.len() < max_length.unwrap() {
                    compressed_ids.resize(max_length.unwrap(), self.pad_token_id);
                }
            }
            _ => {}
        }

        ((compressed_ids, codebook), i)
    }

    fn inner_decode(&self, compressed_ids: &Vec<usize>) -> Vec<usize> {
        let bump = Bump::new();

        let mut ids: Vec<usize> = Vec::with_capacity(compressed_ids.len());
        let mut codebook: FxHashMap<usize, &[usize]> = FxHashMap::default();

        let mut next_id: usize = self.initial_vocab_size;
        let mut previous_ids: &[usize] = &[];

        for &id in compressed_ids {
            if self.disabled_ids.contains(&id) {
                previous_ids = &[];
                ids.push(id);
                continue;
            }

            let current_ids: &[usize];

            if id < self.initial_vocab_size {
                current_ids = bump.alloc_slice_copy(&[id]);
            } else if let Some(slice) = codebook.get(&id) {
                current_ids = slice;
            } else if previous_ids.len() == self.max_subtokens {
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
                && next_id < self.initial_vocab_size + self.max_codebook_size
                && previous_ids.len() < self.max_subtokens
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

    #[inline(always)]
    fn get_attention_mask(&self, compressed_ids: &Vec<usize>) -> Vec<usize> {
        compressed_ids
            .iter()
            .map(|&id| (id != self.pad_token_id) as usize)
            .collect()
    }

    #[inline(always)]
    fn get_padding_strategy(&self, padding: Option<PaddingType>) -> PaddingStrategy {
        if padding.is_none() {
            return PaddingStrategy::DoNotPad;
        }

        let padding = padding.unwrap();

        match padding {
            PaddingType::Str(padding_str) => {
                if padding_str == "longest" {
                    PaddingStrategy::Longest
                } else if padding_str == "max_length" {
                    PaddingStrategy::MaxLength
                } else {
                    PaddingStrategy::DoNotPad
                }
            }
            PaddingType::Bool(padding_bool) => {
                if padding_bool {
                    PaddingStrategy::Longest
                } else {
                    PaddingStrategy::DoNotPad
                }
            }
        }
    }
}

#[pymethods]
impl LZWCompressor {
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
        }
    }

    fn encode(
        &self,
        ids: Vec<usize>,
        padding: Option<PaddingType>,
        truncation: Option<bool>,
        max_length: Option<usize>,
    ) -> (Vec<usize>, Vec<usize>, Codebook) {
        let truncation = truncation.unwrap_or(false);
        assert!(!truncation || max_length.is_some());

        let padding_strategy = self.get_padding_strategy(padding);
        let ((compressed_ids, codebook), _) =
            self.inner_encode(&ids, 0, padding_strategy, truncation, max_length);

        let codebook = Codebook::new(&self, &codebook);
        let attention_mask = self.get_attention_mask(&compressed_ids);

        (compressed_ids, attention_mask, codebook)
    }

    fn decode(&self, compressed_ids: Vec<usize>) -> Vec<usize> {
        self.inner_decode(&compressed_ids)
    }

    fn batch_encode(
        &self,
        ids: Vec<Vec<usize>>,
        padding: Option<PaddingType>,
        truncation: Option<bool>,
        max_length: Option<usize>,
    ) -> (Vec<Vec<usize>>, Vec<Vec<usize>>, Vec<Codebook>) {
        let truncation = truncation.unwrap_or(false);
        assert!(!truncation || max_length.is_some());

        let padding_strategy = self.get_padding_strategy(padding);
        let (outputs, _): (Vec<(Vec<usize>, FxHashMap<Vec<usize>, usize>)>, Vec<usize>) = ids
            .par_iter()
            .map(|ids| self.inner_encode(ids, 0, padding_strategy, truncation, max_length))
            .unzip();

        let (mut compressed_ids, hashmap_codebooks): (
            Vec<Vec<usize>>,
            Vec<FxHashMap<Vec<usize>, usize>>,
        ) = outputs.into_iter().unzip();

        match padding_strategy {
            PaddingStrategy::Longest => {
                let max_length = compressed_ids.iter().map(|ids| ids.len()).max().unwrap();
                compressed_ids
                    .iter_mut()
                    .for_each(|ids| ids.resize(max_length, self.pad_token_id));
            }
            _ => {}
        }

        let codebooks = hashmap_codebooks
            .iter()
            .map(|codebook| Codebook::new(&self, codebook))
            .collect();
        let attention_masks = compressed_ids
            .iter()
            .map(|compressed_ids| self.get_attention_mask(compressed_ids))
            .collect();

        (compressed_ids, attention_masks, codebooks)
    }

    fn batch_decode(&self, compressed_ids: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
        compressed_ids
            .par_iter()
            .map(|ids| self.inner_decode(ids))
            .collect()
    }
}

#[pyclass]
pub struct CodebookManager {
    initial_vocab_size: usize,
    max_codebook_size: usize,
    max_subtokens: usize,
    pad_token_id: usize,
    disabled_ids: Set,
    codebooks: Vec<Codebook>,
    previous_ids: Vec<Vec<usize>>,
    next_ids: Vec<usize>,
    first_update: bool,
}

impl CodebookManager {
    fn inner_update_codebook(&mut self, ids: &[usize], batch_index: usize) -> (Vec<usize>, usize) {
        for &hyper_id in ids {
            for id in self.get_subtokens(hyper_id, batch_index) {
                if self.disabled_ids.contains(&id) {
                    self.previous_ids[batch_index].clear();
                    continue;
                }

                let mut current_ids: Vec<usize>;

                if id < self.initial_vocab_size {
                    current_ids = vec![id];
                } else if let Some(entry) = self.codebooks[batch_index].get(id) {
                    current_ids = entry;
                } else if self.previous_ids[batch_index].len() == self.max_subtokens {
                    current_ids = self.previous_ids[batch_index].clone();
                } else {
                    current_ids = self.previous_ids[batch_index].clone();
                    current_ids.push(self.previous_ids[batch_index][0]);

                    self.codebooks[batch_index].insert(id, current_ids.clone());
                }

                if !self.previous_ids[batch_index].is_empty()
                    && self.next_ids[batch_index] < self.initial_vocab_size + self.max_codebook_size
                    && self.previous_ids[batch_index].len() < self.max_subtokens
                {
                    self.previous_ids[batch_index].push(current_ids[0]);

                    self.codebooks[batch_index].insert(
                        self.next_ids[batch_index],
                        self.previous_ids[batch_index].clone(),
                    );
                }

                self.previous_ids[batch_index] = current_ids;
            }
        }

        self.codebooks[batch_index].get_updates(self.first_update)
    }
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
            codebooks: Vec::new(),
            previous_ids: Vec::new(),
            next_ids: Vec::new(),
            first_update: true,
        }
    }

    fn set_codebooks(&mut self, codebooks: Vec<Codebook>) {
        let num_codebooks = codebooks.len();
        self.codebooks = codebooks;
        self.next_ids = vec![self.initial_vocab_size; num_codebooks];
        self.previous_ids = vec![Vec::new(); num_codebooks];
        self.first_update = true;
    }

    fn get_subtokens(&self, id: usize, batch_index: usize) -> Vec<usize> {
        self.codebooks[batch_index]
            .get(id)
            .unwrap_or_else(|| vec![id])
            .to_vec()
    }

    fn update_codebooks(&mut self, ids: Vec<Vec<usize>>) -> (Vec<Vec<usize>>, Vec<usize>) {
        assert_eq!(ids.len(), self.codebooks.len());

        let (updates, num_updates) = if self.first_update {
            self.codebooks
                .iter_mut()
                .map(|codebook| codebook.get_updates(self.first_update))
                .unzip()
        } else {
            ids.iter()
                .enumerate()
                .map(|(batch_index, ids)| self.inner_update_codebook(ids, batch_index))
                .unzip()
        };
        self.first_update = false;

        (updates, num_updates)
    }

    fn reset(&mut self) {
        self.next_ids.clear();
        self.codebooks.clear();
        self.first_update = true;
        self.previous_ids.clear();
    }
}

#[pymodule]
fn compression(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Codebook>()?;
    m.add_class::<LZWCompressor>()?;
    m.add_class::<CodebookManager>()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
}
