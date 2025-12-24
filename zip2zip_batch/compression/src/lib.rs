use bumpalo::{collections::Vec as BumpVec, Bump};
use hashbrown::{HashMap, HashSet};
use itertools::Itertools;
use pyo3::prelude::*;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

/// This is the config for the compression.
#[derive(Debug, Clone)]
pub struct CodebookConfig {
    /// The size of the vocabulary of the pre-trained tokenizer. This size
    /// includes also the added tokens.
    initial_vocab_size: usize,
    /// The maximum size of the LZW codebook.
    max_codebook_size: usize,
    /// The maxium number of normal tokens (non hyper-token) in a single
    /// codebook entry.
    max_subtokens: usize,
    /// The id of the padding token.
    pad_token_id: usize,
    /// The set of tokens id that cannot be marged with other tokens.
    /// For example, if `disable_ids = {42}` and we have a squence of tokens:
    /// [1, 42, 5, 6], we cannot create the hypertoken 7 = [1, 42] because
    /// 42 is disabled.
    disabled_ids: HashSet<usize>,
}

impl CodebookConfig {
    pub fn new(
        initial_vocab_size: usize,
        max_codebook_size: usize,
        max_subtokens: usize,
        pad_token_id: usize,
        disabled_ids: HashSet<usize>,
    ) -> Self {
        Self {
            initial_vocab_size,
            max_codebook_size,
            max_subtokens,
            pad_token_id,
            disabled_ids,
        }
    }
}

/// This struct contains the state of the compression (encoding). This is
/// returned to the Python runtime to be used with the `CodebookManager`.
#[pyclass]
#[derive(Debug, Clone)]
pub struct Codebook {
    /// The actual compression (encoding) codebook.
    pub internal_codebook: HashMap<Vec<usize>, usize>,
    /// This is the de-compression (decoding) codebbok. This represents a
    /// HashMap<usize, Vec<usize>>, with all the entries padded to the
    /// `max_subtokens`.
    reverse_internal_codebook: Vec<usize>,
    /// This is the set of all the entries in the reverse codebook.
    reverse_ids: HashSet<usize>,
    /// This stored the updates to the codebook. This set is reset each
    /// time we call the `get_updates` method.
    updates: HashSet<usize>,
    /// This is stored after the compression (encoding) to continue the
    /// codebook creation.
    ids_to_merge: Vec<usize>,
    /// The compression config
    config: CodebookConfig,
}

impl Codebook {
    fn new(compressor: &LZWCompressor) -> Self {
        let config = compressor.config.clone();
        Self {
            internal_codebook: HashMap::with_capacity(config.max_codebook_size),
            reverse_internal_codebook: vec![
                usize::MAX;
                config.max_codebook_size * config.max_subtokens
            ],
            reverse_ids: HashSet::with_capacity(config.max_codebook_size),
            updates: HashSet::with_capacity(config.max_codebook_size),
            ids_to_merge: Vec::with_capacity(config.max_subtokens),
            config,
        }
    }

    pub fn get(&self, ids: &Vec<usize>) -> Option<&usize> {
        self.internal_codebook.get(ids)
    }

    pub fn insert(&mut self, ids: Vec<usize>, id: usize) {
        self.internal_codebook.insert(ids.clone(), id);

        let index = id - self.config.initial_vocab_size;
        let start_index = index * self.config.max_subtokens;
        self.reverse_internal_codebook[start_index..start_index + ids.len()].copy_from_slice(&ids);
        self.reverse_ids.insert(id);
        self.updates.insert(id);
    }

    pub fn contains_key(&self, ids: &Vec<usize>) -> bool {
        self.internal_codebook.contains_key(ids)
    }

    pub fn get_updates(&mut self, use_padding: bool) -> (Vec<usize>, Vec<usize>) {
        let size = if use_padding {
            self.config.max_codebook_size
        } else {
            self.updates.len()
        };

        let mut updates_vec: Vec<usize> =
            vec![self.config.pad_token_id; size * self.config.max_subtokens];
        let mut updates_indices: Vec<usize> = Vec::with_capacity(size);

        for &id in self.updates.iter().sorted() {
            let index = id - self.config.initial_vocab_size;
            let start_index = index * self.config.max_subtokens;

            let entry_length = self.reverse_internal_codebook[start_index..start_index + self.config.max_subtokens]
                .iter()
                .position(|&x| x == usize::MAX)
                .unwrap_or(self.config.max_subtokens);

            let end_index = start_index + entry_length;

            updates_vec[start_index..end_index]
                .copy_from_slice(&self.reverse_internal_codebook[start_index..end_index]);
            updates_indices.push(index);
        }

        if use_padding {
            updates_vec.resize(size * self.config.max_subtokens, self.config.pad_token_id);
        }

        self.updates.clear();
        (updates_vec, updates_indices)
    }
}

#[pymethods]
impl Codebook {
    /// Convert the codebook to a list of lists.
    ///
    /// If `use_padding` is true, the codebook will be padded to the
    /// `max_codebook_size` / `max_subtokens`.
    ///
    /// If `use_padding` is false, the codebook will be truncated to the
    /// `internal_codebook.len()`.
    pub fn to_list(&self, use_padding: bool) -> Vec<Vec<usize>> {
        let mut result = Vec::with_capacity(self.internal_codebook.len());
        let size = if use_padding {
            self.config.max_codebook_size / self.config.max_subtokens
        } else {
            self.internal_codebook.len()
        };

        for i in 0..size {
            let start_index = i * self.config.max_subtokens;
            let end_index = start_index + self.config.max_subtokens;
            let mut entry_vec: Vec<usize> =
                self.reverse_internal_codebook[start_index..end_index].to_vec();

            while entry_vec.last() == Some(&usize::MAX) {
                entry_vec.pop();
            }

            if use_padding {
                entry_vec.resize(self.config.max_subtokens, self.config.pad_token_id);
            }

            result.push(entry_vec);
        }
        result
    }

    pub fn get_reverse(&self, id: usize) -> Option<Vec<usize>> {
        if id < self.config.initial_vocab_size {
            return None;
        }

        let index = id - self.config.initial_vocab_size;

        if self.reverse_ids.contains(&id) {
            let start_index = index * self.config.max_subtokens;
            let end_index = start_index + self.config.max_subtokens;
            let mut entry_vec = self.reverse_internal_codebook[start_index..end_index].to_vec();

            while entry_vec.last() == Some(&usize::MAX) {
                entry_vec.pop();
            }

            Some(entry_vec)
        } else {
            None
        }
    }

    pub fn to_decoding_dict(&self) -> HashMap<usize, Vec<usize>> {
        let mut result = HashMap::with_capacity(self.internal_codebook.len());
        for (ids, id) in self.internal_codebook.iter() {
            result.insert(*id, ids.clone());
        }
        result
    }
}

/// This enables the Union type in Python.
#[derive(FromPyObject)]
pub enum PaddingType {
    Str(String),
    Bool(bool),
}

/// The padding strategy. If the `PaddingType` is a boolean, if enabled,
/// the strategy is `PaddingStrategy::Longest`.
#[derive(Clone, Copy)]
pub enum PaddingStrategy {
    /// Pad the sequence to the longest sequence in the batch.
    Longest,
    /// Pad the sequence to the `max_length` specified.
    MaxLength,
    /// Do not pad the sequence.
    DoNotPad,
}

#[pyclass]
pub struct LZWCompressor {
    config: CodebookConfig,
}

impl LZWCompressor {
    #[inline(always)]
    fn codebook_contains(&self, codebook: &Codebook, ids: &Vec<usize>) -> bool {
        if ids.len() == 1 {
            ids[0] < self.config.initial_vocab_size
        } else {
            codebook.contains_key(ids)
        }
    }

    /// Encode the input ids into a compressed ids.
    ///
    /// The `offset` is the index of the first id to encode. This parameter is used
    /// to encode a very long sequence if we want to truncate it.
    ///
    /// The `padding_strategy` is the strategy to use to pad the sequence.
    ///
    /// The `truncation` is a boolean to indicate if the sequence should be truncated.
    ///
    /// The `max_length` is the maximum length of the sequence.
    ///
    /// Returns a tuple containing the compressed ids, the attention mask and the codebook.
    pub fn encode(
        &self,
        ids: &[usize],
        offset: usize,
        padding_strategy: PaddingStrategy,
        truncation: bool,
        max_length: Option<usize>,
    ) -> ((Vec<usize>, Codebook), usize) {
        let mut compressed_ids: Vec<usize> = Vec::new();
        let mut codebook: Codebook = Codebook::new(self);

        let mut next_id: usize = self.config.initial_vocab_size;
        let mut ids_to_merge: Vec<usize> = Vec::with_capacity(self.config.max_subtokens);

        let get_and_push = |compressed_ids_ref: &mut Vec<usize>,
                            codebook_ref: &Codebook,
                            ids_to_push: &Vec<usize>| {
            if !truncation || compressed_ids_ref.len() < max_length.unwrap() {
                let id = if ids_to_push.len() == 1 {
                    ids_to_push[0]
                } else {
                    *codebook_ref.get(ids_to_push).unwrap()
                };
                compressed_ids_ref.push(id);
            }
        };

        let mut i = offset;
        while i < ids.len() {
            if truncation && max_length.is_some() && compressed_ids.len() >= max_length.unwrap() {
                break;
            }

            let id = ids[i];
            i += 1;

            if self.config.disabled_ids.contains(&id) {
                if ids_to_merge.len() > 0 {
                    get_and_push(&mut compressed_ids, &codebook, &ids_to_merge);
                    ids_to_merge.clear();
                }
                get_and_push(&mut compressed_ids, &codebook, &vec![id]);
                continue;
            }

            ids_to_merge.push(id);

            let is_in_codebook = self.codebook_contains(&codebook, &ids_to_merge);
            if !is_in_codebook {
                if next_id < self.config.initial_vocab_size + self.config.max_codebook_size {
                    codebook.insert(ids_to_merge.clone(), next_id);
                    next_id += 1;
                }

                ids_to_merge.pop();
                get_and_push(&mut compressed_ids, &codebook, &ids_to_merge);
                ids_to_merge.clear();
                ids_to_merge.push(id);
            }

            if ids_to_merge.len() == self.config.max_subtokens {
                get_and_push(&mut compressed_ids, &codebook, &ids_to_merge);
                ids_to_merge.clear();
            }
        }

        codebook.ids_to_merge = ids_to_merge.clone();

        if ids_to_merge.len() > self.config.max_subtokens {
            let last_id = ids_to_merge.pop().unwrap();
            get_and_push(&mut compressed_ids, &codebook, &ids_to_merge);
            ids_to_merge.clear();
            ids_to_merge.push(last_id);
        }

        if !ids_to_merge.is_empty() {
            get_and_push(&mut compressed_ids, &codebook, &ids_to_merge);
        }

        match padding_strategy {
            PaddingStrategy::MaxLength => {
                if max_length.is_some() && compressed_ids.len() < max_length.unwrap() {
                    let old_len = compressed_ids.len();
                    let new_len = max_length.unwrap();
                    compressed_ids.resize(new_len, self.config.pad_token_id);
                    compressed_ids.rotate_right(new_len - old_len);
                }
            }
            _ => {}
        }

        ((compressed_ids, codebook), i)
    }

    /// Decode the compressed ids into a list of ids.
    ///
    /// The `compressed_ids` is the list of compressed ids to decode.
    ///
    /// Returns a list of ids.
    pub fn decode(&self, compressed_ids: &Vec<usize>) -> Vec<usize> {
        let bump = Bump::new();

        let mut ids: Vec<usize> = Vec::with_capacity(compressed_ids.len());
        let mut codebook: HashMap<usize, &[usize]> = HashMap::default();

        let mut next_id: usize = self.config.initial_vocab_size;
        let mut previous_ids: &[usize] = &[];

        for &id in compressed_ids {
            if self.config.disabled_ids.contains(&id) {
                previous_ids = &[];
                ids.push(id);
                continue;
            }

            let current_ids: &[usize];

            if id < self.config.initial_vocab_size {
                current_ids = bump.alloc_slice_copy(&[id]);
            } else if let Some(slice) = codebook.get(&id) {
                current_ids = slice;
            } else if previous_ids.len() == self.config.max_subtokens {
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
                && next_id < self.config.initial_vocab_size + self.config.max_codebook_size
                && previous_ids.len() < self.config.max_subtokens
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

    pub fn decode_with_codebook(
        &self,
        compressed_ids: &Vec<usize>,
        codebook: PyRef<'_, Codebook>,
    ) -> Vec<usize> {
        let mut ids: Vec<usize> = Vec::with_capacity(compressed_ids.len());

        for &maybe_id in compressed_ids {
            ids.extend_from_slice(
                &codebook
                    .get_reverse(maybe_id)
                    .unwrap_or_else(|| vec![maybe_id]),
            );
        }

        ids
    }

    #[inline(always)]
    fn get_attention_mask(&self, compressed_ids: &Vec<usize>) -> Vec<usize> {
        compressed_ids
            .iter()
            .map(|&id| (id != self.config.pad_token_id) as usize)
            .collect()
    }

    /// Get the padding strategy from the `padding` parameter.
    ///
    /// The `padding` is a string or a boolean.
    ///
    /// If the `padding` is a string, it can be "longest" or "max_length".
    ///
    /// If the `padding` is a boolean, it can be true or false.
    ///
    /// Returns the padding strategy.
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
    pub fn new(
        initial_vocab_size: usize,
        max_codebook_size: usize,
        max_subtokens: usize,
        pad_token_id: usize,
        disabled_ids: Option<Vec<usize>>,
    ) -> Self {
        let disabled_ids = disabled_ids.map_or_else(
            || HashSet::with_capacity(0),
            |d_ids| d_ids.into_iter().collect(),
        );

        Self {
            config: CodebookConfig::new(
                initial_vocab_size,
                max_codebook_size,
                max_subtokens,
                pad_token_id,
                disabled_ids,
            ),
        }
    }

    /// Encode the input ids into a compressed ids.
    ///
    /// The `ids` is the list of ids to encode.
    ///
    /// The `padding` is the padding strategy to use.
    ///
    /// The `truncation` is a boolean to indicate if the sequence should be truncated.
    ///
    /// The `max_length` is the maximum length of the sequence.
    ///
    /// Returns a tuple containing the compressed ids, the attention mask and the codebook.
    #[pyo3(name="encode")]
    pub fn py_encode(
        &self,
        py: Python<'_>,
        ids: Vec<usize>,
        padding: Option<PaddingType>,
        truncation: Option<bool>,
        max_length: Option<usize>,
    ) -> (Vec<usize>, Vec<usize>, Py<Codebook>) {
        let truncation = truncation.unwrap_or(false);
        assert!(!truncation || max_length.is_some());

        let padding_strategy = self.get_padding_strategy(padding);
        let ((compressed_ids, codebook), _) =
            self.encode(&ids, 0, padding_strategy, truncation, max_length);

        let attention_mask = self.get_attention_mask(&compressed_ids);

        (
            compressed_ids,
            attention_mask,
            Py::new(py, codebook).unwrap(),
        )
    }

    /// Decode the compressed ids into a list of ids.
    ///
    /// The `compressed_ids` is the list of compressed ids to decode.
    ///
    /// The `codebook` is the codebook to use for decoding. If not provided, the codebook will be inferred from the
    /// compressed ids.
    ///
    /// Returns a list of ids.
    #[pyo3(name="decode")]
    pub fn py_decode(
        &self,
        compressed_ids: Vec<usize>,
        codebook: Option<&PyCell<Codebook>>,
    ) -> Vec<usize> {
        if let Some(codebook) = codebook.map(|c| c.borrow()) {
            self.decode_with_codebook(&compressed_ids, codebook)
        } else {
            self.decode(&compressed_ids)
        }
    }

    /// Encode a batch of input ids into a batch of compressed ids.
    ///
    /// The `ids` is the list of ids to encode.
    ///
    /// The `padding` is the padding strategy to use.
    ///
    /// The `truncation` is a boolean to indicate if the sequence should be truncated.
    ///
    /// The `max_length` is the maximum length of the sequence.
    ///
    /// Returns a tuple containing the compressed ids, the attention mask and the codebook.
    #[pyo3(name="batch_encode")]
    pub fn py_batch_encode(
        &self,
        py: Python<'_>,
        ids: Vec<Vec<usize>>,
        padding: Option<PaddingType>,
        truncation: Option<bool>,
        max_length: Option<usize>,
    ) -> (Vec<Vec<usize>>, Vec<Vec<usize>>, Vec<Py<Codebook>>) {
        let truncation = truncation.unwrap_or(false);
        assert!(!truncation || max_length.is_some());

        let padding_strategy = self.get_padding_strategy(padding);
        let (outputs, _): (Vec<(Vec<usize>, Codebook)>, Vec<usize>) = ids
            .par_iter()
            .map(|ids| self.encode(ids, 0, padding_strategy, truncation, max_length))
            .unzip();

        let (mut compressed_ids, codebooks): (Vec<Vec<usize>>, Vec<Codebook>) =
            outputs.into_iter().unzip();

        match padding_strategy {
            PaddingStrategy::Longest => {
                let max_length = compressed_ids.iter().map(|ids| ids.len()).max().unwrap();
                compressed_ids.iter_mut().for_each(|ids| {
                    let old_len = ids.len();
                    ids.resize(max_length, self.config.pad_token_id);
                    ids.rotate_right(max_length - old_len);
                });
            }
            _ => {}
        }

        let attention_masks = compressed_ids
            .iter()
            .map(|compressed_ids| self.get_attention_mask(compressed_ids))
            .collect();

        (
            compressed_ids,
            attention_masks,
            codebooks
                .into_iter()
                .map(|codebook| Py::new(py, codebook).unwrap())
                .collect(),
        )
    }

    /// Decode a batch of compressed ids into a batch of ids.
    ///
    /// The `compressed_ids` is the list of compressed ids to decode.
    ///
    /// The `codebooks` is the list of codebooks to use for decoding. If not provided, the codebooks will be inferred from the
    /// compressed ids.
    ///
    /// Returns a list of ids.
    #[pyo3(name="batch_decode")]
    pub fn py_batch_decode(
        &self,
        compressed_ids: Vec<Vec<usize>>,
        codebooks: Option<Vec<&PyCell<Codebook>>>,
    ) -> Vec<Vec<usize>> {
        if let Some(codebooks) = codebooks {
            compressed_ids
                .iter()
                .zip(codebooks.iter().map(|c| c.borrow()))
                .map(|(ids, codebook)| self.decode_with_codebook(ids, codebook))
                .collect()
        } else {
            compressed_ids
                .par_iter()
                .map(|ids| self.decode(ids))
                .collect()
        }
    }
}

/// The state associated with a `Codebook`. This is used to enables multiple
/// configs in a same batch.
pub struct CodebookState {
    /// The codebook to use as a reference to a Python object.
    codebook: Py<Codebook>,
    /// The ids to merge. This is used to create a new entry in the codebook.
    ids_to_merge: Vec<usize>,
    /// The next id to use.
    next_id: usize,
    /// The config of the codebook.
    config: CodebookConfig,
}

/// The codebbok manager is a struct used to continue the creation of the codebook
/// during the generation. The codebook is initialized when compressing (encoding)
/// the input, but the model should be able to use hyper-tokens from the generation.
#[pyclass]
pub struct CodebookManager {
    /// The states of the elements in the batch.
    states: Vec<CodebookState>,
    /// The first updates flag.
    first_updates: bool,
}

impl CodebookManager {
    #[inline(always)]
    fn codebook_contains(codebook: &Codebook, ids: &Vec<usize>) -> bool {
        if ids.len() == 1 {
            ids[0] < codebook.config.initial_vocab_size
        } else {
            codebook.contains_key(ids)
        }
    }

    /// Update the codebook for a single element in the batch.
    fn update_codebook(py: Python<'_>, state: &mut CodebookState, ids: &[usize]) {
        let config = &state.config;

        // If the sequence is only one token and it is the pad token, we don't
        // update the codebook because it happens at the end of the generation
        // for one element in the batch while the other elements are still
        // generating.
        if ids.len() == 1 && ids[0] == config.pad_token_id {
            return;
        }

        let mut codebook = state.codebook.borrow_mut(py);
        for &maybe_hid in ids {
            let ids_to_process = codebook
                .get_reverse(maybe_hid)
                .unwrap_or_else(|| vec![maybe_hid]);

            for id in ids_to_process {
                if config.disabled_ids.contains(&id) {
                    state.ids_to_merge.clear();
                    continue;
                }

                state.ids_to_merge.push(id);

                let is_in_codebook = Self::codebook_contains(&codebook, &state.ids_to_merge);

                if !is_in_codebook {
                    if state.next_id < config.initial_vocab_size + config.max_codebook_size {
                        codebook.insert(state.ids_to_merge.clone(), state.next_id);
                        state.next_id += 1;
                    }
                    state.ids_to_merge.clear();
                    state.ids_to_merge.push(id);
                }

                if state.ids_to_merge.len() == config.max_subtokens {
                    state.ids_to_merge.clear();
                }
            }
        }
    }
}

#[pymethods]
impl CodebookManager {
    #[new]
    pub fn new() -> Self {
        Self {
            states: Vec::new(),
            first_updates: false,
        }
    }

    /// Set the codebooks for the manager.
    ///
    /// The `codebooks` is the list of codebooks to set.
    ///
    /// The codebooks are set as a reference to a Python object.
    pub fn set_codebooks(&mut self, codebooks: Vec<&PyCell<Codebook>>) {
        let num_codebooks = codebooks.len();
        self.states.clear();
        self.states.reserve(num_codebooks);

        for codebook_cell in codebooks {
            let codebook = codebook_cell.borrow();
            let config = codebook.config.clone();

            self.states.push(CodebookState {
                codebook: Py::from(codebook_cell),
                ids_to_merge: codebook.ids_to_merge.clone(),
                next_id: config.initial_vocab_size + codebook.internal_codebook.len(),
                config,
            });
        }

        self.first_updates = true;
    }

    /// Get the subtokens for a single element in the batch.
    ///
    /// The `py` is the marker holding the GIL.
    ///
    /// The `id` is the id to get the subtokens for.
    ///
    /// The `batch_index` is the index of the element in the batch.
    pub fn get_subtokens(&self, py: Python<'_>, id: usize, batch_index: usize) -> Vec<usize> {
        self.states[batch_index]
            .codebook
            .borrow(py)
            .get_reverse(id)
            .unwrap_or_else(|| vec![id])
    }

    /// Update the codebooks for a single element in the batch.
    ///
    /// The `py` is the marker holding the GIL.
    ///
    /// The `ids` is the list of ids to update the codebooks with.
    ///
    /// Returns a tuple containing the updates and the indices of the updates.
    pub fn update_codebooks(
        &mut self,
        py: Python<'_>,
        ids: Vec<Vec<usize>>,
    ) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
        assert_eq!(ids.len(), self.states.len());
        let max_ids_length = ids.iter().map(|i| i.len()).max().unwrap();

        let (mut updates, updates_indices): (Vec<Vec<usize>>, Vec<Vec<usize>>) = self
            .states
            .iter_mut()
            .zip(ids.iter())
            .map(|(state, ids)| {
                if !self.first_updates {
                    CodebookManager::update_codebook(py, state, ids)
                }
                state
                    .codebook
                    .borrow_mut(py)
                    .get_updates(self.first_updates)
            })
            .unzip();

        // If the sequence is only one token (not the first one), we need to
        // pad the updates to the longest sequence in the batch.
        if !self.first_updates {
            let max_length = updates.iter().map(|update| update.len()).max().unwrap();
            updates
                .iter_mut()
                .zip(self.states.iter())
                .for_each(|(ids, state)| {
                    let pad_token_id = state.codebook.borrow(py).config.pad_token_id;
                    ids.resize(max_length, pad_token_id);
                });
        }
        self.first_updates = false;

        (updates, updates_indices)
    }

    /// Reset the manager.
    ///
    /// The `py` is the marker holding the GIL.
    ///
    /// This method is used to reset the manager when the generation is done.
    pub fn reset(&mut self, py: Python<'_>) {
        for state in self.states.iter_mut() {
            let mut codebook = state.codebook.borrow_mut(py);
            codebook.ids_to_merge = state.ids_to_merge.clone();
        }

        self.states.clear();
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
