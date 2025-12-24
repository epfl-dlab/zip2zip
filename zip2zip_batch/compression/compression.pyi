from typing import Dict, List, Optional, Tuple, Union

class Codebook:
    """
    Class for storing the LZW codebook.
    """

    def to_list(self, use_padding: bool) -> List[List[int]]:
        """
        Getter for the codebook as a list of lists.

        Args:
            use_padding: Whether to use padding. If True, the codebook will be padded with the pad_token_id
            up to the maximum length of the codebook. Each entry in the codebook will also be padded with
            the pad_token_id up to the maximum number of subtokens per entry.

        Returns:
            The codebook as a list of lists.
        """
        ...
    def to_decoding_dict(self) -> Dict[int, List[int]]:
        """
        Get the decoding dictionary for the codebook.
        """
        ...

class LZWCompressor:
    """
    Class for compressing and decompressing sequences of tokens.
    """

    def __init__(
        self,
        initial_vocab_size: int,
        max_codebook_size: int,
        max_subtokens: int,
        pad_token_id: int,
        disabled_ids: Optional[List[int]] = None,
    ) -> None:
        """
        Initialize the LZWCompressor.

        Args:
            initial_vocab_size: The initial size of the vocabulary.
            max_codebook_size: The maximum size of the codebook.
            max_subtokens: The maximum number of subtokens per entry.
            pad_token_id: The token id to use for padding.
            disabled_ids: A list of token ids to disable.
        """
        ...
    def encode(
        self,
        ids: List[int],
        padding: Union[str, bool] = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
    ) -> Tuple[List[int], List[int], Codebook]:
        """
        Encode (or compress) a sequence of tokens.

        Args:
            ids: The sequence of tokens to encode.
            padding: The padding strategy to use. Can be a string (one of "longest", "max_length", "do_not_pad")
            or a boolean (True or False). Default is False.
            truncation: Whether to truncate the sequence if it is too long. Default is False.
            max_length: The maximum length of the sequence. Default is None.

        Returns:
            A tuple containing the encoded sequence, the attention mask, and the codebook.
        """
        ...
    def decode(
        self, compressed_ids: List[int], codebook: Optional[Codebook] = None
    ) -> List[int]:
        """
        Decode (or decompress) a sequence of compressed tokens.

        Args:
            compressed_ids: The sequence of compressed tokens to decode.
            codebook: The codebook to use for decoding. If not provided, the codebook will be inferred from the
            compressed ids.

        Returns:
            The decoded sequence of tokens.
        """
        ...
    def batch_encode(
        self,
        ids: List[List[int]],
        padding: Union[str, bool],
        truncation: bool,
        max_length: Optional[int],
    ) -> Tuple[List[List[int]], List[List[int]], List[Codebook]]:
        """
        Encode (or compress) a batch of sequences of tokens.

        Args:
            ids: The batch of sequences of tokens to encode.
            padding: The padding strategy to use. Can be a string (one of "longest", "max_length", "do_not_pad")
            or a boolean (True or False). Default is False.
            truncation: Whether to truncate the sequences if they are too long. Default is False.
            max_length: The maximum length of the sequences. Default is None.

        Returns:
            A tuple containing the encoded sequences, the attention masks, and the codebooks.
        """
        ...
    def batch_decode(
        self,
        compressed_ids: List[List[int]],
        codebooks: Optional[List[Codebook]] = None,
    ) -> List[List[int]]:
        """
        Decode (or decompress) a batch of sequences of compressed tokens.

        Args:
            compressed_ids: The batch of sequences of compressed tokens to decode.
            codebooks: The codebooks to use for decoding. If not provided, the codebooks will be inferred from the
            compressed ids.

        Returns:
            The decoded batch of sequences of tokens.
        """
        ...

class CodebookManager:
    """
    Class for managing the codebooks. During generation, the codebook manager will update the codebooks
    with the new tokens.
    """

    def __init__(self) -> None:
        """
        Initialize the CodebookManager.
        """
        ...
    def set_codebooks(self, codebooks: List[Codebook]) -> None:
        """
        Set the codebooks. This method should be used before starting generation.

        Args:
            codebooks: The list of codebooks to set.
        """
        ...
    def get_subtokens(self, id: int, batch_index: int) -> List[int]:
        """
        Get the subtokens for a given token id and batch index.

        Args:
            id: The token id.
            batch_index: The batch index.

        Returns:
            The subtokens for the given token id and batch index.
        """
        ...
    def update_codebooks(
        self, ids: List[List[int]]
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Update the codebooks with the new tokens.

        Args:
            ids: The new tokens to update the codebooks with.

        Returns:
            A tuple containing the new entries in the codebooks and the indices of the new entries.
        """
        ...
    def reset(self) -> None:
        """
        Reset the codebook manager.

        This method should be called when the generation is finished.
        """
        ...
