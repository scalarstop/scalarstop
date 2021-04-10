"""Utilities for naming things."""

import hashlib
import json
import math
import uuid
from typing import Any, Mapping, Optional

DEFAULT_HASH_ID_ALPHABET = "123456789abcdefghijkmnopqrstuvwxyz"


def hash_id(
    params: Mapping[str, Any],
    *,
    alphabet: str = DEFAULT_HASH_ID_ALPHABET,
    length: Optional[int] = None,
) -> str:
    """
    Create a short hash-like ID of any JSON-serializable data structure.

    Args:
        params: A JSON-serializable object (dictionaries, lists, etc.)
            that you want to compute a unique ID for.

        alphabet: The character alphabet to make IDs out of.
            By default, this defaults to the lowercase ASCII alphbet
            characters and the numbers ``1`` to ``9``. The number ``0`` is
            removed to avoid confusion with the letter ``o``.

        length: The chosen length of the hash ID. This defaults to a
            length that should make the ID approximately as random as
            a UUID. This default length depends on the number
            of unique characters in the alphabet.

    Returns:
        The hash ID as a string.

    """
    # We calculate how long the ID needs to be
    # in order for it to be as unique as a UUID.
    # However, because we're using hashing rather than a random
    # number generator, we have to consider that the hashing
    # algorithm could have a higher rate of collisions than a UUID.
    if length is None:
        length = math.ceil(math.log(2 ** 122, len(alphabet)))
    json_string = json.dumps(
        params,
        ensure_ascii=True,
        allow_nan=False,
        indent=None,
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(json_string).digest()
    return decode_bytes_to_alphabet(digest, alphabet=alphabet, length=length)


def decode_bytes_to_alphabet(
    digest: bytes, *, alphabet: str = DEFAULT_HASH_ID_ALPHABET, length: int
) -> str:
    """
    Decode an arbitrary byte string into our alphabet of choice.

    Args:

        alphabet: The character alphabet to make IDs out of.
            By default, this defaults to the lowercase ASCII alphbet
            characters and the numbers ``1`` to ``9``. The number ``0`` is
            removed to avoid confusion with the letter ``o``.

        length: The chosen length of the hash ID. This defaults to a
            length that should make the ID approximately as random as
            a UUID. This default length depends on the number
            of unique characters in the alphabet.

    """
    acc = int.from_bytes(digest, byteorder="big") % len(alphabet) ** length
    string = ""
    while acc:
        acc, idx = divmod(acc, len(alphabet))
        string = alphabet[idx : idx + 1] + string
    return string


def temporary_filename(base_filename: str) -> str:
    """
    Append a random string to an existing filename to create a temporary filename.

    This is different from making a temporary filename on /tmp or somewhere else.
    Our goal is to create a temporary file that is guaranteed to be on the sam
    filesystem as wherever ``base_filename`` is.
    """
    return "--".join(
        (base_filename, decode_bytes_to_alphabet(uuid.uuid4().bytes, length=10))
    )
