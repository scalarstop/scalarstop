"""Tools for simulating dataclasses without having to specify all of the parameter types."""


class AttrDict(dict):
    """
    A dictionary subclass that allows accessing dictionary keys as parameters.

    This is useful for providing an interface similar to a
    :py:class:`~scalarstop.hyperparams.HyperparamsType` instance,
    but without having to know the keys and values ahead of time.

    ``AttrDict`` subclasses :py:class:`dict` so it can be serialized
    to JSON.

    Examples:

    >>> d = AttrDict(a=1, b=2)
    >>> d
    {'a': 1, 'b': 2}
    >>> d["a"]
    1
    >>> d.a
    1
    >>> d["b"]
    2
    >>> d.b
    2
    """

    def __getattribute__(self, key: str):
        """Check if the key is in the dictionary. Otherwise treat the key as an attribute."""
        try:
            return dict.__getitem__(self, key)
        except KeyError:
            return dict.__getattribute__(self, key)
