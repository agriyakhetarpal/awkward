# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numbers

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def type(array):
    """
    The high-level type of an `array` (many types supported, including all
    Awkward Arrays and Records) as #ak.types.Type objects.

    The high-level type ignores #layout differences like
    #ak.contents.ListArray versus #ak.contents.ListOffsetArray, but
    not differences like "regular-sized lists" (i.e.
    #ak.contents.RegularArray) versus "variable-sized lists" (i.e.
    #ak.contents.ListArray and similar).

    Types are rendered as [Datashape](https://datashape.readthedocs.io/)
    strings, which makes the same distinctions.

    For example,

        ak.Array([[{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [2, 2]}],
                  [],
                  [{"x": 3.3, "y": [3, 3, 3]}]])

    has type

        3 * var * {"x": float64, "y": var * int64}

    but

        ak.Array(np.arange(2*3*5).reshape(2, 3, 5))

    has type

        2 * 3 * 5 * int64

    Some cases, like heterogeneous data, require [extensions beyond the
    Datashape specification](https://github.com/blaze/datashape/issues/237).
    For example,

        ak.Array([1, "two", [3, 3, 3]])

    has type

        3 * union[int64, string, var * int64]

    but "union" is not a Datashape type-constructor. (Its syntax is
    similar to existing type-constructors, so it's a plausible addition
    to the language.)
    """
    with ak._util.OperationErrorContext(
        "ak.type",
        dict(array=array),
    ):
        return _impl(array)


def _impl(array):
    if array is None:
        return ak.types.UnknownType()

    elif isinstance(
        array,
        tuple(x.type for x in ak.types.numpytype._dtype_to_primitive_dict),
    ):
        return ak.types.NumpyType(
            ak.types.numpytype._dtype_to_primitive_dict[array.dtype]
        )

    elif isinstance(array, (bool, np.bool_)):
        return ak.types.NumpyType("bool")

    elif isinstance(array, numbers.Integral):
        return ak.types.NumpyType("int64")

    elif isinstance(array, numbers.Real):
        return ak.types.NumpyType("float64")

    elif isinstance(
        array,
        (
            ak.highlevel.Array,
            ak.highlevel.Record,
            ak.highlevel.ArrayBuilder,
        ),
    ):
        return array.type

    elif isinstance(array, np.ndarray):
        if len(array.shape) == 0:
            return _impl(array.reshape((1,))[0])
        else:
            try:
                out = ak.types.numpytype._dtype_to_primitive_dict[array.dtype.type]
            except KeyError as err:
                raise ak._util.error(
                    TypeError(
                        "numpy array type is unrecognized by awkward: %r"
                        % array.dtype.type
                    )
                ) from err
            out = ak.types.NumpyType(out)
            for x in array.shape[-1:0:-1]:
                out = ak.types.RegularType(out, x)
            return ak.types.ArrayType(out, array.shape[0])

    elif isinstance(array, ak._ext.ArrayBuilder):
        form = ak.forms.from_json(array.form())
        return ak.types.ArrayType(form.type_from_behavior(None), len(array))

    elif isinstance(array, ak.record.Record):
        return array.array.form.type

    elif isinstance(array, ak.contents.Content):
        return array.form.type

    else:
        raise ak._util.error(TypeError(f"unrecognized array type: {array!r}"))