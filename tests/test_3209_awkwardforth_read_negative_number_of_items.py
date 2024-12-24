# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import platform
import sys

import numpy as np
import pytest

import awkward as ak

IS_WASM = sys.platform == "emscripten" or platform.machine in ["wasm32", "wasm64"]


pytestmark = pytest.mark.skipif(
    IS_WASM,
    reason="32-bit WASM does not yet comply with ForthMachine32",
)


def test_read_negative_number_of_items():
    vm = ak.forth.ForthMachine32("input source -5 source #q-> stack")
    vm.run({"source": np.array([1, 2, 3, 4, 5], dtype=np.int64)})
    assert vm.stack == []

    vm = ak.forth.ForthMachine32("input source output sink float64 -5 source #q-> sink")
    vm.run({"source": np.array([1, 2, 3, 4, 5], dtype=np.int64)})
    assert vm.output("sink").tolist() == []


def test_read_negative_and_positive_number_of_items():
    vm = ak.forth.ForthMachine32(
        "input source -5 source #q-> stack 5 source #q-> stack"
    )
    vm.run({"source": np.array([1, 2, 3, 4, 5], dtype=np.int64)})
    assert vm.stack == [1, 2, 3, 4, 5]

    vm = ak.forth.ForthMachine32(
        "input source output sink float64 -5 source #q-> sink 5 source #q-> sink"
    )
    vm.run({"source": np.array([1, 2, 3, 4, 5], dtype=np.int64)})
    assert vm.output("sink").tolist() == [1, 2, 3, 4, 5]


def test_read_positive_and_negative_number_of_items():
    vm = ak.forth.ForthMachine32(
        "input source 5 source #q-> stack -5 source #q-> stack"
    )
    vm.run({"source": np.array([1, 2, 3, 4, 5], dtype=np.int64)})
    assert vm.stack == [1, 2, 3, 4, 5]

    vm = ak.forth.ForthMachine32(
        "input source output sink float64 5 source #q-> sink -5 source #q-> sink"
    )
    vm.run({"source": np.array([1, 2, 3, 4, 5], dtype=np.int64)})
    assert vm.output("sink").tolist() == [1, 2, 3, 4, 5]
