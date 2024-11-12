import pytest
import os
import json
import bempp_cl


def joinall(*ls):
    if len(ls) == 1:
        return ls[0]
    return os.path.join(ls[0], joinall(*ls[1:]))


folder = joinall(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..")


def test_version_number_consistency():
    if not os.path.isfile(os.path.join(folder, "VERSION")):
        pytest.skip("Can only run this test from the source dir")

    with open(os.path.join(folder, "VERSION")) as f:
        v1 = f.read().strip()

    with open(os.path.join(folder, "codemeta.json")) as f:
        v2 = json.load(f)["version"]
        if v2.startswith("v"):
            v2 = v2[1:]

    v3 = bempp_cl.__version__

    assert v1 == v2 == v3
