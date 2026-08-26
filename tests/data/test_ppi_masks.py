"""Regression tests for the PPI rec/lig token masks.

These pin the seven failure modes fixed relative to the original `ppi_affinity`
"initial bones" commit. They are deliberately dependency-light: the dtype and
astuple definitions are extracted textually so the test runs without rdkit.
"""

from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "boltz"


def _load_tokenv2():
    src = (SRC / "data" / "types.py").read_text()
    block = src[src.index("TokenV2 = [") :]
    block = block[: block.index("\n]") + 2]
    ns = {"np": np}
    exec(block, ns)  # noqa: S102
    return ns["TokenV2"]


def _load_tokendata():
    from dataclasses import dataclass
    from typing import Optional

    src = (SRC / "data" / "tokenize" / "boltz2.py").read_text()
    cls = src[src.index("@dataclass\nclass TokenData") : src.index("def token_astuple")]
    fn = src[src.index("def token_astuple") : src.index("def compute_frame")]
    ns = {"np": np, "dataclass": dataclass, "Optional": Optional}
    exec(cls + fn, ns)  # noqa: S102
    return ns["TokenData"], ns["token_astuple"]


BASE_KW = dict(
    token_idx=0, atom_idx=0, atom_num=1, res_idx=0, res_type=1, res_name="ALA",
    sym_id=0, asym_id=0, entity_id=0, mol_type=0, center_idx=0, disto_idx=0,
    center_coords=np.zeros(3, np.float32), disto_coords=np.zeros(3, np.float32),
    resolved_mask=True, disto_mask=True, modified=False,
    frame_rot=np.zeros(9, np.float32), frame_t=np.zeros(3, np.float32),
    frame_mask=1, cyclic_period=0,
)


def test_dtype_declares_ppi_fields():
    """The original branch read token_data['ppi_rec_mask'] without declaring it."""
    names = np.dtype(_load_tokenv2()).names
    assert "ppi_rec_mask" in names
    assert "ppi_lig_mask" in names


def test_record_size_stays_divisible_by_four():
    """types.py notes mol_type is i4 so the record divides by 4; two bool fields
    break that unless padded."""
    assert np.dtype(_load_tokenv2()).itemsize % 4 == 0


def test_mask_defaults_are_bools_not_tuples():
    """`affinity_mask: bool = False,` silently defaults to the tuple (False,)."""
    TokenData, _ = _load_tokendata()
    tok = TokenData(**BASE_KW)
    for name in ("affinity_mask", "ppi_rec_mask", "ppi_lig_mask"):
        assert getattr(tok, name) is False, f"{name} default is not a bool"


def test_astuple_width_matches_dtype():
    TokenData, token_astuple = _load_tokendata()
    dtype = np.dtype(_load_tokenv2())
    assert len(token_astuple(TokenData(**BASE_KW))) == len(dtype.names)


@pytest.mark.parametrize("rec,lig", [(True, False), (False, True), (False, False)])
def test_masks_roundtrip_through_structured_array(rec, lig):
    TokenData, token_astuple = _load_tokendata()
    tokv2 = _load_tokenv2()
    tok = TokenData(**BASE_KW, ppi_rec_mask=rec, ppi_lig_mask=lig)
    arr = np.array([token_astuple(tok)], dtype=tokv2)
    assert bool(arr["ppi_rec_mask"][0]) is rec
    assert bool(arr["ppi_lig_mask"][0]) is lig


def test_schema_initialises_ppi_names_before_use():
    """`ppi_rec` was assigned only on the PPI branch but read unconditionally,
    so every small-molecule `binder:` YAML raised NameError."""
    src = (SRC / "data" / "parse" / "schema.py").read_text()
    init = src.index("ppi_rec_name: Optional[str] = None")
    assert init < src.index('spec["rec"]')
    assert "from re import I" not in src


def test_featurizer_emits_both_masks():
    src = (SRC / "data" / "feature" / "featurizerv2.py").read_text()
    assert '"ppi_rec_token_mask": ppi_rec_mask' in src
    assert '"ppi_lig_token_mask": ppi_lig_mask' in src
