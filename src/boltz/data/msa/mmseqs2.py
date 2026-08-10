# Originally from https://github.com/sokrypton/ColabFold/blob/main/colabfold/colabfold.py,
# then adapted onto Ordaos' internal Redis-queue MSA service. Now calls the Info MSA pipeline
# API (ordaosbio/engine#2013) via ordaos_structure.msa instead -- see
# ordaosbio/structure#630/#631 for the equivalent migration on the caller side.
import logging
from typing import Union

from ordaos_structure.msa import get_or_wait

logger = logging.getLogger(__name__)


def run_mmseqs2(  # noqa: D103
    x: Union[str, list[str]],
    prefix: str = "tmp",  # noqa: ARG001 -- was a local cache dir for the old tarball download
    use_env: bool = True,  # noqa: ARG001 -- Info's cache params always include the env db; see
    # ordaosbio/structure#631 for why there's currently no way to explicitly opt out
    use_filter: bool = False,  # noqa: ARG001 -- no equivalent knob in the Info API
    use_pairing: bool = True,
    pairing_strategy: str = "greedy",  # noqa: ARG001 -- Info's pair_mode has no strategy knob
    host_url: str = "#",  # noqa: ARG001 -- unused; kept for call-site compatibility
) -> list[str]:
    """Fetch MSAs for `x` from the Info MSA pipeline API, blocking until ready.

    Preserves the original function's contract exactly (same signature, same return shape: one
    a3m-formatted text block per entry in `x`, in `x`'s order with duplicates repeated for
    homomers) so `boltz.main.compute_msa` -- which splits the result back into
    ``key,sequence`` rows itself -- needs no changes.
    """
    seqs = [x] if isinstance(x, str) else x

    # Distinct chains in first-seen order -- what a query is keyed by; homomer duplicates in
    # `seqs` all resolve to the one lookup and share its result below.
    seqs_unique: list[str] = []
    for seq in seqs:
        if seq not in seqs_unique:
            seqs_unique.append(seq)

    pair_mode = "paired" if use_pairing else "unpaired"
    canonical = get_or_wait(seqs_unique, pair_mode=pair_mode)

    blocks: list[str] = []
    for chain in canonical.chains:
        if use_pairing:
            # Info's paired_rows are (key, sequence) tuples: key==0 is this chain's own query
            # row, key>=1 are cross-chain pairing indices (shared across chains), key==-1 is
            # unpaired carry-over -- excluded here, this is the paired-only fetch. compute_msa
            # only reads alternating (header, sequence) lines and assigns its own key from row
            # *position*, so sorting by key before emitting preserves the original pairing.
            rows = sorted((r for r in (chain.paired_rows or []) if r[0] >= 0), key=lambda r: r[0])
            block = "\n".join(f">{key}\n{seq}" for key, seq in rows)
            blocks.append(block + "\n" if block else ">0\n\n")
        else:
            # Already a raw a3m block in the same alternating header/sequence shape.
            blocks.append(chain.unpaired_a3m or ">0\n\n")

    return [blocks[seqs_unique.index(seq)] for seq in seqs]
