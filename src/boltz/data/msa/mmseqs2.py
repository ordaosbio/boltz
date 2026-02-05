# From https://github.com/sokrypton/ColabFold/blob/main/colabfold/colabfold.py

import logging
import os
import random
import tarfile
import time
from shutil import move
from typing import Union

import requests
from tqdm import tqdm

from ordaos_structure.msa_client import RedisMSAClient

logger = logging.getLogger(__name__)

TQDM_BAR_FORMAT = (
    "{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]"
)


def run_mmseqs2(  # noqa: PLR0912, D103, C901, PLR0915
    x: Union[str, list[str]],
    prefix: str = "tmp",
    use_env: bool = True,
    use_filter: bool = False,
    use_pairing: bool = True,
    pairing_strategy: str = "greedy",
    host_url: str = "#",
) -> tuple[list[str], list[str]]:
    # submission_endpoint = "ticket/pair" if use_pairing else "ticket/msa"
    _ = host_url

    # Set header agent as boltz
    headers = {}
    headers["User-Agent"] = "boltz"

    rc = RedisMSAClient()

    def submit(seqs: list[str], mode: str, N: int = 101):
        n, query = N, ""
        for seq in seqs:
            query += f">{n}\n{seq}\n"
            n += 1

        #while True:
        error_count = 0
        try:
            # https://requests.readthedocs.io/en/latest/user/advanced/#advanced
            # "good practice to set connect timeouts to slightly larger than a multiple of 3"
            out = rc.submit(seqs, mode=mode)
        except requests.exceptions.Timeout:
            logger.warning("Timeout while submitting to MSA server. Retrying...")
        except Exception as e:
            error_count += 1
            logger.warning(
                f"Error while fetching result from MSA server. Retrying... ({error_count}/5)"
            )
            logger.warning(f"Error: {e}")
            time.sleep(5)
            if error_count > 5:
                raise
            return submit(seqs, mode=mode)


        return {
            "id": out,
            "status": rc.status(out) or "UNKNOWN"
        }

    def status(ID):
        out = rc.wait(ID)
        return out

    def download(ID, path):
        rc.download(ID, path)

    # process input x
    seqs = [x] if isinstance(x, str) else x

    # setup mode
    if use_filter:
        mode = "env" if use_env else "all"
    else:
        mode = "env-nofilter" if use_env else "nofilter"

    if use_pairing:
        mode = ""
        # greedy is default, complete was the previous behavior
        if pairing_strategy == "greedy":
            mode = "pairgreedy-nofilter"
        elif pairing_strategy == "complete":
            mode = "paircomplete-nofilter"
        if use_env:
            mode = mode + "-env"

    # define path
    path = f"{prefix}_{mode}"
    if not os.path.isdir(path):
        os.mkdir(path)

    # call mmseqs2 api
    tar_gz_file = f"{path}/out.tar.gz"
    N, REDO = 101, True

    # deduplicate and keep track of order
    seqs_unique = []
    # TODO this might be slow for large sets
    [seqs_unique.append(x) for x in seqs if x not in seqs_unique]
    Ms = [N + seqs_unique.index(seq) for seq in seqs]
    # lets do it!
    if not os.path.isfile(tar_gz_file):
        TIME_ESTIMATE = 150 * len(seqs_unique)
        out = {"status": "INIT"}
        with tqdm(total=TIME_ESTIMATE, bar_format=TQDM_BAR_FORMAT) as pbar:
            while REDO:
                pbar.set_description("SUBMIT")

                if out["status"] == "INIT":
                    out = submit(seqs_unique, mode, N)

                # Resubmit job until it goes through
                while out["status"] in ["UNKNOWN", "RATELIMIT"]:
                    sleep_time = 5 + random.randint(0, 5)
                    logger.error(f"Sleeping for {sleep_time}s. Reason: {out['status']}")
                    # resubmit
                    time.sleep(sleep_time)
                    if "id" in out:
                        out["status"] = status(out["id"])
                    else:
                        out = submit(seqs_unique, mode, N)


                if out["status"] == "ERROR":
                    msg = (
                        "MMseqs2 API is giving errors. Please confirm your "
                        " input is a valid protein sequence. If error persists, "
                        "please try again an hour later."
                    )
                    raise Exception(msg)

                if out["status"] == "MAINTENANCE":
                    msg = (
                        "MMseqs2 API is undergoing maintenance. "
                        "Please try again in a few minutes."
                    )
                    raise Exception(msg)

                # wait for job to finish
                ID, TIME = out["id"], 0
                pbar.set_description(out["status"])
                while out["status"] in ["UNKNOWN", "RUNNING", "PENDING"]:
                    t = 5 + random.randint(0, 5)
                    logger.error(f"Sleeping for {t}s. Reason: {out['status']}")
                    time.sleep(t)
                    pbar.set_description(out["status"])
                    if out["status"] == "RUNNING":
                        TIME += t
                        pbar.update(n=t)
                    out["status"] = status(out["id"])

                if out["status"] == "COMPLETE":
                    if TIME < TIME_ESTIMATE:
                        pbar.update(n=(TIME_ESTIMATE - TIME))
                    REDO = False
                    break

                if out["status"] == "ERROR":
                    REDO = False
                    break
                    msg = (
                        "MMseqs2 API is giving errors. Please confirm your "
                        " input is a valid protein sequence. If error persists, "
                        "please try again an hour later."
                    )
                    raise Exception(msg)

                logger.error(f"out_status unknown {out}")

                raise Exception(f"out_status unknown {out}")

            # Download results
            download(ID, tar_gz_file)

    # prep list of a3m files
    if use_pairing:
        a3m_files = [f"{path}/pair.a3m"]
    else:
        a3m_files = [f"{path}/final.a3m"]
        if use_env:
            a3m_files.append(f"{path}/bfd.mgnify30.metaeuk30.smag30.a3m")

    # extract a3m files
    if any(not os.path.isfile(a3m_file) for a3m_file in a3m_files):
        with tarfile.open(tar_gz_file) as tar_gz:
            tar_gz.extractall(path)

    if use_pairing:
        move(f"{path}/final.a3m", f"{path}/pair.a3m")

    # gather a3m lines
    a3m_lines = {}
    for a3m_file in a3m_files:
        update_M, M = True, None
        for line in open(a3m_file, "r"):
            if len(line) > 0:
                if "\x00" in line:
                    line = line.replace("\x00", "")
                    update_M = True
                if line.startswith(">") and update_M:
                    M = int(line[1:].rstrip())
                    update_M = False
                    if M not in a3m_lines:
                        a3m_lines[M] = []
                a3m_lines[M].append(line)

    a3m_lines = ["".join(a3m_lines[n]) for n in Ms]
    return a3m_lines
