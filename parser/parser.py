import os
from typing import Dict, List, Any, Optional
import time
import sys
import pickle
import multiprocessing

# allow us to import from this current directory
parser_dir: str = "/".join(__file__.split("/")[:-1])
sys.path.insert(1, parser_dir)

from utils import (
    process_UE4_string_to_value,
    convert_to_np,
    get_filename_from_path,
    extend_data,
)
import numpy as np

# used as the dictionary key when the data has no explicit title (ie. included as raw array)
_no_title_key: str = "data"
cache_dir: str = os.path.join(parser_dir, "cache")
os.makedirs(cache_dir, exist_ok=True)


def parse_row(
    data: Dict[str, Any],
    data_line: str,
    title: Optional[str] = "",
    t: Optional[int] = 0,
) -> None:
    # cleanup data line
    data_line: List[str] = data_line.replace("}", "{").replace("{", ",").split(",")

    if title == "":  # find title with the data line
        title: str = data_line[0].split(":")[0]
        # remove title for first elem
        data_line[0] = data_line[0].replace(f"{title}:", "")
    working_map: Dict[str, Any] = {} if title not in data else data[title]
    data[title] = working_map  # ensure this working set contributes to the larger set

    if t != 0:
        if (
            "t" not in working_map
        ):  # in case we need to also link the time associated with this
            working_map["t"] = []
        working_map["t"].append(t)
    subtitle: str = ""  # subtitle for elements within the dictionary
    for element in data_line:

        # case when empty string occurs (from "{" or "}" splitting)
        if element == "":
            subtitle = ""  # reset subtitle name
            continue

        # case when only element (after title) is present (eg. TimestampCarla)
        if ":" not in element:
            if _no_title_key not in working_map:
                working_map[_no_title_key] = []
            working_map[_no_title_key].append(process_UE4_string_to_value(element))
            continue

        # common case
        key_value: List[str] = element.split(":")
        key_value: List[str] = [elem for elem in key_value if len(elem) > 0]
        if len(key_value) == 1:  # contains only sublabel, like "COMBINED"
            subtitle = key_value[0]
        elif len(key_value) == 2:  # typical key:value pair
            key, value = key_value
            key = f"{subtitle}{key}"
            value = process_UE4_string_to_value(value)  # evaluate what we think this is

            # add to the working map
            if key not in working_map:
                working_map[key] = []
            working_map[key].append(value)
        else:
            raise NotImplementedError


def validate(data: Dict[str, Any], L: Optional[int] = None) -> None:
    # verify the data structure is reasonable
    if L is None:
        L: int = len(data["TimestampCarla"][_no_title_key])
    for k in data.keys():
        if k == "CustomActor":
            continue
        if isinstance(data[k], dict):
            validate(data[k], L)
        elif isinstance(data[k], list):
            assert len(data[k]) == L or len(data[k]) == L - 1
        else:
            raise NotImplementedError

    # ensure the custom actor data is also good
    if "CustomActor" in data:
        CA_lens = [len(x) for x in data["CustomActor"].values()]
        assert min(CA_lens) == max(CA_lens)  # all same lens


def read_chunk(path: str, pid: int, num_cores: int, debug=False) -> Dict[str, Any]:
    # these are the group types we are using for now
    DReyeVR_core: str = "[DReyeVR]"
    DReyeVR_CA: str = "[DReyeVR_CA]"
    data = {}
    num_lines: int = 0
    with open(path, "r") as f:  # read only, allows for concurrency
        num_lines = len(f.readlines())
    with open(path, "r") as f:  # read only, allows for concurrency
        start_line: int = pid * (num_lines // num_cores)
        end_line: int = min((pid + 1) * (num_lines // num_cores), num_lines)
        if num_cores > 1:
            time.sleep(pid * 0.01)  # so these prints all happen in line
            print(f"[P{pid}]\t Reading from [{start_line}:{end_line})", flush=True)
        start_t: float = time.time()
        for i, line in enumerate(f.readlines()):
            if start_line <= i < end_line:
                # remove leading spaces
                line = line.strip(" ")
                # checking the line(s) for core DReyeVR data
                if line[: len(DReyeVR_core)] == DReyeVR_core:
                    data_line: str = line.strip(DReyeVR_core).strip("\n")
                    parse_row(data, data_line)
                    if debug:
                        validate(data)

                # checking the line(s) for DReyeVR custom actor data
                elif line[: len(DReyeVR_CA)] == DReyeVR_CA:
                    data_line: str = line.strip(DReyeVR_CA).strip("\n")
                    t = data["TimestampCarla"][_no_title_key][-1]  # get carla time
                    parse_row(data, data_line, title="CustomActor", t=t)
                    if debug:
                        validate(data)

                # print status
                if num_cores <= 1 and i % 500 == 0:
                    t: float = time.time() - start_t
                    print(
                        f"Lines read: {i} \t {100 * i / num_lines:.3f}% @ {t:.3f}s",
                        end="\r",
                        flush=True,
                    )

    if num_cores > 1:
        filename = get_filename_from_path(path)
        cache_data(data, f"{filename}.{pid}")
    return data


def stitch_data(data: Dict[str, Any], num_cores: int, path: str) -> Dict[str, Any]:
    filename = get_filename_from_path(path)
    for pid in range(num_cores):
        data_pid = try_load_data(f"{filename}.{pid}")
        if len(data) == 0:
            data = data_pid
        else:
            data = extend_data(data, data_pid)
    return data


def parse_file(
    path: str,
    force_reload: Optional[bool] = False,
    debug: Optional[bool] = False,
    num_cores: Optional[int] = 4,
) -> Dict[str, np.ndarray or dict]:
    if force_reload is False:
        """try to load cached data"""
        data = try_load_data(path)
        if data is not None:
            return data

    # this function reads in a DReyeVR recording file and parses every line to return
    # a dictionary following the parser structure depending on the group types

    assert os.path.exists(path)
    print(f"Reading DReyeVR recording file: {path}")

    data: Dict[str, List[Any]] = {}
    start_t: float = time.time()

    if num_cores <= 1:
        data = read_chunk(path, 0, 1, False)
    else:
        jobs = []
        for pid in range(num_cores):
            j = multiprocessing.Process(
                target=read_chunk,
                args=(
                    path,
                    pid,
                    num_cores,
                    debug,
                ),
            )
            jobs.append(j)

        for j in jobs:
            j.start()

        for j in jobs:
            j.join()

        data = stitch_data(data, num_cores, path)
    n: int = len(data["TimestampCarla"][_no_title_key])
    print(f"successfully read {n} frames in {time.time() - start_t:.3f}s")

    # TODO: do everything in np from the get-go rather than convert at the end
    data = convert_to_np(data)
    cache_data(data, path)
    return data


def try_load_data(path: str) -> Optional[Dict[str, Any]]:
    actual_name = get_filename_from_path(path)
    filename = f"{os.path.join(cache_dir, actual_name)}.pkl"
    data = None
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
        print(f"Loaded data from {filename}")
    else:
        print(f"Did not find data at {filename}")
    return data


def cache_data(data: Dict[str, Any], path: str) -> None:
    actual_name: str = path.split("/")[-1].replace(".txt", "")
    os.makedirs(cache_dir, exist_ok=True)
    filename = f"{os.path.join(cache_dir, actual_name)}.pkl"
    with open(filename, "wb") as filehandler:
        pickle.dump(data, filehandler)
    print(f"cached data to {filename}")
