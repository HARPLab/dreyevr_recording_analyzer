import os
from typing import Dict, List, Any, Optional
import time

filename: str = "/Users/gustavo/carla/carla.mac/PythonAPI/examples/gus.txt"

data: Dict[str, List[Any]] = {}


def process_value(value: str) -> Any:
    ret = value
    if (
        "X=" in value  # x
        or "Y=" in value  # y (or yaw)
        or "Z=" in value  # z
        or "P=" in value  # pitch
        or "R=" in value  # roll
    ):
        # coming from an FVector or FRotator
        raw_data = value.replace("=", " ").split(" ")
        ret = [process_value(elem) for elem in raw_data[1::2]]  # every *other* odd
    else:
        try:
            ret = eval(value)
        except Exception:
            ret = value
    return ret


def gather_data(
    data_line: str, title: Optional[str] = "", t: Optional[int] = 0
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
        if "t" not in working_map:
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
            if "data" not in working_map:
                working_map["data"] = []
            working_map["data"].append(process_value(element))
            continue

        # common case
        key_value: List[str] = element.split(":")
        key_value: List[str] = [elem for elem in key_value if len(elem) > 0]
        if len(key_value) == 1:  # contains only sublabel, like "COMBINED"
            subtitle = key_value[0]
        elif len(key_value) == 2:  # typical key:value pair
            key, value = key_value
            key = f"{subtitle}{key}"
            value = process_value(value)  # evaluate what we think this is

            # add to the working map
            if key not in working_map:
                working_map[key] = []
            working_map[key].append(value)


def validate(data: Dict[str, Any], L: Optional[int] = None) -> None:
    # verify the data structure is reasonable
    if L is None:
        L: int = len(data["TimestampCarla"]["data"])
    for k in data.keys():
        if k == "CustomActor":
            # TODO
            continue
        if isinstance(data[k], dict):
            validate(data[k], L)
        elif isinstance(data[k], list):
            assert len(data[k]) == L or len(data[k]) == L - 1
        else:
            assert False


def DReyeVR_parser(path: str) -> float:
    assert os.path.exists(path)
    print(f"Reading DReyeVR recording file: {path}")

    DReyeVR_core: str = "  [DReyeVR]"
    DReyeVR_CA: str = "  [DReyeVR_CA]"
    with open(path) as f:
        start_t: float = time.time()
        for i, line in enumerate(f.readlines()):

            # checking the line(s) for core DReyeVR data
            if line[: len(DReyeVR_core)] == DReyeVR_core:
                data_line: str = line.strip(DReyeVR_core).strip("\n")
                gather_data(data_line)
                validate(data)

            # checking the line(s) for DReyeVR custom actor data
            elif line[: len(DReyeVR_CA)] == DReyeVR_CA:
                data_line: str = line.strip(DReyeVR_CA).strip("\n")
                t = data["TimestampCarla"]["data"][-1]  # get carla time
                gather_data(data_line, title="CustomActor", t=t)

            # print status
            if i % 500 == 0:
                t: float = time.time() - start_t
                print(f"Lines read: {i} @ {t:.3f}s", end="\r", flush=True)
    return t


t = DReyeVR_parser(filename)

n: int = len(data["TimestampCarla"]["data"])
print(f"successfully read {n} frames in {t:.3f}s")
