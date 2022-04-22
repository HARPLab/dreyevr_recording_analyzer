import pandas as pd
from typing import Dict, List, Any
from parser import parse_file
from utils import (
    check_for_periph_data,
    convert_to_df,
    split_along_subgroup,
    get_good_idxs,
)
from visualizer import plot_versus
import numpy as np
import argparse

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="DReyeVR recording parser")
    argparser.add_argument(
        "-f",
        "--file",
        metavar="P",
        default=None,
        type=str,
        help="path of the (human readable) recording file",
    )
    args = argparser.parse_args()
    filename: str = args.file
    if filename is None:
        print("Need to pass in the recording file")
        exit(1)

    """parse the file"""
    data: Dict[str, np.ndarray or dict] = parse_file(filename)

    # """append/generate periph data if available"""
    # # check for periph data
    # PeriphData = check_for_periph_data(data)
    # if PeriphData is not None:
    #     data["PeriphData"] = PeriphData

    # """convert to pandas df"""
    # # need to split along groups so all data lengths are the same
    # data_groups = split_along_subgroup(data, ["CustomActor"])
    # data_groups_df: List[pd.DataFrame] = [convert_to_df(x) for x in data_groups]

    t: np.ndarray = data["TimestampCarla"]["data"]

    """visualize some interesting data!"""
    pupil_L = data["EyeTracker"]["LEFTPupilDiameter"]
    # drop invalid data
    good_pupil_diam = lambda x: x > 0  # need positive diameter
    good_idxs = get_good_idxs(pupil_L, good_pupil_diam)
    pupil_L = pupil_L[good_idxs]
    time_L = t[good_idxs]

    plot_versus(
        data_x=time_L,
        name_x="Time",
        data_y=pupil_L,
        name_y="Left pupil diameter",
        units_y="mm",
        units_x="s",
        lines=True,
    )

    pupil_R = data["EyeTracker"]["RIGHTPupilDiameter"]
    good_idxs = get_good_idxs(pupil_R, good_pupil_diam)
    pupil_R = pupil_R[good_idxs]
    time_R = t[good_idxs]
    plot_versus(
        data_x=time_R,
        name_x="Time",
        data_y=pupil_R,
        name_y="Right pupil diameter",
        units_y="mm",
        units_x="s",
        lines=True,
    )
