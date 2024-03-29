from typing import Callable, Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import time


def get_filename_from_path(path: str) -> str:
    # TODO: use something more platform independent?
    delim: str = "/" if "/" in path else "\\"  # windows uses \ for paths

    # actual_name: str = path.split(delim)[-1].replace(".txt", "")
    final_chunk_name = path.split(delim)[-1]
    actual_name = final_chunk_name[: final_chunk_name.find(".")]
    return actual_name


def get_good_idxs(arr: np.ndarray, criteria: Callable[[Any], bool]) -> np.ndarray:
    good_idxs = np.where(criteria(arr) == True)
    return good_idxs


def flatten_dict(
    data: Dict[str, Any], sep: Optional[str] = "_"
) -> Dict[str, List[Any]]:
    flat = {}
    for k in data.keys():
        if isinstance(data[k], dict):
            k_flat = flatten_dict(data[k])
            for kd in k_flat.keys():
                key: str = f"{k}{sep}{kd}"
                flat[key] = k_flat[kd]
        else:
            flat[k] = data[k]
    # check no lingering dictionaies
    for k in flat.keys():
        assert not isinstance(flat[k], dict)
    return flat


def split_along_subgroup(
    data: Dict[str, Any], subgroups: List[str]
) -> Tuple[Dict[str, Any]]:
    # splits the one large dict along subgroups such as DReyeVR core and custom-actor data
    # TODO: make generalizable for arbitrary subgroups!
    ret = []
    for sg in subgroups:
        if sg in data.keys():
            ret.append({sg: data.pop(sg)})  # include the sub-data as its own dict
    ret.append(data)  # include all other data as its own "default" group
    return tuple(ret)


def process_UE4_string_to_value(value: str) -> Any:
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
        ret = [
            process_UE4_string_to_value(elem) for elem in raw_data[1::2]
        ]  # every *other* odd
    else:
        try:
            ret = eval(value)
        except Exception:
            ret = value
    return ret


def convert_to_np(data: Dict[str, Any]) -> Dict[str, np.ndarray or dict]:
    np_data = {}
    for k in data.keys():
        if isinstance(data[k], dict):
            np_data[k] = convert_to_np(data[k])
        else:
            np_data[k] = np.array(data[k])
    return np_data


def convert_standalone_dict_to_list(
    data: Dict[str, Any], standalone_key: str
) -> Dict[str, Any]:
    # converts a "standalone" dictionary as follows:
    # if dict == {standalone_key : array} converts to list(array) for simplicity
    clean_data = {}
    for k in data.keys():
        if k == standalone_key:
            # should only happen with raw array data (ex. TimestampCarla)
            assert len(data) == 1
            return list(data[k])
        else:
            if isinstance(data[k], dict):
                clean_data[k] = convert_standalone_dict_to_list(data[k], standalone_key)
            else:
                clean_data[k] = data[k]
    return clean_data


def convert_to_list(data: Dict[str, np.ndarray or dict]) -> Dict[str, list or dict]:
    list_data = {}
    for k in data.keys():
        if isinstance(data[k], dict):
            list_data[k] = convert_to_list(data[k])
        else:
            assert isinstance(data[k], np.ndarray)
            list_data[k] = data[k].tolist()
    return list_data


def normalize(arr: np.ndarray, axis=1) -> np.ndarray:
    # normalize arr
    arr = (arr.T / (np.linalg.norm(arr, axis=axis))).T
    assert np.allclose(np.linalg.norm(arr, axis=axis), 1, 0.0001)
    return arr


def VectorFromRotator(arr: np.ndarray) -> np.ndarray:
    # implementing FRotator::Vector()
    n = len(arr)
    assert arr.shape == (n, 3)
    # note this FRotator holds degrees as (pitch, yaw, roll)
    pitch: np.ndarray = arr[:, 0]
    yaw: np.ndarray = arr[:, 1]
    roll: np.ndarray = arr[:, 2]  # not needed
    CP = np.cos(pitch)
    SP = np.sin(pitch)
    CY = np.cos(yaw)
    SY = np.sin(yaw)
    vec = np.array([CP * CY, CP * SY, SP]).T
    assert vec.shape == (n, 3)
    return vec


def RotateVector(vec: np.ndarray, rot: np.ndarray) -> np.ndarray:
    # implementing FRotator::RotateVector()
    # https://docs.unrealengine.com/4.27/en-US/API/Runtime/Core/Math/FRotator/RotateVector/
    n = len(vec)
    assert vec.shape == (n, 3)
    assert rot.shape == (n, 3)  # rotator is in degrees (pitch, yaw, roll)
    # rotmat = np.zeros((n, 3, 3)) # creating rotation matrices
    pitch: np.ndarray = rot[:, 0]
    yaw: np.ndarray = rot[:, 1]
    roll: np.ndarray = rot[:, 2]
    CP = np.cos(pitch)
    SP = np.sin(pitch)
    CY = np.cos(yaw)
    SY = np.sin(yaw)
    CR = np.cos(roll)
    SR = np.sin(roll)
    ZERO = np.zeros(n)
    ONE = np.ones(n)
    # create the big rotation matrix for each rotator in rot from euler angles
    # http://planning.cs.uiuc.edu/node102.html
    yaw_rotmat = np.array(
        [
            [CY, -SY, ZERO],  # this is
            [SY, CY, ZERO],  # the yaw (counterclockwise)
            [ZERO, ZERO, ONE],  # rottation matrix
        ]
    )
    yaw_rotmat = np.moveaxis(yaw_rotmat, 2, 0)
    assert yaw_rotmat.shape == (n, 3, 3)
    pitch_rotmat = np.array(
        [
            [CP, ZERO, SP],  # this is
            [ZERO, ONE, ZERO],  # the pitch (counterclockwise)
            [-SP, ZERO, CP],  # rotation matrix
        ]
    )
    pitch_rotmat = np.moveaxis(pitch_rotmat, 2, 0)
    assert pitch_rotmat.shape == (n, 3, 3)
    roll_rotmat = np.array(
        [
            [ONE, ZERO, ZERO],  # this is
            [ZERO, CR, -SR],  # the roll (counterclockwise)
            [ZERO, SR, CR],  # rotation matrix
        ]
    )
    roll_rotmat = np.moveaxis(roll_rotmat, 2, 0)
    assert roll_rotmat.shape == (n, 3, 3)
    rotmat = np.matmul(yaw_rotmat, np.matmul(pitch_rotmat, roll_rotmat))
    # apply the rotation matrices to the vector
    rotated = np.array([np.matmul(rotmat[i], vec[i]) for i in range(n)])
    assert rotated.shape == (n, 3)
    return rotated


def check_for_periph_data(data: Dict[str, Any]) -> Optional[Dict[str, np.ndarray]]:
    # first determine if we have a legacy periph recording or not
    assert "UserInputs" in data
    start_t: float = time.time()
    periph_keys: List[str] = [
        "gaze2target_pitch",
        "gaze2target_yaw",
        "head2target_pitch",
        "head2target_yaw",
        "LightOn",
        "ButtonPressed",
    ]

    PeriphData: Dict[str, np.ndarray] = None

    if "gaze2target_pitch" in data["UserInputs"]:
        # using legacy periph, extract them from UserInputs
        PeriphData = {k: data["UserInputs"].pop(k) for k in periph_keys}
    elif "CustomActor" in data and "PeriphTarget" in data["CustomActor"]["Name"]:
        # using updated periph system, extract from implicit representation
        PeriphData: Dict[str, Any] = {}

        # need to only extract the Custom Actor data when it is a PeriphTarget
        t = len(data["TimestampCarla"])  # all the frames
        Visibility: np.ndarray = np.zeros(t, dtype=np.int32)
        PeriphOnlyDataIdxs: List[int] = []
        for i, CA_t in enumerate(data["CustomActor"]["t"]):
            if data["CustomActor"]["Name"][i] == "PeriphTarget":
                # t = data["CustomActor"]["t"][i]
                idx = np.searchsorted(data["TimestampCarla"], CA_t)
                Visibility[idx] = 1  # effectively same as "LightOn"
                PeriphOnlyDataIdxs.append(i)

        # then, get only the periph data's fields
        PeriphOnlyData = data["CustomActor"]["Location"][PeriphOnlyDataIdxs]
        assert np.sum(Visibility) == len(PeriphOnlyData)

        # then, extrapolate to the entire length t by inserting to the larger array
        extrapolated_periphtarget_location: np.ndarray = np.zeros((t, 3))

        # first, need to extend the periph target locations
        PO_data_idx: int = 0
        for i in range(len(Visibility)):
            if Visibility[i] == 1:
                extrapolated_periphtarget_location[i] = PeriphOnlyData[PO_data_idx]
                PO_data_idx += 1
        assert PO_data_idx == len(PeriphOnlyData)

        # now we can use the raw data from the rest of the recording
        assert (
            extrapolated_periphtarget_location.shape
            == data["EgoVariables"]["CameraLocAbs"].shape
        )
        RotVecDirection: np.ndarray = normalize(
            extrapolated_periphtarget_location - data["EgoVariables"]["CameraLocAbs"]
        )

        # correct COMBINEDGazeDir for invalid (zerovector) gaze vectors
        # HACK: just replace ZeroVector's with (1, 0, 0)

        CombinedEyeGaze = data["EyeTracker"]["COMBINEDGazeDir"]
        bad_idxs = np.where(
            (CombinedEyeGaze[:, 0] == 0)
            & (CombinedEyeGaze[:, 1] == 0)
            & (CombinedEyeGaze[:, 2] == 0)
        )
        CombinedEyeGaze[bad_idxs] = np.array([1, 0, 0])  # HACK, but works since invalid

        GazeDir = normalize(
            RotateVector(
                vec=CombinedEyeGaze,
                rot=data["EgoVariables"]["CameraRotAbs"],
            )
        )

        gaze_p, gaze_y = get_angles(GazeDir, RotVecDirection)
        PeriphData["gaze2target_pitch"] = gaze_p
        PeriphData["gaze2target_yaw"] = gaze_y

        HeadVec = normalize(VectorFromRotator(data["EgoVariables"]["CameraRotAbs"]))

        head_p, head_y = get_angles(HeadVec, RotVecDirection)
        PeriphData["head2target_pitch"] = head_p
        PeriphData["head2target_yaw"] = head_y
        assert gaze_p.shape == gaze_y.shape == head_p.shape == head_y.shape

        PeriphData["LightOn"] = Visibility

        # compute all the real data from implicit representation
        PeriphData["ButtonPressed"] = (  # logical or for turn signals
            data["UserInputs"]["TurnSignalLeft"] | data["UserInputs"]["TurnSignalRight"]
        )

    else:
        # no periph in this recording
        pass
    if PeriphData is not None:
        # ensure validation
        lens: List[int] = [len(x) for x in PeriphData.values()]
        assert max(lens) == min(lens)
        print(f"gathered periph data in {time.time() - start_t:.3f}s")
    return PeriphData


def get_angles(dir1: np.ndarray, dir2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    #  Rotating Vectors back to world coordinate frame to get angles pitch and yaw
    dir1_x0 = dir1[:, 0]
    dir1_y0 = dir1[:, 1]
    dir1_z0 = dir1[:, 2]
    dir2_x0 = dir2[:, 0]
    dir2_y0 = dir2[:, 1]
    dir2_z0 = dir2[:, 2]
    # dir1_x0 = dir1[0]
    # dir1_y0 = dir1[1]
    # dir1_z0 = dir1[2]
    # dir2_x0 = dir2[0]
    # dir2_y0 = dir2[1]
    # dir2_z0 = dir2[2]

    # Multiplying dir1 by Rotation Z Matrix
    dir1Gaze_yaw = np.arctan2(dir1_y0, dir1_x0)  # rotation about z axis
    dir1_x1 = dir1_x0 * np.cos(dir1Gaze_yaw) + dir1_y0 * np.sin(dir1Gaze_yaw)
    dir1_z1 = dir1_z0

    # Multiplying dir1 by Rotation Y Matrix
    dir1Gaze_pitch = np.arctan2(dir1_z1, dir1_x1)  # rotation about y axis

    # Multiplying dir2 by Rotation Z Matrix
    dir2_x1 = dir2_x0 * np.cos(dir1Gaze_yaw) + dir2_y0 * np.sin(dir1Gaze_yaw)
    dir2_y1 = -dir2_x0 * np.sin(dir1Gaze_yaw) + dir2_y0 * np.cos(dir1Gaze_yaw)
    dir2_z1 = dir2_z0

    # Multiplying dir2 by Rotation Y Matrix
    dir2_x2 = dir2_x1 * np.cos(dir1Gaze_pitch) + dir2_z1 * np.sin(dir1Gaze_pitch)
    dir2_y2 = dir2_y1
    dir2_z2 = -dir2_x1 * np.sin(dir1Gaze_pitch) + dir2_z1 * np.cos(dir1Gaze_pitch)

    # Get Yaw
    yaw = np.arctan2(dir2_y2, dir2_x2)

    # Get Pitch
    pitch = np.arctan2(dir2_z2, dir2_x2)

    return (pitch, yaw)


def convert_to_df(data: Dict[str, Any]) -> pd.DataFrame:
    start_t: float = time.time()
    data = convert_to_list(data)
    data = flatten_dict(data)
    data = _rename_to_match_downstream(data)
    lens = [len(x) for x in data.values()]
    assert min(lens) == max(lens)  # all lengths are equal!
    # NOTE: pandas can't haneld high dimensional np arrays, so we just use lists
    df = pd.DataFrame.from_dict(data)
    print(f"created DReyeVR df in {time.time() - start_t:.3f}s")
    return df


def _rename_to_match_downstream(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    if "CustomActor_t" in data:
        return data  # do nothing
    new_data: Dict[str, List[Any]] = {}
    for k in data.keys():
        new_key: str = k
        if "EyeTracker_" in new_key:
            new_key = new_key.replace("EyeTracker_", "")
            groups: List[str] = ["COMBINED", "LEFT", "RIGHT"]
            for g in groups:
                if g in new_key:
                    var_name = new_key.replace(g, "")
                    new_key = f"{var_name}_{g}"
        elif "PeriphData" in new_key:
            new_key = new_key.replace("PeriphData_", "")
        elif "UserInputs" in new_key:
            new_key = new_key.replace("UserInputs_", "")
        elif "EgoVariables" in new_key:
            new_key = new_key.replace("EgoVariables_", "")
        else:
            pass  # no key in og df yet
        new_data[new_key] = data[k]
    assert len(new_data) == len(data)
    return new_data


def compute_YP(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # using arr as a matrix of direction vectors
    n: int = len(arr)
    assert arr.shape == (n, 3)  # only supports 3D vectors
    arr_norm = normalize(arr)

    xs = arr_norm[:, 0]
    ys = arr_norm[:, 1]
    zs = arr_norm[:, 2]

    gaze_pitches = np.arctan2(zs, xs) * 180 / np.pi
    gaze_yaws = np.arctan2(ys, xs) * 180 / np.pi
    return gaze_yaws, gaze_pitches
