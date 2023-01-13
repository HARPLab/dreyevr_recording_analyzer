from pathlib import Path
import cv2
import pickle as pkl
import PIL
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


#########################
# DReyeVR dataloader Fns
#########################

import sys
# sys.path.append("/scratch/abhijatb/Bosch22")
# sys.path.append("../Bosch22")
from dreyevr_parser.parser import parse_file
from typing import Dict, List, Any
from utils import (
    check_for_periph_data,
    convert_to_df,
    split_along_subgroup,
    get_good_idxs,
)
from visualizer import plot_versus
from parser_utils import GetGazeDeviationFromHead
from ibmm import EyeClassifier


def load_dreyevr_data_and_gazeevent_annotate(recordingtxt_path):
    df_new, og_df = load_dreyevr_dataframe(recordingtxt_path)
    og_df = add_gaze_event2df(og_df)
    return df_new, og_df
    
    
def load_dreyevr_dataframe(recordingtxt_path):
    '''
    load _dreyevr_dataframe
    '''
    data: Dict[str, np.ndarray or dict] = parse_file(str(recordingtxt_path))
    """append/generate periph data if available"""
    # check for periph data
    PeriphData = check_for_periph_data(data)
    if PeriphData is not None:
        data["PeriphData"] = PeriphData

    """convert to pandas df"""
    # need to split along groups so all data lengths are the same
    data_groups = split_along_subgroup(data, ["CustomActor"])
    data_groups_df: List[pd.DataFrame] = [convert_to_df(x) for x in data_groups]
    df_new = data_groups_df[-1] 
    
    CAs_present = data_groups_df[0].CustomActor_Name.unique() # 'PeriphCross', 'PeriphTarget'
    assert ('PeriphCross' in CAs_present and 'PeriphTarget' in CAs_present)

    CA_df = data_groups_df[0]
    CA_df.rename(columns={"CustomActor_t": "TimestampCarla"}, inplace=True)
    og_df = data_groups_df[1]
    
    return df_new, og_df, CA_df


def get_hits_and_misses(og_df, CA_df): 
    # find the indices where lights came on and went off
    lighton_rows = og_df["LightOn"].diff().fillna(0)==1
    lightoff_rows = og_df["LightOn"].diff().fillna(0)==-1
    # og_df[lighton_rows].head()
    lighton_idcs = og_df[lighton_rows].index
    num_targets_spawned = sum(lighton_rows)
    # find the indices where the button was pressed
    buttonPress_rows = og_df["ButtonPressed"].diff().fillna(0)==1
    buttonRelease_rows = og_df["ButtonPressed"].diff().fillna(0)==-1
    num_button_presses = sum(buttonPress_rows)
    # matchup
    rightPress_rows = og_df["TurnSignalRight"].diff().fillna(0)==1
    rightRelease_rows = og_df["TurnSignalRight"].diff().fillna(0)==-1

    leftPress_rows = og_df["TurnSignalLeft"].diff().fillna(0)==1
    leftRelease_rows = og_df["TurnSignalLeft"].diff().fillna(0)==-1

    df2 = og_df.copy()
    df2['gaze_x'] = og_df.GazeDir_COMBINED.apply(lambda x: x[0])
    df2['gaze_y'] = og_df.GazeDir_COMBINED.apply(lambda x: x[1])
    df2['gaze_z'] = og_df.GazeDir_COMBINED.apply(lambda x: x[2])

    gaze_pitches = np.arctan2(df2.gaze_z, df2.gaze_x)*180/np.pi
    gaze_yaws = np.arctan2(df2.gaze_y, df2.gaze_x)*180/np.pi

    low_conf_gazeidcs = (gaze_pitches*gaze_yaws == 0)
    gaze_pitches = gaze_pitches[~low_conf_gazeidcs] ; gaze_yaws = gaze_yaws[~low_conf_gazeidcs]

    # for every light appearance
    # find the nearest button press, before the next target appearance
    max_reaction_time_allowed = 1 #seconds
    time_offsets = []
    hits_and_misses = []

    for idx_num, lighton_idx in tqdm(enumerate(lighton_idcs)):
        offset = 0
        target_tuple = (og_df.loc[lighton_idx], False)
        while lighton_idx+offset < max(og_df.index):
            time_offset = og_df.loc[lighton_idx+offset, "TimeElapsed"] - og_df.loc[lighton_idx, "TimeElapsed"]   

            if (og_df.loc[lighton_idx+offset, "ButtonPressed"] == 1):
                # print( "{0:1.2f}s".format(time_offset))
                # found a response
                # now check if the paddle side matched the PT appearance side wrt the FC
                TimestampCarla_lighton = og_df.loc[lighton_idx, "TimestampCarla"]
                CA_row_lighton = CA_df[CA_df["TimestampCarla"] == TimestampCarla_lighton]
                FC_row_lighton = CA_row_lighton[CA_row_lighton.CustomActor_Name == "PeriphCross"]
                PT_row_lighton = CA_row_lighton[CA_row_lighton.CustomActor_Name == "PeriphTarget"]

                # is the 'y' of the PT less than that of the FC? (left spawn)        
                # TODO : if we want to add a margin of error for stuff that is close, we can do it?
                if PT_row_lighton.CustomActor_Location.squeeze()[1] < FC_row_lighton.CustomActor_Location.squeeze()[1]:
                    if og_df.loc[lighton_idx+offset, "TurnSignalLeft"] == 1:
                        time_offsets += [time_offset]
                        target_tuple = (og_df.loc[lighton_idx], og_df.loc[lighton_idx+offset])
                        # print("Hit @ #{}, {}".format(idx_num, lighton_idx))
                        break
                    elif og_df.loc[lighton_idx+offset, "TurnSignalRight"] == 1:
                        # this is the wrong side response
                        # print("Wrong sided response @ {}".format(lighton_idx))
                        break
                else:
                    if og_df.loc[lighton_idx+offset, "TurnSignalRight"] == 1:
                        time_offsets += [time_offset]
                        target_tuple = (og_df.loc[lighton_idx], og_df.loc[lighton_idx+offset])
                        # print("Hit @ #{}, {}".format(idx_num, lighton_idx))
                        break
                    elif og_df.loc[lighton_idx+offset, "TurnSignalLeft"] == 1:
                        # this is the wrong side response
                        # print("Wrong sided response @ {}".format(lighton_idx))
                        break
            else:
                if time_offset > max_reaction_time_allowed:
                    # print("Miss @ #{}, {}".format(idx_num+1, lighton_idx))
                    break
                offset += 1
                # print(offset)
        hits_and_misses += [target_tuple]
        # print("{}/{} hits with a {}s average reaction time".format(len(time_offsets), len(lighton_idcs), sum(time_offsets)/len(time_offsets)))
    
    return hits_and_misses

def add_gaze_event2df(df_new):    
    '''
    Given the data_df add a column for gaze events
    '''
    
    df2 = df_new.copy()
    # add approx head compensation
    df2['Cgaze_x'] = df_new.GazeDir_COMBINED.apply(lambda x: x[0])
    df2['Cgaze_y'] = df_new.GazeDir_COMBINED.apply(lambda x: x[1])
    df2['Cgaze_z'] = df_new.GazeDir_COMBINED.apply(lambda x: x[2])

    # gaze+head values
    gaze_pitches, gaze_yaws = GetGazeDeviationFromHead(df2.Cgaze_x, df2.Cgaze_y, df2.Cgaze_z)
    # head_rots = df2.CameraRot.values
    head_pitches =   df2.CameraRot.apply(lambda x: x[0])
    head_yaws = df2.CameraRot.apply(lambda x: x[2])
    gaze_head_pitches = gaze_pitches + head_pitches
    gaze_head_yaws = gaze_yaws + head_yaws       

    # Create the new pd
    gazeHeadDF = pd.DataFrame(df2[['TimeElapsed']])
    gazeHeadDF = gazeHeadDF.rename(columns={'TimeElapsed':'timestamp'})
    gazeHeadDF['confidence'] = (df2.EyeOpennessValid_LEFT*df2.EyeOpennessValid_RIGHT).astype(bool)
#     gazeHeadDF['x'] = gaze_head_pitches
#     gazeHeadDF['y'] = gaze_head_yaws
#     gazeHeadDF['z'] = np.zeros(len(gaze_head_pitches))
    gazeHeadDF['x'] = df2['Cgaze_x']
    gazeHeadDF['y'] = df2['Cgaze_y']
    gazeHeadDF['z'] = df2['Cgaze_z']

    vel_w = EyeClassifier.preprocess(gazeHeadDF, dist_method="vector")
#     vel_w = EyeClassifier.preprocess(gazeHeadDF, dist_method="euclidean")
    model = EyeClassifier()
    model.fit(world=vel_w)
    # raw_vel = vel_w[np.logical_not(vel_w.velocity.isna())].velocity.values
    # raw_vel[raw_vel > raw_vel.mean() + 3 * raw_vel.std()]
    # print("Velocity Means: ",model.world_model.means_)
    # 0- fix, 1- sacc, -1 ->noise
    labels, indiv_labels = model.predict(world=vel_w)
    labels_unique = labels
    
    df_new = df_new.join(labels_unique["label"])
    return df_new    