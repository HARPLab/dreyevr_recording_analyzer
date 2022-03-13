import numpy as np
import pandas as pd
import re

def parse_new_dreyevr_rec(path_to_recording : str, COMBINED_GAZE_ONLY=True):
    lines = []
    with open(path_to_recording) as f:
        lines = f.readlines()

    data = []
    
    # create list of recorded values and frame data
    frame = []
    for i, line in enumerate(lines):
        if "Frame " in line:
            if frame != []:
                data.append(frame)
            frame = []
            frame.append(line.strip("\n"))
        if "[DReyeVR]" in line:
            frame.append(line.strip("; \n"))
        
    col_names = get_colnames(data, COMBINED_GAZE_ONLY)
    df = pd.DataFrame(columns = col_names)
    df.set_index('FrameNum', drop=True, inplace=True)
    
    from tqdm import tqdm 
    
    for data_row in tqdm(data):
        parse_and_add_row(data_row, df, COMBINED_GAZE_ONLY)
    
    df = df.convert_dtypes()
    #convert xyztypes
    string_cols = df.columns[df.dtypes=='string']
    for string_col in string_cols:
        df[string_col] = df[string_col].apply(convert_XYZstr2array)
    
    return df

def get_colnames(data, COMBINED_GAZE_ONLY):
    col_names = []

    for row in data[0]:
        row = row.strip('[DReyeVR]')
        t1 = row.split(':')    
        row_header = t1[0]
        row = ":".join(t1[1:])
        # print(row_header)
        row_elements = row.split(',')
        # row = row.strip(row_header)

        if row_header=='TimestampCarla':
            col_names.append(row_header)
            pass

        elif row_header=='EyeTracker':
            col_names.append(row_elements[0].split(':')[0])
            col_names.append(row_elements[1].split(':')[0])                       
            #  re.findall("([A-Z]{2,})", row) returns ['COMBINED', 'LEFT', 'RIGHT']        
            # combined, left, right = re.findall("\{(.*?)\}", row) # returns the contents of the {}
            gaze_struct_names = re.findall("([A-Z]{2,})", row)
            gaze_structs = re.findall("\{(.*?)\}", row)
            if COMBINED_GAZE_ONLY:
                gaze_struct_colnames = re.findall("[A-Z][a-z]+(?:[A-Z][a-z]+)*", gaze_structs[0])
                gaze_struct = gaze_structs[0]
                gaze_struct_colnames = [x+"_"+gaze_struct_names[0] for x in gaze_struct_colnames]
                col_names += gaze_struct_colnames
            else:
                for gctr, gaze_struct in enumerate(gaze_structs):
                    gaze_struct_colnames = re.findall("[A-Z][a-z]+(?:[A-Z][a-z]+)*", gaze_struct)
                    gaze_struct = gaze_struct
                    gaze_struct_colnames = [x+"_"+gaze_struct_names[gctr] for x in gaze_struct_colnames]
                    col_names += gaze_struct_colnames
                # raise NotImplementedError("have to implement parsing all 3 gaze values. only COMBINED available")
            # col_names += gaze_struct_colnames
            pass
        elif row_header=='FocusInfo':
            pass
        elif row_header=='EgoVariables':
            pass
        elif row_header=='UserInputs':
            for row_element in row_elements:
                if row_element == '':
                    continue
                col_names.append(row_element.split(':')[0])
            break
            pass
        else:
            # Frame framenum at timestamp seconds
            col_names.append('FrameNum')
            col_names.append('TimeElapsed')    
            # framenum, timestamp = re.findall(r"\d+[.]?\d*", row_header) 
            # print(framenum, timestamp)
    return col_names

def parse_and_add_row(data_row, df : pd.DataFrame, COMBINED_GAZE_ONLY):
    # parse first line to get frame and timestamp for this row
    frame, timestamp = re.findall(r"\d+[.]?\d*", data_row[0])
    frame = int(frame)
    timestamp = float(timestamp)
    # print(frame, timestamp)
    df.loc[frame, "TimeElapsed"] = timestamp
    
    for row in data_row[1:]:
        row = row.strip('[DReyeVR]')
        t1 = row.split(':')    
        row_header = t1[0]
        row = ":".join(t1[1:])
        # print(row_header)
        row_elements = row.split(',')

        if row_header=='TimestampCarla':
            df.loc[frame, row_header] = float(row_elements[0])
            pass

        elif row_header=='EyeTracker':
            df.loc[frame, row_elements[0].split(':')[0]] = float(row_elements[0].split(':')[1])
            df.loc[frame, row_elements[1].split(':')[0]] = float(row_elements[1].split(':')[1])

            #  re.findall("([A-Z]{2,})", row) returns ['COMBINED', 'LEFT', 'RIGHT']
            gaze_struct_names = re.findall("([A-Z]{2,})", row)
            # combined, left, right = re.findall("\{(.*?)\}", row) # returns the contents of the {}
            gaze_structs = re.findall("\{(.*?)\}", row)

            if COMBINED_GAZE_ONLY:
                gaze_struct_colnames = re.findall("[A-Z][a-z]+(?:[A-Z][a-z]+)*", gaze_structs[0])
                gaze_struct = gaze_structs[0]
                gaze_struct_colnames = [x+"_"+gaze_struct_names[0] for x in gaze_struct_colnames]
                gaze_elements = gaze_structs[0][:-1].split(',') # last one is empty so removing
                for i, gaze_elem in enumerate(gaze_elements):
                    gaze_measurement = gaze_elem.split(':')[1]
                    try:
                        gaze_measurement = float(gaze_measurement)
                    except ValueError:
                        pass
                    df.loc[frame, gaze_struct_colnames[i]] = gaze_measurement
            else:
                for gsctr, gaze_struct in enumerate(gaze_structs):
                    gaze_struct_colnames = re.findall("[A-Z][a-z]+(?:[A-Z][a-z]+)*", gaze_struct)
                    gaze_struct_colnames = [x+"_"+gaze_struct_names[gsctr] for x in gaze_struct_colnames]
                    gaze_elements = gaze_struct[:-1].split(',') # last one is empty so removing
                    for i, gaze_elem in enumerate(gaze_elements):
                        gaze_measurement = gaze_elem.split(':')[1]
                        try:
                            gaze_measurement = float(gaze_measurement)
                        except ValueError:
                            pass
                        df.loc[frame, gaze_struct_colnames[i]] = gaze_measurement
                # raise NotImplementedError("have to implement parsing all 3 gaze values. only COMBINED available")
            pass
        elif row_header=='FocusInfo':
            pass
        elif row_header=='EgoVariables':
            pass
        elif row_header=='UserInputs':
            for row_element in row_elements[:-2]:
                colname, measurement = row_element.split(':')
                try:
                    measurement = float(measurement)
                except ValueError:
                    pass
                df.loc[frame, colname] = measurement
                
    return df

def read_periph_recording(path_to_recording : str) -> pd.DataFrame:
    lines = []
    with open(path_to_recording) as f:
        lines = f.readlines()

    data = []
    # create list of recorded values and frame data
    for i, line in enumerate(lines):
        if "gaze2target_pitch" in line:
            line = line.replace("VALIDITY: ", "")
            line = line.replace("INPUTS: ", "")
            data.append(line.strip("; \n"))
        elif "Frame " in line:
            data.append(line.strip("\n"))          
    
    # get all columns from recording file:
    col_names = []
    for item in data[1].split(';'):
        col_name = item.split(':')[0]
        # col_name = col_name.strip(' ')
        col_name = col_name.replace(" ", "")
        col_names.append(col_name)
    # frame nums and times in separate line so init separately
    col_names.append('FrameNum')
    col_names.append('TimeElapsed')

    df = pd.DataFrame(columns = col_names)
    df.set_index('FrameNum', drop=True, inplace=True)
    from tqdm import tqdm 
    
    for i, line in tqdm(enumerate(data)):
        if i % 2 == 0:
            frame, timestamp = re.findall(r"\d+[.]?\d*", line)
            frame = int(frame)
            timestamp = float(timestamp)
            # print(frame, timestamp)
            df.loc[frame, "TimeElapsed"] = timestamp
        else:
            for item in line.split(';'):
                col_name, val = item.split(':')
                # col_name = col_name.strip(" ")
                col_name = col_name.replace(" ", "")
                if '{' in val:
                    val = val[2:-1]
                    val = np.fromstring(val, dtype=float, sep=',')
                elif col_name=="FActorName":
                    pass
                else:
                    val = float(val)
                # print(col_name, val)
                try:
                    df.loc[frame, col_name] = val
                except ValueError:
                    df.loc[frame, col_name] = val.tolist()     
            
    return df


def convert_XYZstr2array(xyz_str):
    xyz_str = xyz_str.split(" ")
    arr = [float(each.split("=")[1]) for each in xyz_str]
    return np.array(arr)

def GetGazeDeviationFromHead(gaze_x, gaze_y, gaze_z):
    # generates pitch and yaw angles of gaze ray from head direction
    # head direction is (1,0,0)
    yaw = np.arctan2(gaze_y, gaze_x)
    pitch = np.arctan2(gaze_z, gaze_x)
    
    return yaw*180/np.pi, pitch*180/np.pi