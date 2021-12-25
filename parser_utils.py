import numpy as np
import pandas as pd

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
        col_name = col_name.strip(' ')
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
            # print(frame, timestamp)
            df.loc[frame, "TimeElapsed"] = timestamp
        else:
            for item in line.split(';'):
                col_name, val = item.split(':')
                col_name = col_name.strip(' ')
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