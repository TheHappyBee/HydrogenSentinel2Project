
import os
import coastline
import b
import ee
import geemap
import pandas
import math
from LSTM import ConvLSTMCell, ConvLSTM, ConvLSTMModel
import torch
from torch import nn
import torch.nn.functional as F
import numpy
import sys
import itertools

ee.Authenticate()

ee.Initialize(project= "experimentation-472813")

numpy.set_printoptions(threshold=sys.maxsize)


US = [[-171, 18], [-66, 71]]

imageCol = ee.ImageCollection('NOAA/CPC/Temperature') \
    .filterDate(f'{b.cpcStart}-01-01', f'{b.cpcEnd}-01-01') \
    .select('tmin', 'tmax') \
    .toList(365 * 46)
# def findMeanIgnoreDefault(data):
#     sum = 0
#     num = 0
#     for val in data:
#       if (val != 9999):
#         sum += val
#         num += 1
#     return sum / num
# def finder(longitude, latitude, df):
#     low = 0
#     high = len(df['longitude'])
#     ans = 0
#     while (low <= high):
#         mid = low + (high - low) / 2
#         if (int(df['longitude'][mid]) == longitude and int(df['latitude'][mid]) == latitude):
#             ans = mid
#             breakhttps://stackoverflow.com/
#         elif (int(df['longitude'][mid]) < longitude and int(df['latitude'][mid]) < latitude):
#             low = mid + 1
#         else:
#             high = mid - 1
#     return df.loc[[ans]]
        
def getElevationData():
    df = pandas.read_csv("elevations.csv")
    return df
def exp(a):
    return math.e ** a
def specficToRelative(value, temp, press):
    press = press / 100
    es = 6.112 * exp((17.67 * temp)/(temp + 243.5))
    e = value * press / (0.378 * value + 0.622)
    rh = e / es
    return 100 * rh
def loLaTable():
    df = getElevationData()
    table = {}
    for index, row in df.iterrows():
        stuff = []
        columns = df.columns.tolist()
        for i in range(2, len(columns)):
            stuff.append(row[columns[i]])
        table[str(row['longitude']) + " " + str(row['latitude'])] = stuff
    return table
loLaTa = loLaTable()


def createSingleTimeStep(year, day):
    """Fast server-side aggregation for a single timestep across the grid.

    This builds a FeatureCollection of all grid rectangles and runs
    reduceRegions once per band, which is far faster than per-cell reduceRegion
    calls.
    """
    # helper to convert EE FeatureCollection or Dictionary to DataFrame
    def ee_result_to_df(obj):
        if obj is None:
            return pandas.DataFrame()
        try:
            df_try = geemap.ee_to_df(obj)
            if isinstance(df_try, dict):
                return pandas.DataFrame([df_try])
            return df_try
        except Exception:
            try:
                info = obj.getInfo()
                # If getInfo returns an EE FeatureCollection-like dict with 'features'
                # each element contains a 'properties' dict. Extract properties list.
                if isinstance(info, dict) and 'features' in info and isinstance(info['features'], list):
                    props = []
                    for f in info['features']:
                        p = f.get('properties')
                        if isinstance(p, dict):
                            props.append(p)
                    if props:
                        return pandas.DataFrame(props)
                # If info itself is a dict mapping keys to values, wrap it
                if isinstance(info, dict):
                    return pandas.DataFrame([info])
                if isinstance(info, list):
                    return pandas.DataFrame(info)
            except Exception:
                return pandas.DataFrame()

    currentDay = ee.Image(imageCol.get((year - b.cpcStart) * day))
    windDataset = b.windData.toList(365 * (b.rtmaend - b.rtmastart))
    yearCalc = year - b.rtmastart
    currentWind = ee.Image(windDataset.get(yearCalc * day)) if yearCalc >= 0 else None
    yearCalc2 = year - b.chrstart
    currentHum = ee.Image(windDataset.get(yearCalc2 * day)) if (year > b.rtmastart and year < b.rtmaend) else None

    items = list(loLaTa.keys())
    # build server-side FeatureCollection
    feats = []
    for idx, item in enumerate(items):
        i = int(float(item.split(" ")[0]))
        j = int(float(item.split(" ")[1]))
        geom = ee.Geometry.Rectangle([i, j, i + b.step, j + b.step])
        feats.append(ee.Feature(geom).set({'idx': idx}))

    fc = ee.FeatureCollection(feats)
    print("doing server side reductions")
    # server-side reductions
    tmin_fc = currentDay.select('tmin').reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=30)
    tmax_fc = currentDay.select('tmax').reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=30)

    ugrd_fc = vgrd_fc = spfh_fc = rh_fc = None
    if currentWind is not None and (year > b.rtmastart and year < b.rtmaend):
        ugrd_fc = currentWind.select('UGRD').reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=30)
        vgrd_fc = currentWind.select('VGRD').reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=30)
        spfh_fc = currentWind.select('SPFH').reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=30)
    if currentHum is not None and (year > b.chrstart and year < b.chrend):
        rh_fc = currentHum.select('relative_humidity').reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=30)
    print("converting to dataframes")
    # Convert FCs to DataFrames
    tmin_df, tmax_df = ee_result_to_df(tmin_fc), ee_result_to_df(tmax_fc)

    def _rename_mean(df, newname):
        cols = [c for c in df.columns if c not in ('id', 'longitude', 'latitude', 'idx')]
        if cols:
            df = df.rename(columns={cols[0]: newname})
        else:
            df[newname] = None
        return df

    tmin_df = _rename_mean(tmin_df, 'tmin_mean')
    tmax_df = _rename_mean(tmax_df, 'tmax_mean')

    merged = pandas.merge(tmin_df[['idx', 'tmin_mean']], tmax_df[['idx', 'tmax_mean']], on='idx', how='outer')

    if ugrd_fc is not None:
        ugrd_df = _rename_mean(ee_result_to_df(ugrd_fc), 'ugrd_mean')
        vgrd_df = _rename_mean(ee_result_to_df(vgrd_fc), 'vgrd_mean')
        spfh_df = _rename_mean(ee_result_to_df(spfh_fc), 'spfh_mean')
        merged = merged.merge(ugrd_df[['idx', 'ugrd_mean']], on='idx', how='left')
        merged = merged.merge(vgrd_df[['idx', 'vgrd_mean']], on='idx', how='left')
        merged = merged.merge(spfh_df[['idx', 'spfh_mean']], on='idx', how='left')

    if rh_fc is not None:
        rh_df = _rename_mean(ee_result_to_df(rh_fc), 'rh_mean')
        merged = merged.merge(rh_df[['idx', 'rh_mean']], on='idx', how='left')

    # Build the step list in the original order
    print("formatting data")
    step = [[0.0] * 7 for _ in range(len(items))]
    for _, row in merged.iterrows():
        idx = int(row['idx'])
        item = items[idx]
        i = int(float(item.split(' ')[0]))
        j = int(float(item.split(' ')[1]))
        tmin_m = float(row['tmin_mean']) if pandas.notna(row.get('tmin_mean')) else 0.0
        tmax_m = float(row['tmax_mean']) if pandas.notna(row.get('tmax_mean')) else 0.0
        step[idx][0] = (tmin_m + tmax_m) / 2
        step[idx][1] = coastline.calc_distance_to_coastline(i, j)
        step[idx][2] = loLaTa[item][0]
        step[idx][3] = loLaTa[item][12] if len(loLaTa[item]) > 12 else 0

        if 'ugrd_mean' in row and pandas.notna(row.get('ugrd_mean')) and pandas.notna(row.get('vgrd_mean')):
            res_U = float(row['ugrd_mean'])
            res_V = float(row['vgrd_mean'])
            step[idx][4] = math.atan2(res_V, res_U) if res_U != 0 or res_V != 0 else 0.0
            step[idx][5] = (res_U ** 2 + res_V ** 2) ** 0.5
        else:
            step[idx][4] = 0.0
            step[idx][5] = 0.0

        if 'spfh_mean' in row and pandas.notna(row.get('spfh_mean')):
            temp = (tmin_m + tmax_m) / 2
            step[idx][6] = specficToRelative(float(row['spfh_mean']), temp, 1013)
        elif 'rh_mean' in row and pandas.notna(row.get('rh_mean')):
            step[idx][6] = float(row['rh_mean'])
        else:
            step[idx][6] = 0.0

    return step
        
            
            
if __name__ == '__main__':
    batch = 4
    seq_len = 5
    C, H, W = 1, 3101, 7

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConvLSTMModel(input_channels=C, hidden_channels=[16,32], kernel_size=3, num_layers=2, out_channels=1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # --- Helpers ---------------------------------------------------------
    def sequences_to_tensor(seq_list):
        """Convert a list of sequences (each sequence is a list/array of timesteps)
        where each timestep is a 2D H x W array, into a torch tensor shaped
        (N, T, C, H, W) with C=1.
        seq_list: list of sequences, each sequence: list of T arrays shaped (H, W)
        """
        import numpy as _np
        arr = _np.array(seq_list, dtype=_np.float32)  # (N, T, H, W)
        if arr.ndim != 4:
            raise ValueError(f'seq_list must produce array with ndim==4, got {arr.ndim}')
        # add channel dim at index 2 -> (N, T, 1, H, W)
        arr = arr[:, :, _np.newaxis, :, :]
        return torch.from_numpy(arr)


    def train_step_by_step(model,
                           trainX,
                           trainY,
                           optimizer,
                           criterion,
                           device=None,
                           epochs=1,
                           K=5,
                           per_step_targets=False,
                           clip_grad_norm=1.0,
                           save_checkpoint=None):
        """Train ConvLSTM model timestep-by-timestep with truncated BPTT.

        trainX: (B, T, C, H, W)
        trainY: if per_step_targets -> (B, T, outC, H, W)
                else -> (B, outC, H, W) (target for final timestep)
        K: detach/truncate after every K steps
        """
        if device is None:
            device = next(model.parameters()).device
        model.to(device)

        trainX = trainX.to(device)
        trainY = trainY.to(device)

        B, T, C, H, W = trainX.shape

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            hidden = None
            optimizer.zero_grad()

            for t in range(T):
                x_t = trainX[:, t].unsqueeze(1)  # (B,1,C,H,W)
                out_t, hidden = model(x_t, hidden)

                if per_step_targets:
                    target_t = trainY[:, t]
                else:
                    target_t = trainY if (t == T - 1) else None

                if target_t is not None:
                    loss = criterion(out_t, target_t)
                    (loss / K).backward()
                    total_loss += loss.item()

                if ((t + 1) % K == 0) or (t == T - 1):
                    if clip_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    if hidden is not None:
                        hidden = [(h.detach(), c.detach()) for (h, c) in hidden]

            avg_loss = total_loss / T
            print(f'Epoch [{epoch+1}/{epochs}] avg_loss_per_timestep={avg_loss:.6f}')

            if save_checkpoint:
                ckpt = {'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),}
                torch.save(ckpt, save_checkpoint)
        # return the final hidden state so the caller can continue/inspect
        return hidden

    # --------------------------------------------------------------------

    # Example: build a small training batch from your createSingleTimeStep function
    # (Replace this with your actual dataset creation/loader.)
    start = 2011
    seq_len_demo = 1
    seqs = []
    for i in range(batch):
        # collect seq_len_demo timesteps (here we use the same day index as a placeholder)
        print(start)
        seq = [createSingleTimeStep(start + j, day) for day in range(1, 365 + 1) for j in range(seq_len_demo)]
        seqs.append(seq)
        start += seq_len_demo

    # Convert to tensor (N, T, C, H, W)
    trainX = sequences_to_tensor(seqs)
    # target: predict final timestep (use last frame as target here)
    trainY = trainX[:, -1]  # (B, C, H, W)
    # Train timestep-by-timestep with truncated BPTT
    os.makedirs('checkpoints', exist_ok=True)
    # Train and get final hidden state
    final_hidden = train_step_by_step(model, trainX, trainY, optimizer, criterion, device=device, epochs=3, K=1, per_step_targets=False, save_checkpoint='checkpoints/convlstm_latest.pth')

    # Save final model weights (state_dict)
    final_path = 'checkpoints/convlstm_final.pth'
    torch.save({'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}, final_path)
    print(f'Saved final model checkpoint to {final_path}')

    # Make a prediction for the next timestep using the last observed frame as input
    # next_input shape should be (B, 1, C, H, W)
    next_input = trainX[:, -1].unsqueeze(1).to(device)
    # Use the final hidden state from training (if available), otherwise None
    pred_out, pred_hidden = model(next_input, final_hidden)
    print('Next-step prediction shape:', pred_out.shape)
    # Print a small patch of the first sample's output for quick inspection
    sample = pred_out
    # clamp and convert to CPU numpy for printing
    sample_np = sample.detach().cpu().numpy()
    with open("final.txt", "w+") as w:
        w.write(str(sample_np))
