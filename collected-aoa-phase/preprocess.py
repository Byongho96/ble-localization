import os
import csv
import datetime
import pandas as pd

def to_utc_timestamp_ms(iso_str):
    dt = datetime.datetime.fromisoformat(iso_str)
    return int(dt.replace(tzinfo=datetime.timezone.utc).timestamp() * 1000)

def gt():
    """
    Reads a CSV file with columns: StartTimeISO, EndTimeISO, X, Y
    Converts StartTimeISO and EndTimeISO to UTC timestamps in milliseconds
    Adds new columns: StartTimestamp, EndTimestamp, Z (default 0)
    Reorders columns as: StartTimeISO, StartTimestamp, EndTimeISO, EndTimestamp, X, Y, Z
    Saves the transformed data to a new CSV file at the specified output path
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Read the input CSV file
    input_path = os.path.join(base_dir, "../dataset/0414/gt/anchor4-west-1.csv")
    df = pd.read_csv(input_path)

    # Compute new timestamp columns
    df["StartTimestamp"] = df["StartTimeISO"].apply(to_utc_timestamp_ms)
    df["EndTimestamp"] = df["EndTimeISO"].apply(to_utc_timestamp_ms)

    # Reorder columns
    df = df[["StartTimeISO", "StartTimestamp", "EndTimeISO", "EndTimestamp", "X", "Y", "Z"]]

    # Write the transformed DataFrame to the output CSV file
    output_path = os.path.join(base_dir, "../dataset/0414/gt/anchor4-west.csv")
    df.to_csv(output_path, index=False)

def beacons():
    """
    Reads a CSV file with columns: Timestamp, Data
    Converts Timestamp to UTC timestamp in milliseconds
    Parses the Data column to extract Tag, RSSI, Azimuth, Elevation, Channel, Anchor, Sequence
    Saves the transformed data to a new CSV file at the specified output path
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Read the input CSV file
    input_path = os.path.join(base_dir, "../dataset/0414/beacons/anchor4-west-1.csv")
    output_path = os.path.join(base_dir, "../dataset/0414/beacons/anchor4-west.csv")

    with open(input_path, mode='r', encoding='utf-8') as infile, \
        open(output_path, mode='w', encoding='utf-8', newline='') as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        next(reader)  # Skip header
        
        writer.writerow(["Timestamp", "Tag", "RSSI", "Azimuth", "Elevation", "Channel", "Anchor", "Sequence"])
        
        for row in reader:
            timestamp, data = row
            timestamp = to_utc_timestamp_ms(timestamp)
            
            # Extract relevant data
            data_parts = data.split(',')
            tag = data_parts[0].split(':')[1]
            rssi = int(data_parts[1])
            azimuth = int(data_parts[2])
            elevation = int(data_parts[3])
            channel = int(data_parts[5])
            anchor = data_parts[6].strip('"')
            sequence = int(data_parts[-1])
            
            # Write formatted row
            writer.writerow([timestamp, tag, rssi, azimuth, elevation, channel, anchor, sequence])

if __name__ =="__main__": 
    gt()
    # beacons()