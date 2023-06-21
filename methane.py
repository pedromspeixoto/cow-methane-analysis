import os
import sys
import pandas as pd
import numpy as np
import csv  
import math
import shutil

from argparse import ArgumentParser
from scipy.signal import find_peaks, butter, filtfilt
import matplotlib.pyplot as plt

# list all files with a specific extension in a directory
def list_files(dir, extension):
    r = []
    for root, dirs, files in os.walk(dir):
        dirs.sort()
        for name in files:
            file_extension = os.path.splitext(name)[1]
            if (file_extension == extension):
                r.append(os.path.join(root, name))
    return r

# get cow name from input text
def get_cow_name(df, sample_file):
    row = df[df['Sample File'] == sample_file]
    if not row.empty:
        return row['Cow Name'].iloc[0]
    else:
        return None

# low pass filter
def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

if __name__ == "__main__":

    # read input arguments
    parser = ArgumentParser()
    parser.add_argument("-f", "--folder",
                        dest="folder", required=True,
                        help="folder to scan for files and generate aggregate csv file")
    parser.add_argument("-c", "--cow-name-csv",
                        dest="cow_name_csv", required=True,
                        help="csv file with the corresponding cow names")
    parser.add_argument("-d", "--force-delete",
                        dest="force_delete", required=False,
                        help="flag to understand if the file should be deleted if already exists")
    args = parser.parse_args()

    if args.force_delete == 'N':
        force_delete = False
    else:
        force_delete = True

    print("**** LDM - Methane Analysis - Starting process to generate aggregate csv file ***")

    # get current working directory
    directory = os.getcwd()
    path_to_folder = os.path.join(directory, args.folder)

    # check if aggregate file already exists
    aggregate_file_name = "aggregate.csv"
    aggregate_file_path = os.path.join(path_to_folder, aggregate_file_name)

    if os.path.exists(aggregate_file_path):
        if force_delete:
            print("WARN - File already exists (%s). Recreating the file." % aggregate_file_name)
            os.remove(aggregate_file_path)
        else:
            print("ERROR - Aggregate csv file already exists in target folder (%s). Stopping the process." % aggregate_file_name)
            sys.exit()

    # check if stats file already exists
    stats_file_name = "stats.csv"
    stats_file_path = os.path.join(path_to_folder, stats_file_name)

    if os.path.exists(stats_file_path):
        if force_delete:
            print("WARN - File already exists (%s). Recreating the file." % stats_file_name)
            os.remove(stats_file_path)
        else:
            print("ERROR - Stats csv file already exists in target folder (%s). Stopping the process." % stats_file_name)
            sys.exit()

    # output images folder
    graph_images_path = os.path.join(path_to_folder, "graphs")

    if os.path.exists(graph_images_path):
        if force_delete:
            print("WARN - Graph images already exists (%s). Recreating the directory." % graph_images_path)
            shutil.rmtree(graph_images_path)
            os.mkdir(graph_images_path)
        else:
            print("ERROR - Graph images already exists (%s). Stopping the process." % stats_file_name)
            sys.exit()
    else:
        print("INFO - Graph images does not exist (%s). Creating the directory." % graph_images_path)
        os.mkdir(graph_images_path)

    # get cow name file
    cow_file_path = os.path.join(directory, args.cow_name_csv)
    if os.path.exists(cow_file_path):
        cows = pd.read_csv(cow_file_path, skiprows=0)
    else:
        print("ERROR - Cow profile file not found (%s). Stopping the process." % cow_file_path)
        sys.exit()

    # get all csv files
    csv_files_list = list_files(path_to_folder, '.csv')
    print("%s files were detected for processing. Creating aggregate csv file..." % len(csv_files_list))

    # loop all csv files and write to aggregate file
    first_iteration = True
    row = 0
    for file in csv_files_list:
        file_name = os.path.splitext(os.path.basename(file))[0]
        series = pd.read_csv(file, skiprows=6)
        no_of_samples = len(series)
        series.insert(0, "Sample File", [file_name] * no_of_samples, True)

        # get cow name
        cow_no = get_cow_name(cows, file_name)

        # initialize merged data frame it it is the first iteration
        if first_iteration:
            # init aggregate file
            aggregate_out_file = open(aggregate_file_path, "wt")
            series.to_csv(aggregate_file_path, mode='a', index=False, header=True)

            # init stats file
            stats_out_file = open(stats_file_path, "w", newline='')
            writer = csv.writer(stats_out_file)
            stats_header = ["Cow No", "Profile File", "Measurement Duration (s)","Number of Samples", "Total CH4", "Max", "Min", "Mean", "Std Deviation", 
                            "Calculated Eructation Threshold", "Total Respiratory CH4", "Mean Respiratory CH4", "Std. Dev. Respiratory CH4", 
                            "Respiratory Peaks", "Total Eructation CH4", "Mean Eructation CH4", "Std. Dev. Eructation CH4", "Eructation Peaks",
                            "Eructation Events", "Percentage of Respiratory Emissions", "Percentage of Eructation Emissions", "Max Outliers Adjusted"]
            writer.writerow(stats_header)
            first_iteration = False
        else:
            series.to_csv(aggregate_file_path, mode='a', index=False, header=False)

        # print file name
        print("Processing file %s..." % file_name)

        # check if any value is above the threshold
        max_threshold = 1000
        column = series['Measured Value']
        outliers_adjusted = "No"
        if (column[column > max_threshold].count()) > 0:
            # adjust outliers
            outliers_adjusted = "Yes"
            series.loc[series["Measured Value"] > max_threshold, "Measured Value"] = max_threshold

        # mean and standard deviation of profile
        profile_sum = series['Measured Value'].sum()
        profile_average = series['Measured Value'].mean()
        profile_std = series['Measured Value'].std()
        profile_min = series['Measured Value'].min()
        profile_max = series['Measured Value'].max()

        # calculate eructation threshold of profile
        profile_eructation_threshold = profile_average + (profile_std)
        
        # if mean and std are too low, set a minimum threshold
        min_threshold = 60
        if profile_average < min_threshold and profile_std < min_threshold:
            profile_eructation_threshold = min_threshold

        # calculate emissions due to respiratory emissions
        respiration_dataframe = series.copy()
        respiration_dataframe['Measured Value'] = respiration_dataframe['Measured Value'].apply(lambda x: x if x < profile_eructation_threshold else 0)

        respiratory_dataframe_peaks = respiration_dataframe[respiration_dataframe['Measured Value'] != 0]
        profile_respiratory_sum = respiration_dataframe['Measured Value'].sum()
        profile_respiratory_average = respiratory_dataframe_peaks['Measured Value'].mean()
        profile_respiratory_std = respiratory_dataframe_peaks['Measured Value'].std()

        # calculate emissions due to eructation emissions
        eructation_dataframe = series.copy()
        eructation_dataframe['Measured Value'] = eructation_dataframe['Measured Value'].apply(lambda x: x if x >= profile_eructation_threshold else 0)

        eructation_dataframe_peaks = eructation_dataframe[eructation_dataframe['Measured Value'] != 0]
        profile_eructation_sum = eructation_dataframe['Measured Value'].sum()
        profile_eructation_average = eructation_dataframe_peaks['Measured Value'].mean()
        profile_eructation_std = eructation_dataframe_peaks['Measured Value'].std()

        # create auxiliary array and add points in beggining and end as 0 (to detect peaks in beggining and end of series)
        raw = series['Measured Value'].values
        raw = np.insert(raw, 0, 0)
        raw = np.append(raw, 0)

        all_profile_peaks, _ = find_peaks(raw, prominence=10)
        respiratory_peaks = []
        eructation_peaks = []
        min_peak_threshold = min_threshold
        for peak in all_profile_peaks:
            if raw[peak] < profile_eructation_threshold:
                respiratory_peaks.append(peak)
            else:
                if raw[peak] > min_peak_threshold:
                    eructation_peaks.append(peak)
                else:
                    respiratory_peaks.append(peak)

        # calculate percentages
        percentage_respiratory = math.floor((respiration_dataframe['Measured Value'].sum() / profile_sum) * 100)
        percentage_eructation = math.ceil((eructation_dataframe['Measured Value'].sum() / profile_sum) * 100)

        # filter values.
        n = len(series['Measured Value'].values)
        T = n / 2
        fs = n / T
        cutoff_low = 0.12
        nyq = 0.5 * fs
        order = 2

        # calculate number of eructation events
        try:
            filtered_signal = butter_lowpass_filter(eructation_dataframe['Measured Value'].values, cutoff_low, fs, order)
            event_distance = 30
            eructation_events, _ = find_peaks(filtered_signal, prominence= profile_std, height=profile_eructation_threshold-profile_std, distance=event_distance)
            profile_eructation_events = len(eructation_events)
        except:
            print("Error calculating erucation events for file %s." % file_name)
            profile_eructation_events = 0

        # write to file
        writer.writerow([cow_no, file_name, T, n, profile_sum, profile_max, profile_min, profile_average, profile_std, profile_eructation_threshold,
                        profile_respiratory_sum, profile_respiratory_average, profile_respiratory_std, len(respiratory_peaks),
                        profile_eructation_sum, profile_eructation_average, profile_eructation_std, len(eructation_peaks),
                        profile_eructation_events, percentage_respiratory, percentage_eructation, outliers_adjusted])
        
        # generate graphs
        plt.subplot(4, 1, 1)
        plt.plot(respiratory_peaks, raw[respiratory_peaks], "ob"); plt.plot(eructation_peaks, raw[eructation_peaks], "or"); plt.plot(raw); plt.axhline(y=profile_eructation_threshold,linewidth=1, color='k')
        plt.ylim([0, 1000])
        plt.subplot(4, 1, 2)
        plt.plot(respiration_dataframe['Measured Value'].values)
        plt.ylim([0, 1000])
        plt.subplot(4, 1, 3)
        plt.plot(eructation_dataframe['Measured Value'].values)
        plt.ylim([0, 1000])
        plt.subplot(4, 1, 4)
        plt.plot(eructation_events, filtered_signal[eructation_events], "or"); plt.plot(filtered_signal)
        plt.ylim([0, 1000])
        
        # save graph as image
        graph_image_filename = str(cow_no) + "_" + file_name + ".png"
        plt.savefig(os.path.join(graph_images_path, graph_image_filename))

        # close graph
        plt.clf()

        row = row + 1

    # close files
    aggregate_out_file.close()
    stats_out_file.close()