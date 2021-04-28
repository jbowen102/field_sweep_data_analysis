print("Importing modules...")
import os           # Used for analyzing file paths and directories
import csv          # Needed to read in and write out data
import argparse     # Used to parse optional command-line arguments
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
# https://stackoverflow.com/questions/18603270/progress-indicator-during-pandas-operations
# https://pypi.org/project/tqdm/#pandas-integration
# Gives warning if tqdm version <4.33.0. Ignore.
# https://github.com/tqdm/tqdm/issues/780
import pandas as pd # Series and DataFrame structures
import numpy as np
import traceback
import time
from datetime import datetime
import getpass
from PIL import Image
import hashlib
import glob
from tqdm import tqdm

try:
    import matplotlib
    matplotlib.use("Agg") # no UI backend for use w/ WSL
    # https://stackoverflow.com/questions/43397162/show-matplotlib-plots-and-other-gui-in-ubuntu-wsl1-wsl2
    import matplotlib.pyplot as plt # Needed for optional data plotting.
    PLOT_LIB_PRESENT = True
except ImportError:
    PLOT_LIB_PRESENT = False
# https://stackoverflow.com/questions/3496592/conditional-import-of-modules-in-python

import wpfix as wp
print("...done\n")

# global constants: directory structure
RAW_DATA_DIR = "./data_raw"
BUFFER_DIR = "./data_buffer"
OUTPUT_DIR = "./data_out"
PLOT_DIR = "./plots"

# Find Desktop path, default destination for log files.
username = getpass.getuser()
# https://stackoverflow.com/questions/842059/is-there-a-portable-way-to-get-the-current-username-in-python
home_contents = os.listdir("/mnt/c/Users/%s" % username)
onedrive = [folder for folder in home_contents if "OneDrive -" in folder][0]
desktop_path = "/mnt/c/Users/%s/%s/Desktop" % (username, onedrive)
LOG_DIR = os.path.join(desktop_path, "fsda_errors")
if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)

HEADER_HT = 5 # how many non-data rows at top of raw file.
# KEY_CHANNELS = [
#
# ]
#
# CHANNEL_UNITS = {"time": "s",
#                  "pedal_sw": "off/on",
#                  "pedal_v": "V",
#                  "engine_spd": "rpm",
#                  "gnd_speed": "mph",
#                  "throttle": "deg",
#                  "wtq_RR": "ft-lb",
#                  "wtq_LR": "ft-lb",
#                  "wspd_RR": "rpm",
#                  "wspd_LR": "rpm"
#                  }


class FilenameError(Exception):
    pass

class FileLocError(Exception):
    pass

class DataReadError(Exception):
    pass



class SingleRun(object):
    """Represents a single run from the raw_data directory.
    No data is read in until read_data() method called.
    """
    def __init__(self, auto_find=False, verbose=False, warn_prompt=False):
        # Create a new object to store and print output info
        self.Doc = Output(verbose, warn_prompt)

        if auto_find:
            self.input_path = self.find_run()
        else:
            self.input_path = self.prompt_for_run()

        # self.parse_run_num()
        self.run_label = "TEST" # TEMP

        # Docmument in metadata string to include in output file
        self.meta_str = "Input file: '%s' | " % self.input_path

        try:
            self.process_data()
        except Exception:
            self.log_exception("Processing")


    def find_run(self):
        raw_dir_contents = os.listdir(RAW_DATA_DIR)
        if len(raw_dir_contents) > 1:
            raise FileLocError("More than one file found in data_raw folder.")
        if (len(raw_dir_contents) < 1 or
           not os.path.isfile(os.path.join(RAW_DATA_DIR, raw_dir_contents[0]))):
            raise FileLocError("No file found in data_raw folder.")
        else:
            return os.path.join(RAW_DATA_DIR, raw_dir_contents[0])


    def prompt_for_run(self):
        """Prompts user for what run to process
        Returns SingleRun object."""

        input_str = input("Enter run path\n> ")
        # Need it to be long enough for wpfix not to raise exception.
        while len(input_str) < 2:
            input_str = input("Invalid path entered.\n\t%s\n"
                                    "Enter valid run path\n> " % input_str)

        # Convert Windows path to UNIX format if needed.
        target_path = wp.wpfix(input_str)
        while not os.path.isfile(target_path):
            target_path = wp.wpfix(input("Invalid path entered.\n\t%s\n"
                                    "Enter valid run path\n> " % target_path))

        return target_path

        # run_prompt = "Enter run num (four digits)\n> "
        # target_run_num = input(run_prompt)

        # while not self.validate_run_num(target_run_num):
        #     target_run_num = input(run_prompt)
        # return self.run_dict.get(target_run_num)

    def get_run_label(self):
        return self.run_label
        # not sure if these runs will have simple labels
        # was being extracted by self.parse_run_num()

    def process_data(self):
        """Apply all processing methods to this run."""
        self.read_data()
        # self.add_math_channels()

    def read_data(self):
        """Read in run's data from data_raw directory."""

        with open(self.input_path, "r") as input_ascii_file:
            self.Doc.print("\nReading input data from %s" % self.input_path)
            data_in = csv.reader(input_ascii_file, delimiter="\t")
            # https://stackoverflow.com/questions/7856296/parsing-csv-tab-delimited-txt-file-with-python

            raw_data_dict = {}
            for i, input_row in enumerate(data_in):
                if i == 2:
                    # print channels for debugging
                    self.Doc.print("\tInput file channels:\t" +
                                "  -  ".join([str(c) for c in input_row]), True)

                    # index channels in dict for future reference.
                    self.channel_dict = {}
                    for pos, channel_str in enumerate(input_row):
                        channel = channel_str.split("\\")[0]
                        self.channel_dict[pos] = channel
                        # Set up empty lists for each entry in raw_data_dict.
                        raw_data_dict[channel] = []
                    continue
                elif i == 4:
                    # print units for debugging
                    self.Doc.print("\tInput file units:\t" +
                                "  -  ".join([str(c) for c in input_row]), True)

                    # index units in dict for future reference.
                    self.units_dict = {}
                    for pos, units in enumerate(input_row):
                        channel = self.channel_dict[pos]
                        self.units_dict[channel] = units
                    continue
                elif i < HEADER_HT:
                    # ignore headers
                    continue
                else:
                    # self.Doc.print("Data Row %d:\t" % (i-HEADER_HT) +
                    #             "  -  ".join([str(c) for c in input_row]), True)
                    for n, value_str in enumerate(input_row):
                        channel = self.channel_dict[n]
                        try:
                            raw_data_dict[channel].append(float(value_str))
                        except ValueError:
                            # blanks or any other non-data string
                            raw_data_dict[channel].append(np.nan)
        # for key in raw_data_dict:
        #     print("%s: %d" % (key, len(raw_data_dict[key])))

        # Convert the dict to a pandas DataFrame for easier manipulation
        # and analysis.
        self.raw_df = pd.DataFrame(data=raw_data_dict,
                                                    index=raw_data_dict["time"])
        self.Doc.print("...done")

        self.Doc.print("\nraw_df after reading in data:", True)
        self.Doc.print(self.raw_df.to_string(max_rows=10, max_cols=7,
                                                show_dimensions=True), True)

        ## validate that channels we care about are in there.


        self.Doc.print("", True)


    def add_math_channels(self):
        """Run calculations on data and store in new dataframe."""
        self.math_df = pd.DataFrame(index=self.raw_df.index)
        # https://stackoverflow.com/questions/18176933/create-an-empty-data-frame-with-index-from-another-data-frame

        self.add_ss_avgs()


    def add_ss_avgs(self):
        """Identify steady-state regions of data and calculate average vals."""
        es_win_size_avg = 5001  # window size for engine speed rolling avg.
        es_win_size_slope = 201 # win size for rolling slope of engine speed rolling avg.

        egt_win_size_avg = 1001  # window size for EGT rolling avg.
        egt_win_size_slope = 1001 # win size for rolling slope of EGT rolling avg.

        es_slope_cr = 10  # rpm/s.
        # Engine-speed slope (max) criterion to est. steady-state. Abs value
        egt_slope_cr = 1.5  # degrees Celsius per second.
        # EGT slope (max) criterion to est. steady-state. Abs value
        time_cr = 2.0 # seconds. Continuous period the criteria must be met to
                      # keep event.

        # Document in metadata string for output file:
        self.meta_str += ("Steady-state calc criteria: "
                          "eng speed slope magnitude less than %s rpm/s, "
                          "EGT slope magnitude less than %s degrees(C)/s | "
                                                % (es_slope_cr, egt_slope_cr))
        self.meta_str += ("Steady-state calc rolling window sizes: "
                    "%d for engine-speed avg, %d for engine-speed slope | "
                                        % (es_win_size_avg, es_win_size_slope))
        self.meta_str += ("Steady-state calc rolling window sizes: "
                    "%d for EGT avg, %d for EGT slope | "
                                    % (egt_win_size_avg, egt_win_size_slope))

        # Create rolling average and rolling (regression) slope of rolling avg
        # for engine speed.
        es_rolling_avg = self.raw_df.rolling(
                        window=es_win_size_avg, center=True)["Engine_RPM"].mean()

        # Create and register a new tqdm instance with pandas.
        # Have to manually feed it the total iteration count.
        tqdm.pandas(total=len(self.raw_df.index)-(es_win_size_avg-1)-(es_win_size_slope-1))
        # https://stackoverflow.com/questions/48935907/tqdm-not-showing-bar
        self.Doc.print("Calculating rolling regression on engine speed data...")
        es_rolling_slope = es_rolling_avg.rolling(
                window=es_win_size_slope, center=True).progress_apply(
                                        lambda x: np.polyfit(x.index, x, 1)[0])
        self.Doc.print("...done")

        # Create rolling average and rolling (regression) slope of rolling avg
        # for EGT.
        egt_rolling_avg = self.raw_df.rolling(
               window=egt_win_size_avg, center=True)["Exhaust_Temperature"].mean()
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html

        tqdm.pandas(total=len(self.raw_df.index)-(egt_win_size_avg-1)-(egt_win_size_slope-1))
        # https://stackoverflow.com/questions/48935907/tqdm-not-showing-bar
        self.Doc.print("\nCalculating rolling regression on EGT data...")
        egt_rolling_slope = egt_rolling_avg.rolling(
                window=egt_win_size_slope, center=True).progress_apply(
                                        lambda x: np.polyfit(x.index, x, 1)[0])
        # https://stackoverflow.com/questions/18603270/progress-indicator-during-pandas-operations
        self.Doc.print("...done")


        # Apply engine speed slope criteria to isolate steady-state events.
        ss_filter = (      (es_rolling_slope < es_slope_cr)
                         & (es_rolling_slope > -es_slope_cr))
                         # & (self.math_df["egt_rolling_slope"] < egt_slope_cr)
                         # & (self.math_df["egt_rolling_slope"] > -egt_slope_cr) )
        # egt_slope_cr and es_slope_cr are abs value so have to apply on high
        # and low end.


        # # Apply engine speed and EGT slope criteria to isolate steady-state events.
        # ss_filter = (      (self.math_df["es_rolling_slope"] < es_slope_cr)
        #                  & (self.math_df["es_rolling_slope"] > -es_slope_cr)
        #                  & (self.math_df["egt_rolling_slope"] < egt_slope_cr)
        #                  & (self.math_df["egt_rolling_slope"] > -egt_slope_cr) )
        # egt_slope_cr and es_slope_cr are abs value so have to apply on high
        # and low end.


        # Mask off every data point not meeting the filter criteria.
        es_rol_avg_mskd = es_rolling_avg.mask(~ss_filter)
        es_rslope_mskd = es_rolling_slope.mask(~ss_filter)
        # Convert to a list of indices.
        valid_times = es_rol_avg_mskd[~es_rol_avg_mskd.isna()]

        if len(valid_times) == 0:
            # If no times were stored, then alert user but continue with
            # program.
            self.Doc.warn("No valid steady-state events found in run %s.")
            # self.Doc.warn("No valid steady-state events found in run %s (Criteria: "
            #     "speed slope >%d mph/s, speed >%d mph, and throttle <%d deg).\n"
            #                        "Processing will continue without abridging."
            #             % (self.run_label, gs_slope_cr, gspd_cr, throttle_cr))
            # # Take care of needed assignments that are typically down below.
            # self.sync_df["gs_rolling_avg"] = gs_rolling_avg
            # self.sync_df["gs_rolling_slope"] = gs_rolling_slope
            # self.sync_df["downhill_filter"] = downhill_filter
            # self.sync_df["trendlines"] = np.nan
            # self.sync_df["slopes"] = np.nan
            # self.abr_df = self.sync_df.copy(deep=True)
            #
            # self.meta_str += ("No valid downhill events found in run (Criteria: "
            # "speed slope >%d mph/s, speed >%d mph, and throttle <%d deg). "
            # "Data unabridged. | " % (gs_slope_cr, gspd_cr, throttle_cr))
            # return

        # Identify separate continuous ranges.
        cont_ranges = [] # ranges w/ continuous data (no NaNs)
        current_range = [valid_times.index[0]]
        for i, time in enumerate(valid_times.index[1:]):
            prev_time = valid_times.index[i] # i is behind by one.
            if self.raw_df.index.get_loc(time) - self.raw_df.index.get_loc(prev_time) > 1:
                current_range.append(prev_time)
                cont_ranges.append(current_range)
                # Reset range
                current_range = [time]
        # Add last value to end of last range
        current_range.append(time)
        cont_ranges.append(current_range)

        self.Doc.print("\nSteady-state ranges (before imposing length req.):", True)
        for event_range in cont_ranges:
            self.Doc.print("\t%0.2f\t->\t%0.2f"
                                       % (event_range[0], event_range[1]), True)

        valid_slopes = []
        for range in cont_ranges:
            if range[1]-range[0] > time_cr:
                # Must have > time_cr seconds to count.
                valid_slopes.append(range)
            else:
                # Adjust filter to eliminate these extraneous events.
                ss_filter[range[0]:range[1]] = False

        if not valid_slopes:
            # If no times were stored, then alert user but continue with
            # program.
            self.Doc.print("No valid steady-state events found in run %s (after imposing length requirement).")
            # self.Doc.print("\nNo valid downhill events found in run %s "
            # "(Criteria: speed slope >%d mph/s, speed >%d mph, and throttle <%d "
            #     "deg for >%ds).\nProcessing will continue without abridging."
            # % (self.run_label, gs_slope_cr, gspd_cr, throttle_cr, gs_slope_t_cr))
            # self.meta_str += ("No valid downhill events found in run (Criteria: "
            # "speed slope >%d mph/s, speed >%d mph, and throttle <%d deg for "
            # ">%ds). Data unabridged. | "
            #               % (gs_slope_cr, gspd_cr, throttle_cr, gs_slope_t_cr))
            #
            # self.sync_df["gs_rolling_avg"] = gs_rolling_avg
            # self.sync_df["gs_rolling_slope"] = gs_rolling_slope
            # self.sync_df["downhill_filter"] = downhill_filter
            # self.sync_df["trendlines"] = np.nan
            # self.sync_df["slopes"] = np.nan
            # self.abr_df = self.sync_df.copy(deep=True)
            # return
        # else:
        #     # Document in output file
        #     self.meta_str += ("Isolated events where speed slope exceeded %d "
        #     "mph/s with speed >%d mph and throttle <%d deg for >%ds. "
        #     "Removed extraneous surrounding events. "
        #     "These same criteria were used for the downhill calcs. | "
        #         % (gs_slope_cr, gspd_cr, throttle_cr, gs_slope_t_cr))
        #
        # # Document window sizes in metadata string for output file:
        # self.meta_str += ("Isolation and downhill calc rolling window sizes: "
        #                                         "%d for avg, %d for slope | "
        #                                     % (win_size_avg, win_size_slope))

        self.Doc.print("\nSteady-state ranges (after imposing length req.):", True)
        for valid_range in valid_slopes:
            self.Doc.print("\t%0.2f\t->\t%0.2f"
                                       % (valid_range[0], valid_range[1]), True)

        self.Doc.print("\nTotal data points that fail steady-state criteria: %d"
                                        % sum(~ss_filter), True)
        self.Doc.print("Total data points that meet steady-state criteria: %d"
                                         % sum(ss_filter), True)
        # https://stackoverflow.com/questions/12765833/counting-the-number-of-true-booleans-in-a-python-list

        self.math_df["steady_state"] = ss_filter
        self.math_df["egt_rolling_avg"] = egt_rolling_avg
        self.math_df["es_rolling_avg"] = es_rolling_avg
        self.math_df["egt_rolling_slope"] = egt_rolling_slope
        self.math_df["es_rolling_slope"] = es_rolling_slope

        # "Mask off" by assigning NaN where criteria not met.
        self.math_df["egt_rol_avg_mskd"] = self.math_df["egt_rolling_avg"].mask(
                                                                    ~ss_filter)
        self.math_df["es_rol_avg_mskd"] = self.math_df["es_rolling_avg"].mask(
                                                                    ~ss_filter)
        # Masking these too to calculate avg slope off SS region later:
        self.math_df["egt_rslope_mskd"] = self.math_df["egt_rolling_slope"].mask(
                                                                    ~ss_filter)
        self.math_df["es_rslope_mskd"] = self.math_df["es_rolling_slope"].mask(
                                                                    ~ss_filter)
        # https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#indexing-where-mask

        # pandas rolling(), apply(), regression references:
        # https://stackoverflow.com/questions/47390467/pandas-dataframe-rolling-with-two-columns-and-two-rows
        # https://pandas.pydata.org/pandas-docs/version/0.23.4/whatsnew.html#rolling-expanding-apply-accepts-raw-false-to-pass-a-series-to-the-function
        # https://stackoverflow.com/questions/49100471/how-to-get-slopes-of-data-in-pandas-dataframe-in-python
        # https://www.pythonprogramming.net/rolling-apply-mapping-functions-data-analysis-python-pandas-tutorial/
        # https://stackoverflow.com/questions/21025821/python-custom-function-using-rolling-apply-for-pandas
        # http://greg-ashton.physics.monash.edu/applying-python-functions-in-moving-windows.html
        # https://stackoverflow.com/questions/50482884/module-pandas-has-no-attribute-rolling-mean
        # https://stackoverflow.com/questions/45254174/how-do-pandas-rolling-objects-work
        # https://pandas.pydata.org/pandas-docs/stable/user_guide/computation.html
        # https://becominghuman.ai/linear-regression-in-python-with-pandas-scikit-learn-72574a2ec1a5
        # https://medium.com/the-code-monster/split-a-dataset-into-train-and-test-datasets-using-sk-learn-acc7fd1802e0
        # https://towardsdatascience.com/regression-plots-with-pandas-and-numpy-faf2edbfad4f
        # https://data36.com/linear-regression-in-python-numpy-polyfit/


    def plot_data(self, overwrite=False, description=""):
        """Plot various raw and calculated data from run."""
        self.overwrite = overwrite
        self.description = description
        self.Doc.print("") # blank line

        self.plot_demo_segments()

        # self.plot_raw_basic()
        # self.plot_ss_range()

        # each of these calls export_plot() and clears fig afterward.

    def plot_raw_basic(self):
        """Creates a plot showing raw engine speed and throttle opening."""
        ax1 = plt.subplot(411)
        plt.plot(self.raw_df.index, self.raw_df["Engine_RPM"],
                                        label="Engine Speed", color="tab:blue")
        plt.title("Run %s - Raw Data" % self.run_label, loc="left")
        plt.ylabel("Engine Speed (rpm)")

        plt.setp(ax1.get_xticklabels(), visible=False)

        ax2 = plt.subplot(412, sharex=ax1)
        plt.plot(self.raw_df.index, self.raw_df["Throttle_Position"],
                                        label="Throttle", color="tab:purple")
        ax2.set_ylabel("Throttle (deg)")

        plt.setp(ax2.get_xticklabels(), visible=False)

        ax3 = plt.subplot(413, sharex=ax1)
        plt.plot(self.raw_df.index, self.raw_df["Exhaust_Temperature"],
                                    label="Exhaust Temp", color="tab:orange")
        ax3.set_ylabel("Exhaust Temp (C)")

        plt.setp(ax3.get_xticklabels(), visible=False)

        ax4 = plt.subplot(414, sharex=ax1)
        plt.plot(self.raw_df.index, self.raw_df["Coolant_Temp"],
                                        label="Coolant Temp", color="tab:blue")
        ax4.set_ylabel("Coolant Temp (C)")

        ax4.set_xlabel("Time (s)")

        # plt.show() # can't use w/ WSL.
        # https://stackoverflow.com/questions/43397162/show-matplotlib-plots-and-other-gui-in-ubuntu-wsl1-wsl2
        self.export_plot("raw_basic")
        plt.clf()
        # https://stackoverflow.com/questions/8213522/when-to-use-cla-clf-or-close-for-clearing-a-plot-in-matplotlib


    def plot_demo_segments(self):

        # while True:
        #     print("\n")
        #     t1 = int(input("First time:\n> "))
        #     t2 = int(input("Second time:\n> "))

        segments = [[108, 128], [150, 165], [200, 220],
                    [305, 325], [364, 388], [245, 265],
                    [417, 434], [465, 475], [537, 551],
                    [576, 589], [605, 625], [639, 659],
                    [676, 696], [734, 764], [777, 807],
                    [817, 847], [901, 921], [933, 953],
                    [965, 995], [1049, 1069], [1081, 1102],
                    [1112, 1126], [1130, 1142]]

        offset = 50 # seconds

        for i, seg in enumerate(segments):
            self.plot_raw_segment(seg, offset)


    def plot_raw_segment(self, times, time_offset):

        # Handle time values too close to start or end of data.
        # Offset times:
        offset_time1 = max(times[0] - time_offset, self.raw_df.index[0])
        offset_time2 = min(times[1] + time_offset, self.raw_df.index[-1])

        self.Doc.print("\nraw_df chosen segment:", True)
        self.Doc.print(self.raw_df[offset_time1:offset_time2].to_string(
                        max_rows=10, max_cols=7, show_dimensions=True), True)

        ax1 = plt.subplot(411)
        plt.plot(self.raw_df.loc[offset_time1:offset_time2].index,
                 self.raw_df["Engine_RPM"][offset_time1:offset_time2],
                                        label="Engine Speed", color="tab:orange")
        plt.plot(self.raw_df.loc[times[0]:times[1]].index,
                 self.raw_df["Engine_RPM"][times[0]:times[1]],
                                        label="Engine Speed", color="tab:blue")

        print(self.raw_df["Engine_RPM"][times[0]:times[1]].mean())
        ax1.set_ylim([max(self.raw_df["Engine_RPM"][times[0]:times[1]].mean() - 700, 0),
                      min(self.raw_df["Engine_RPM"][times[0]:times[1]].mean() + 700, 8400)])
        plt.title("Run %s - Raw Data" % self.run_label, loc="left")
        plt.ylabel("Engine Speed (rpm)")

        plt.setp(ax1.get_xticklabels(), visible=False)

        ax2 = plt.subplot(412, sharex=ax1)
        plt.plot(self.raw_df.loc[offset_time1:offset_time2].index,
                 self.raw_df["Throttle_Position"][offset_time1:offset_time2],
                                        label="Throttle", color="tab:purple")
        plt.plot(self.raw_df.loc[times[0]:times[1]].index,
                 self.raw_df["Throttle_Position"][times[0]:times[1]],
                                        label="Throttle", color="lightgrey")
        ax2.set_ylim([max(self.raw_df["Throttle_Position"][times[0]:times[1]].mean() - 7, 0),
                      min(self.raw_df["Throttle_Position"][times[0]:times[1]].mean() + 7, 80)])
        ax2.set_ylabel("Throttle (deg)")

        plt.setp(ax2.get_xticklabels(), visible=False)

        ax3 = plt.subplot(413, sharex=ax1)
        plt.plot(self.raw_df.loc[offset_time1:offset_time2].index,
                 self.raw_df["Exhaust_Temperature"][offset_time1:offset_time2],
                                    label="Exhaust Temp", color="yellowgreen")
        plt.plot(self.raw_df.loc[times[0]:times[1]].index,
                 self.raw_df["Exhaust_Temperature"][times[0]:times[1]],
                                    label="Exhaust Temp", color="tab:orange")
        ax3.set_ylabel("Exhaust Temp (C)")

        plt.setp(ax3.get_xticklabels(), visible=False)

        ax4 = plt.subplot(414, sharex=ax1)
        plt.plot(self.raw_df.loc[offset_time1:offset_time2].index,
                 self.raw_df["Coolant_Temp"][offset_time1:offset_time2],
                                        label="Coolant Temp", color="tab:blue")
        ax4.set_ylabel("Coolant Temp (C)")

        ax4.set_xlabel("Time (s)")

        # plt.show() # can't use w/ WSL.
        # https://stackoverflow.com/questions/43397162/show-matplotlib-plots-and-other-gui-in-ubuntu-wsl1-wsl2
        self.export_plot("raw_segment_%ds-%ds" % (times[0], times[1]))
        plt.clf()
        # https://stackoverflow.com/questions/8213522/when-to-use-cla-clf-or-close-for-clearing-a-plot-in-matplotlib



    def plot_ss_range(self):
        """Creates a plot showing highlighted steady-state regions and the data
        used to identify them."""
        # ax1 = plt.subplot(311)
        ax1 = plt.subplot(211)

        plt.plot(self.raw_df.index, self.raw_df["Engine_RPM"],
                                    label="Engine Speed", color="yellowgreen")

        plt.plot(self.raw_df.index, self.math_df["es_rolling_avg"],
                                        label="Rolling Avg", color="tab:orange")
        plt.plot(self.raw_df.index, self.math_df["es_rol_avg_mskd"],
                                        label="Steady-state", color="tab:blue")

        plt.ylabel("Engine Speed (rpm)")

        plt.title("Run %s - Steady-state Isolation"
                                                % self.run_label, loc="left")

        plt.setp(ax1.get_xticklabels(), visible=False)

        # ax2 = plt.subplot(312, sharex=ax1)
        ax2 = plt.subplot(212, sharex=ax1)
        plt.plot(self.raw_df.index, self.raw_df["Exhaust_Temperature"],
                                                label="Exhaust Temp", color="k")
        plt.plot(self.raw_df.index, self.math_df["egt_rolling_avg"],
                                                label="Rolling Avg", color="c")
        plt.plot(self.raw_df.index, self.math_df["egt_rol_avg_mskd"],
                                                label="Steady-state", color="r")
        plt.ylabel("EGT (C)")
        ax2.set_xlabel("Time (s)")

        # plt.setp(ax2.get_xticklabels(), visible=False)

        # ax3 = plt.subplot(313, sharex=ax1)
        # color = "tab:purple"
        # # Convert DF indices from hundredths of a second to seconds
        # plt.plot(self.raw_df.index, self.raw_df["throttle"],
        #                                     label="Throttle", color=color)
        # ax3.set_ylim([-25, 100]) # Shift throttle trace up
        # ax3.set_yticks([0, 20, 40, 60, 80, 100])
        # ax3.set_xlabel("Time (s)")
        # ax3.set_ylabel("Throttle (deg)", color=color)
        # ax3.tick_params(axis="y", labelcolor=color)
        #
        # ax3_twin = ax3.twinx() # second plot on same x axis
        # # https://matplotlib.org/gallery/api/two_scales.html
        # color = "tab:red"
        # ax3_twin.plot(self.raw_df.index, self.raw_df["pedal_sw"], color=color)
        # ax3_twin.set_ylim([-.25, 8]) # scale down pedal switch
        # ax3_twin.set_yticks([0, 1])
        # ax3_twin.set_ylabel("Pedal Switch", color=color)
        # ax3_twin.tick_params(axis="y", labelcolor=color)

        # plt.show() # can't use w/ WSL.
        # https://stackoverflow.com/questions/43397162/show-matplotlib-plots-and-other-gui-in-ubuntu-wsl1-wsl2
        self.export_plot("ss")
        plt.clf()
        # https://stackoverflow.com/questions/8213522/when-to-use-cla-clf-or-close-for-clearing-a-plot-in-matplotlib


    def export_plot(self, type):
        """Exports plot that's already been created with another method.
        Assumes caller method will clear figure afterward."""
        if self.description:
            fig_filepath = ("%s/%s_%s-%s.png"
                        % (PLOT_DIR, self.run_label, type, self.description))
        else:
            fig_filepath = "%s/%s_%s.png" % (PLOT_DIR, self.run_label, type)

        short_hash_len = 6
        # Check for existing fig with same filename including description but
        # EXCLUDING hash.
        wildcard_filename = (os.path.splitext(fig_filepath)[0]
                            + "-#" + "?"*short_hash_len
                            + os.path.splitext(fig_filepath)[1])
        if glob.glob(wildcard_filename) and not self.overwrite:
            ow_answer = ""
            while ow_answer.lower() not in ["y", "n"]:
                self.Doc.print("\n%s already exists in figs folder. Overwrite? (Y/N)"
                                        % os.path.basename(wildcard_filename))
                ow_answer = input("> ")
            if ow_answer.lower() == "y":
                for filepath in glob.glob(wildcard_filename):
                    os.remove(filepath)
                # continue with rest of function
            elif ow_answer.lower() == "n":
                # plot will be cleared in caller function.
                return
        elif glob.glob(wildcard_filename) and self.overwrite:
            for filepath in glob.glob(wildcard_filename):
                os.remove(filepath)
                # Must manually remove because if figure hash changes, it will
                # not overwrite original.

        plt.savefig(fig_filepath)
        # Calculate unique hash value (like a fingerprint) to output in CSV's
        # meta_str. Put in img filename too.
        img_hash = hashlib.sha1(Image.open(fig_filepath).tobytes())
        # https://stackoverflow.com/questions/24126596/print-md5-hash-of-an-image-opened-with-pythons-pil
        hash_text = img_hash.hexdigest()[:short_hash_len]
        fig_filepath_hash = (os.path.splitext(fig_filepath)[0] + "-#"
                                + hash_text + os.path.splitext(fig_filepath)[1])
        os.rename(fig_filepath, fig_filepath_hash)
        self.Doc.print("Exported plot as %s." % fig_filepath_hash)
        self.meta_str += ("Corresponding %s fig hash: '%s' | "
                                                            % (type, hash_text))


    def log_exception(self, operation):
        """Write output file for later debugging upon encountering exception."""
        exception_trace = traceback.format_exc()
        # https://stackoverflow.com/questions/1483429/how-to-print-an-exception-in-python

        timestamp = datetime.now().strftime("%Y-%m-%dT%H%M%S")
        # https://stackoverflow.com/questions/415511/how-to-get-the-current-time-in-python
        filename = "%s_Run%s_%s_error.txt" % (timestamp, self.get_run_label(),
                                                            operation.lower())
        self.Doc.print(exception_trace)
        # Wait one second to prevent overwriting previous error if it occurred
        # less than one second ago.
        time.sleep(1)
        full_path = os.path.join(LOG_DIR, filename)
        with open(full_path, "w") as log_file:
            log_file.write(self.get_output().get_log_dump())

        print("\n%s failed on run %s.\nOutput and exception "
            "trace written to '%s'."
                                % (operation, self.get_run_label(), full_path))
        print("") # blank line
        quit()

    def get_output(self):
        return self.Doc

    def __str__(self):
        return self.run_label

    def __repr__(self):
        return ("SingleRun object for run %s" % self.run_label)



class Output(object):
    """Object to store terminal output for log dump if needed."""

    def __init__(self, verbose, warn_p):
        self.verbose = verbose
        self.warn_prompt = warn_p
        self.log_string = ""

    def print(self, string, verbose_only=False):
        """Wrapper for standard print function that duplicates output to
        run-specific buffer."""
        if verbose_only and not self.verbose:
            # Add everything to log even if not output to screen.
            self.add_to_log(string)
            return
        else:
            self.add_to_log(string)
            print(string)

    def add_to_log(self, string):
        self.log_string += string + "\n"

    def warn(self, warn_string):
        # Add to log and display in terminal.
        self.print("\nWarning: " + warn_string)
        if self.warn_prompt:
            input("Press Enter to continue.")
        else:
            self.print("") # blank line

    def get_log_dump(self):
        return self.log_string


def main_prog():
    """This program runs when Python runs this file."""
    global LOG_DIR

    # Set up command-line argument parser
    # https://docs.python.org/3/howto/argparse.html
    # If you pass in any arguments from the command line after "python fsda.py",
    # this interprets them.
    parser = argparse.ArgumentParser(description="Program to analyze field "
                                                                "sweep data.")
    parser.add_argument("-o", "--over", help="Overwrite existing data in "
                    "data_out folder without prompting.", action="store_true")
    parser.add_argument("-a", "--auto", help="Automatically process data file "
                                    "in data_raw folder.", action="store_true")
    parser.add_argument("-p", "--plot", help="Product plots.",
                                                            action="store_true")
    parser.add_argument("-v", "--verbose", help="Include additional output for "
                                            "debugging.", action="store_true")
    parser.add_argument("-d", "--desc", help="Specify a description string to "
        "append to output file names. Plot plot files included if -p also "
                                            "specified.", type=str, default="")
    parser.add_argument("-l", "--log-dir", help="Specify a directory where log "
        "file containing that run's output and error trace should be saved "
                    "when error encountered. Desktop/fsda_errors is default.",
                                                    type=str, default=LOG_DIR)
    parser.add_argument("-i", "--ignore-warn", help="Do not prompt user to "
                                "acknowledge warnings.", action="store_false")
    # https://www.programcreek.com/python/example/748/argparse.ArgumentParser
    args = parser.parse_args()

    if os.path.isdir(args.log_dir):
        LOG_DIR = args.log_dir  # update global variable
    else:
        raise FilenameError("Bad log-dir argument. Must be valid path. "
                                                                    "Aborting.")

    MyRun = SingleRun(args.auto, args.verbose, args.ignore_warn)

    if args.plot and PLOT_LIB_PRESENT:
        if not os.path.exists(PLOT_DIR):
            # Create folder for output plots if it doesn't exist already.
            os.mkdir(PLOT_DIR)
        MyRun.plot_data(args.over, args.desc)
    elif args.plot:
        print("\nFailed to import matplotlib. Cannot plot data.")

    if not os.path.exists(OUTPUT_DIR):
        # Create folder for output data if it doesn't exist already.
        os.mkdir(OUTPUT_DIR)

    # MyRun.export_data(args.over, args.desc)


if __name__ == "__main__":
    main_prog()
