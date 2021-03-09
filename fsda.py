print("Importing modules...")
import argparse     # Used to parse optional command-line arguments
import getpass
import os           # Used for analyzing file paths and directories
import csv          # Needed to read in and write out data
import pandas as pd # Series and DataFrame structures
import traceback
from datetime import datetime

try:
    import matplotlib
    matplotlib.use("Agg") # no UI backend for use w/ WSL
    # https://stackoverflow.com/questions/43397162/show-matplotlib-plots-and-other-gui-in-ubuntu-wsl1-wsl2
    import matplotlib.pyplot as plt # Needed for optional data plotting.
    PLOT_LIB_PRESENT = True
except ImportError:
    PLOT_LIB_PRESENT = False
# https://stackoverflow.com/questions/3496592/conditional-import-of-modules-in-python
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
LOG_DIR = "/mnt/c/Users/%s/%s/Desktop" % (username, onedrive)

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

class DataReadError(Exception):
    pass


class RunGroup(object):
    """Represents a collection of runs from the data_raw directory."""
    def __init__(self, process_all=False, start_run=False, verbose=False, warn=False):
        self.verbosity = verbose
        self.warn_p = warn
        # create SingleRun object for each run but don't read in data yet.
        # self.build_run_dict()
        # self.process_runs(process_all, start_run)


class SingleRun(object):
    """Represents a single run from the raw_data directory.
    No data is read in until read_data() method called.
    """
    def __init__(self, input_path, verbose=False, warn_prompt=False):
        # Create a new object to store and print output info
        self.Doc = Output(verbose, warn_prompt)
        self.input_path = input_path
        # self.parse_run_num()

        # Docmument in metadata string to include in output file
        self.meta_str = "Input file: '%s' | " % self.input_path

    def parse_run_num(self):
        pass

    def process_data(self):
        """Apply all processing methods to this run."""
        self.read_data()
        # if int(self.run_label[:2]) > 5:
        #     # only needed for torque-meter runs.
        #     self.combine_torque()
        #     self.calc_gnd_speed()
        # self.abridge_data()
        # self.add_math_channels()

    def read_data(self):
        """Read in run's data from data_raw directory."""

        with open(self.input_path, "r") as input_ascii_file:
            self.Doc.print("\nReading input data from %s" % self.input_path)
            data_in = csv.reader(input_ascii_file, delimiter="\t")
            # https://stackoverflow.com/questions/7856296/parsing-csv-tab-delimited-txt-file-with-python

            for i, input_row in enumerate(data_in):
                if i == 2:
                    # print channels for debugging
                    self.Doc.print("\tInput file channels:\t" +
                                "  -  ".join([str(c) for c in input_row]), True)

                    # index channels in dict for future reference.
                    raw_data_dict = {}
                    self.channel_dict = {}
                    for pos, channel in enumerate(input_row):
                        self.channel_dict[pos] = channel
                        # Set up empty lists for each entry in raw_data_dict.
                        raw_data_dict[channel] = []
                    continue
                elif i < HEADER_HT:
                    # ignore headers
                    continue
                else:
                    for n, value_str in enumerate(input_row):
                        channel = self.channel_dict[n]
                        raw_data_dict[channel].append(float(value_str))

        # Convert the dict to a pandas DataFrame for easier manipulation
        # and analysis.
        self.raw_df = pd.DataFrame(data=raw_data_dict,
                                                    index=raw_data_dict["time"])
        self.Doc.print("...done")

        self.Doc.print("\nraw_df after reading in data:", True)
        self.Doc.print(self.raw_df.to_string(max_rows=10, max_cols=7,
                                                show_dimensions=True), True)
        self.Doc.print("", True)

    def get_run_label(self):
        return self.run_label
        # not sure if these runs will have simple labels
        # was being extracted by self.parse_run_num()

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

        input("\n%s failed on run %s.\nOutput and exception "
            "trace written to '%s'.\nPress Enter to skip this run."
                                % (operation, self.get_run_label(), full_path))
        print("") # blank line


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
    parser.add_argument("-a", "--auto", help="Automatically process all data "
                                    "in data_raw folder.", action="store_true")
    parser.add_argument("-o", "--over", help="Overwrite existing data in "
                    "data_out folder without prompting.", action="store_true")
    parser.add_argument("-p", "--plot", help="Product plots.",
                                                            action="store_true")
    parser.add_argument("-v", "--verbose", help="Include additional output for "
                                            "debugging.", action="store_true")
    parser.add_argument("-d", "--desc", help="Specify a description string to "
        "append to output file names. Plot plot files included if -p also "
                                            "specified.", type=str, default="")
    parser.add_argument("-l", "--log-dir", help="Specify a directory where log "
        "file containing that run's output and error trace should be saved "
                    "when error encountered. Desktop used when unspecified.",
                                                    type=str, default=LOG_DIR)
    parser.add_argument("-i", "--ignore-warn", help="Do not prompt user to "
                                "acknowledge warnings.", action="store_false")
    parser.add_argument("-s", "--start", help="Specify run number to start "
        "with when processing all runs (-a option).", type=str, default=False)
    # https://www.programcreek.com/python/example/748/argparse.ArgumentParser
    args = parser.parse_args()

    if os.path.isdir(args.log_dir):
        LOG_DIR = args.log_dir  # update global variable
    else:
        raise FilenameError("Bad log-dir argument. Must be valid path. "
                                                                    "Aborting.")

    AllRuns = RunGroup(args.auto, args.start, args.verbose, args.ignore_warn)

    if args.plot and PLOT_LIB_PRESENT:
        if not os.path.exists(PLOT_DIR):
            # Create folder for output plots if it doesn't exist already.
            os.mkdir(PLOT_DIR)
        AllRuns.plot_runs(args.over, args.desc)
    elif args.plot:
        print("\nFailed to import matplotlib. Cannot plot data.")

    if not os.path.exists(OUTPUT_DIR):
        # Create folder for output data if it doesn't exist already.
        os.mkdir(OUTPUT_DIR)

    AllRuns.export_runs(args.over, args.desc)


if __name__ == "__main__":
    main_prog()
