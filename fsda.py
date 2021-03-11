print("Importing modules...")
import os           # Used for analyzing file paths and directories
import csv          # Needed to read in and write out data
import argparse     # Used to parse optional command-line arguments
import pandas as pd # Series and DataFrame structures
import numpy as np
import traceback
import time
from datetime import datetime
import getpass
from PIL import Image
import hashlib
import glob

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

class DataReadError(Exception):
    pass



class SingleRun(object):
    """Represents a single run from the raw_data directory.
    No data is read in until read_data() method called.
    """
    def __init__(self, verbose=False, warn_prompt=False):
        # Create a new object to store and print output info
        self.Doc = Output(verbose, warn_prompt)

        self.input_path = self.prompt_for_run()
        # self.parse_run_num()
        self.run_label = "TEST" # TEMP

        # Docmument in metadata string to include in output file
        self.meta_str = "Input file: '%s' | " % self.input_path

        try:
            self.process_data()
        except Exception:
            self.log_exception("Processing")


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

        # self.abridge_data()
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
                    for pos, channel in enumerate(input_row):
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


    def plot_data(self, overwrite=False, description=""):
        """Plot various raw and calculated data from run."""
        self.overwrite = overwrite
        self.description = description
        self.Doc.print("") # blank line

        # each of these calls export_plot() and clears fig afterward.
        # self.plot_abridge_compare()
        # self.plot_cvt_ratio()

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
    # test
    MyRun = SingleRun(args.verbose, args.ignore_warn)

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
