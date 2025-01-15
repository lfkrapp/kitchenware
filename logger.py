import os
import pandas as pd
from time import time
from datetime import timedelta


class Logger:
    def __init__(self, log_dir, log_name, verbose=True):
        # define log filepath
        self.log_dir = log_dir
        self.log_name = log_name
        self.log_str_filepath = os.path.join(log_dir, log_name + ".log")
        self.log_lst_filepath = os.path.join(log_dir, log_name + ".dat")

        # define logs
        self.log_s = ""
        self.log_l = []

        # debug flag
        self.verbose = verbose

        # start timer
        self.t0 = time()
        self.ts = self.t0

    def print(self, line_raw):
        # convert line to string
        line = str(line_raw)

        # update log and append to log file
        self.log_s += line + "\n"

        # update log file
        with open(self.log_str_filepath, "a") as fs:
            fs.write(line + "\n")

        # debug print
        if self.verbose:
            print(line)

    def store(self, **kwargs):
        # update list log
        self.log_l.append(kwargs)

        # create series
        s = pd.Series(kwargs)

        # update log file
        with open(self.log_lst_filepath, "a") as fs:
            fs.write(f"{s.to_json()}\n")

    def print_profiling_info(self, n_curr, n_step, n_total):
        dt = time() - self.t0
        dts = time() - self.ts
        eta = (n_total - n_curr) * dt / n_step
        self.print("> Elapsed time: {}".format(timedelta(seconds=dt)))
        self.print("> Since last call: {}".format(timedelta(seconds=dts)))
        self.print("> ETA: {}".format(timedelta(seconds=eta)))
        self.ts = time()

    def restart_timer(self):
        self.ts = time()
