import time
import os

log_path = r'./Logs'


class Logger():
    def __init__(self, file_name, cfg_note):
        self.log_path = f"{log_path}/{file_name}_{cfg_note}_{time.strftime('%y%m%d')}_{time.strftime('%H%M%S')}.log"
        self.log = open(self.log_path, 'w+')

    def log_in(self, *strings_list):
        for string in strings_list:
            tstr = str(string)
            self.log.write(tstr + '\n')
            print(tstr)

    def flush(self):
        self.log.flush()

    def __del__(self):
        self.log.close()