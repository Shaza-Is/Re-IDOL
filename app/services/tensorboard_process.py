import os
import sys

from multiprocessing import Process

class TensorboardServer(Process):
    def __init__(self, log_dp: str):
        super().__init__()
        self.os_name = os.name
        self.log_dp = str(log_dp)

    def run(self):
        if self.os_name == "nt":
            os.system(f"{sys.executable} -m tensorboard.main --logdir '{self.log_dp}' 2> NUL")
        elif self.os_name == "posix":
            os.system(f"{sys.executable} -m tensorboard.main --logdir '{self.log_dp}' >/dev/null 2>&1")
        else:
            raise NotImplementedError(f"No support for OS : {self.os_name}")

class TensorboardSupervisor(object):

    def __init__(self, log_dp: str):
        self.server = TensorboardServer(log_dp)
        self.server.start()
        
    def finalize(self):
        if self.server.is_alive():
            self.server.terminate()
            self.server.join()


