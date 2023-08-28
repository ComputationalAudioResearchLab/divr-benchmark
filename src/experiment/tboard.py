import time
from tensorboard.program import TensorBoard
from torch.utils.tensorboard.writer import SummaryWriter


class TBoard(SummaryWriter):
    def __init__(self, tensorboard_path: str):
        self.logpath = f'{tensorboard_path}/{time.strftime("%Y_%b_%d-%H_%M_%S-%z")}'
        super().__init__(self.logpath)

    def launch(self):
        tb = TensorBoard()
        tb.configure(argv=[None, "--logdir", self.logpath])
        print("TensorBoard available at:: ", tb.launch())

    def keep_alive(self):
        try:
            while True:
                time.sleep(100)
        except (KeyboardInterrupt, SystemExit):
            print("\nExiting\n")
