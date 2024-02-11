import time
from pathlib import Path
from tensorboard.program import TensorBoard
from torch.utils.tensorboard.writer import SummaryWriter


class TBoard(SummaryWriter):
    def __init__(self, tensorboard_path: Path):
        self.logpath = Path(
            f'{tensorboard_path}/{time.strftime("%Y_%b_%d-%H_%M_%S-%z")}'
        )
        self.logpath.mkdir(exist_ok=True, parents=True)
        super().__init__(self.logpath)

    def launch(self):
        tb = TensorBoard()
        tb.configure(argv=[None, "--logdir", str(self.logpath)])
        print("TensorBoard available at:: ", tb.launch())

    def keep_alive(self):
        try:
            while True:
                time.sleep(100)
        except (KeyboardInterrupt, SystemExit):
            print("\nExiting\n")
