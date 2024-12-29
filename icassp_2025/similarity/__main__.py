from class_argparse import ClassArgParser


class Main(ClassArgParser):

    def __init__(self) -> None:
        super().__init__("Similarity")

    def generate_embedding(self):
        from .data import Data

        Data().save_data_pickle()

    def analyse(self):
        from .analysis import Analysis

        Analysis().run()


if __name__ == "__main__":
    Main()()
