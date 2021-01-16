from Trainer import load_dataset
import argparse

if __name__ == "__main__":

    # Initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument("inputfolder", help = "Input File")
    args = parser.parse_args()

    load_dataset([args.inputfolder])
    
