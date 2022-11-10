import argparse
import os
import os.path
from utils import Scanner
import torch

parser = argparse.ArgumentParser(description='Evaluates PowerShell scripts using the specified model.')
parser.add_argument('-s', '--sigfind', action='store_true')
parser.add_argument('-p', '--path')
parser.add_argument('-m', '--model', default='fcnn-384-02')

def main():
    # Parse the arguments
    global args
    args = parser.parse_args()
            
    scanner = Scanner(args.model)
    
    if args.sigfind:
        # Run sigfind function to identify the signature used by the model
        sigfind = scanner.SigFindPerChar(args.path)
    else:
        # Just determine if the specified input file is malicious or not
        score1 = scanner.Score(args.path)
        score2 = scanner.Score(args.path)
        print(f'[*] File:      {args.path}')
        print(f'[*] Malicious: {score1[1] > score1[0]}')
        print(f'[*] Scores:    {score1}')

if __name__ == '__main__':
    main()