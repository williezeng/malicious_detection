import pandas as pd
import os
import argparse

DEFAULT_DIR_NAME = 'datasets'
DEFAULT_INPUT_NAME = os.path.join(DEFAULT_DIR_NAME, 'output2.csv')      # Total number of rows is 404358
DEFAULT_OUTPUT_NAME = 'split_dataset_{}.csv'

ROW_SPLIT = 500
TOTAL_ROWS = 1000


def get_number_of_rows(input_file_path):
    # get the number of lines of the csv file to be read
    number_rows = sum(1 for row in (open(input_file_path)))
    print('There are {} amount of rows in the file {}.'.format(number_rows, input_file_path))


def read_write_data(input_file_path, output_file_path, row_split, total_number_of_rows):
    for i in range(0, total_number_of_rows, row_split):
        df = pd.read_csv(input_file_path, nrows=row_split, skiprows=i)  # skip rows that have been read
        df.to_csv(output_file_path.format(i), index=False, mode='a', chunksize=row_split)  # size of data to append for each loop


def parser():
    prsr = argparse.ArgumentParser()
    prsr.add_argument('--row_split', type=int, default=ROW_SPLIT, help='Each file will have N amount of rows. '
                                                                       'This is done by splitting the data by N amount of rows, default is ' + str(ROW_SPLIT))
    prsr.add_argument('--row_total', type=int, default=TOTAL_ROWS, help='The total amount of rows you want to split. '
                                                                        'A limit is placed to avoid creating a huge amount of files. default is ' + str(TOTAL_ROWS))
    prsr.add_argument('--get_total', default=False, action='store_true', help='print the total amount of rows in the csv file. This will take a while if you file is large.')
    prsr.add_argument('--input', type=str, default=DEFAULT_INPUT_NAME, help='The name of the large csv file that you want to split. The default is ' + DEFAULT_INPUT_NAME)
    prsr.add_argument('--output', type=str, default=DEFAULT_DIR_NAME, help='The split files will be located in this dir. The default is ' + DEFAULT_DIR_NAME)

    return prsr

def split_large_csv():
    args = parser().parse_args()


    if not os.path.exists(args.input):
        raise('The input path does not exist.')
    if not os.path.exists(args.output):
        raise('The output dir does not exist.')

    if args.get_total:
        print('Getting rows. This will take a while')
        get_number_of_rows(args.input)
    else:
        # make sure dir does not have any other split csvs
        for fname in os.listdir(args.output):
            if fname.startswith("split_dataset_"):
                os.remove(os.path.join(args.output, fname))
        read_write_data(args.input, os.path.join(args.output, DEFAULT_OUTPUT_NAME), args.row_split, args.row_total)

if __name__ == '__main__':
    split_large_csv()