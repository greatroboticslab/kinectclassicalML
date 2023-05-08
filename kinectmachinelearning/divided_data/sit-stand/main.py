'Script for joining zero or more JSON files where the top level is a list.'

import json
import sys

USAGE = 'python %s output [input ...]' % sys.argv

if __name__ == '__main__':
    ## Check for valid arguments.
    if len(sys.argv) < 2:
        print(USAGE)

    ## Unpack arguments
    JSON_OUTPUT_FILENAME = sys.argv[1]
    JSON_INPUT_FILENAMES = sys.argv[2:]
    
    ## Build joined JSON file.
    accum = []
    for filename in JSON_INPUT_FILENAMES:
        with open(filename) as f_obj:
            accum.append(json.load(f_obj))

    ## Save joined JSON file.
    with open(JSON_OUTPUT_FILE_NAME) as f_obj:
        json.dump(accum, f_obj)
