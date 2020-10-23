import sys
import argparse


def do():
    parser = argparse.ArgumentParser()

    parser.add_argument('--inputs')
    parser.add_argument('--output', required=False)
    parser.add_argument('--schema', required=False)

    # args = parser.parse_args(["--inputs", "patients:/home/ben/covid/patient_export_ --output geocodes_20200901040146.csv, assessments:/home/ben/covid/assessment_exports_20200901040146.csv"])
    args = parser.parse_args()
    print(args.inputs)
    print(args.output)
    print(args.schema)


    inputs = args.inputs.split(',')
    tokens = [i.strip() for i in inputs]
    if any(':' not in t for t in tokens):
        raise ValueError("'-i/--inputs': must be composed of a comma-separated list of name:file")
    tokens = {t[0]: t[1] for t in (t.split(':', 1) for t in tokens)}
    print(tokens)


if __name__ == '__main__':
    do()


