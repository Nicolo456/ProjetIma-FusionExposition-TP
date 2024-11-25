import argparse

def init_arg_parser():
    parser = argparse.ArgumentParser(description="Execute la fusion d'expostion")
    parser.add_argument('-r', '--reuse', action='store_true', help="Réutilise la dernière image.")
    args = parser.parse_args()
    return args

def is_reuse_arg_present(args):
    return args.reuse




