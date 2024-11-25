import argparse

def init_arg_parser():
    parser = argparse.ArgumentParser(description="Execute la fusion d'expostion")
    parser.add_argument('-r', '--reuse', action='store_true', help="Réutilise la dernière image.")
    parser.add_argument('-e', '--exposition', action='store_true', help="Montre différente changement d'exposition ou transformation.")
    args = parser.parse_args()
    return args




