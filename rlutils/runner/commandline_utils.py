import argparse
import inspect


def get_argparser_from_func(func):
    """ Read the argument of a function and parse it as ArgumentParser. """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    signature = inspect.signature(func)
    for k, v in signature.parameters.items():
        if v.default is inspect.Parameter.empty:
            parser.add_argument('--' + k, help=' ')
        else:
            parser.add_argument('--' + k, type=type(v.default), default=v.default, help=' ')
    return parser


def run_func_as_main(func):
    """ Run function as the main. Put the function arguments into argument parser """
    parser = get_argparser_from_func(func)
    args = vars(parser.parse_args())
    func(**args)
