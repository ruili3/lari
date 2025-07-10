from src.testing import get_args_parser, test

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    test(args)
