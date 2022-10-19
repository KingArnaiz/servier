import logging

from servier.parsing import parse_train_args
from servier.train import cross_validate
from servier.utils import set_logger


# Initialize logger
logger = logging.getLogger('train')
logger.setLevel(logging.DEBUG)
logger.propagate = False

if __name__ == '__main__':
    args = parse_train_args()
    set_logger(logger, args.save_dir, args.quiet)
    cross_validate(args, logger)
