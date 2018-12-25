from argparse import ArgumentParser
from errno import EEXIST
from errno import ENOENT
from os import makedirs
from os import chdir
from os.path import abspath
from os.path import join
from shlex import split
from shutil import rmtree
from subprocess import call
from subprocess import check_call
from subprocess import check_output


def main(argv=None):
    """ Script execution.

    """

    args = _args(argv)
    if args.ssh:
        rsync = "rsync -avz --delete --exclude=cmake-build-* {:s} {:s}:{:s} --filter='P build'".format(args.origin, args.ssh, args.root)
        print(rsync)
        check_call(split(rsync))

    return 0


def _args(argv=None):
    """ Parse command line arguments.

    By default, sys.argv is parsed.

    """
    # The -h/--help option is defined automatically by argparse.
    parser = ArgumentParser()
    parser.add_argument("-S", "--ssh", help="connect to SSH remote host")
    parser.add_argument("origin", help="local path to project origin")
    parser.add_argument("root", help="remote path to project root")
    return parser.parse_args(argv)

# Make the script executable.

if __name__ == "__main__":
    raise SystemExit(main())
