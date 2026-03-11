from typing import Literal
from pathlib import Path
import sys

class Output:
    Mode = Literal['plot', 'save']
    filters = ['blur', 'mask', 'roi', 'analyse', 'landmarks', 'histogram']
    Filter = Literal['mask', 'blur', 'roi', 'analyse', 'landmarks', 'histogram']

    def __init__(self, src, mode: Mode, dst=None, filter=None):
        self.mode: Mode | None = mode
        self.dst: Path | None = None
        self.src: Path | None = None
        self.filter: Filter | None = None

        if mode == 'plot':
            dst = None

            self.src = Path(src)

            if not self.src.exists():
                raise ValueError('File does not exist')

            if not self.src.is_file():
                raise ValueError(f'{src} is not a file.')

        if mode == 'save':
            if not dst:
                raise ValueError('Destination required')
            if not filter:
                raise ValueError('Filter required')

            self.dst = Path(dst)
            self.src = Path(src)
            self.filter: Filter = filter

            if not self.dst.exists():
                raise ValueError('Destination folder does not exist')
            
            if not self.dst.is_dir():
                raise ValueError(f'{dst} is not a directory.')
        
            if not self.src.exists():
                raise ValueError('Source folder does not exist')
            
            if not self.src.is_dir():
                raise ValueError(f'{src} is not a directory.')

            if self.filter not in self.filters:
                raise ValueError(f'Invalid filter. Must be one of {self.filters}')

    @staticmethod
    def help():
        program = sys.argv[0]
        print(f"""Usage:
    Plotting mode:
        {program} <path_to_image>

    Save mode:
        {program} -src <src_folder> -dst <destination_folder> -["mask" | "blur" | "roi" | "analyse" | "landmarks" | "histogram"]""")

    @classmethod
    def from_argv(cls):
        n_args = len(sys.argv) - 1
        args = sys.argv
        
        if n_args == 1:
            arg = args[1]

            if arg in ['-h', '--help']:
                cls.help()
                sys.exit(0)

            try:
                return cls(src=arg, mode='plot')
            except:
                Output.help()
                sys.exit(1)

        elif n_args == 5:
            if '-src' not in args or '-dst' not in args or not any(f"-{filter}" in args for filter in cls.filters):
                Output.help()
                sys.exit(1)

            src_index = args.index('-src')
            dst_index = args.index('-dst')
            filter_indices = [i for i, arg in enumerate(args) if arg[1:] in cls.filters]

            if src_index == -1 or dst_index == -1 or len(filter_indices) == 0:
                Output.help()
                sys.exit(1)
            
            filter_index = filter_indices[0]

            try:
                src = args[src_index + 1]
                dst = args[dst_index + 1]
                filter = args[filter_index][1:]
                return cls(src=src, mode='save', dst=dst, filter=filter)
            except ValueError:
                Output.help()
                sys.exit(1)

        else:
            Output.help()
            sys.exit(1)
