#!./.venv/bin/python
from utils.transform_batch import transform_batch
from utils.plot_utils import plot_transformations
from utils.output import Output
import plantcv.plantcv as pcv
import sys

def main():
    output = Output.from_argv()

    if output.mode == 'plot':
        try:
            img = pcv.readimage(output.src)[0]
        except Exception as e:
            print(f"Cannot open image: {output.src}")
            sys.exit(1)

        plot_transformations(img)
    
    elif output.mode == 'save':
        transform_batch(output.src, output.dst, output.filter)
            
if __name__ == "__main__":
    main()