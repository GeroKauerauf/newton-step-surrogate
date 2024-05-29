# Author: Gero F. Kauerauf

import logging
# logging.basicConfig(filename='info.log', level=logging.INFO)
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

from .data_generation.generate_data import sample_objective_function
