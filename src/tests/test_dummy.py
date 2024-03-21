#!/usr/bin/env python3
# main testing routine

import sys
from pathlib import Path
import numpy as np

src_dir = Path(__file__).resolve().parents[1]

# Add the 'src' directory to sys.path to make the modules importable
sys.path.append(str(src_dir))

from data import generate_polynomial_data


def test_generate_polynomial_data():
    data = generate_polynomial_data(10, np.linspace(0, 1, 21))
    assert len(data) == 10
    assert len(data[0]) == 2
    assert data[0][0].shape == (21,)
    assert data[0][1].shape == (21,)
