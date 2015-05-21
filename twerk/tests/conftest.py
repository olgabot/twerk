import string

import numpy as np
import pandas as pd
import pytest

@pytest.fixture
def X():
    nrow = 20
    ncol = 10
    data = np.random.randn(nrow*ncol).reshape(nrow, ncol)
    index = list(string.ascii_lowercase[:nrow])
    columns = list(string.ascii_uppercase[:ncol])
    data = pd.DataFrame(data, index=index, columns=columns)
    data.ix[:nrow/2, :ncol/2] += 1
    data.ix[:nrow/2, ncol/2:] -= 1
    return data

@pytest.fixture(params=[None, 'spearman', 'pearson', 'kendall', 'bicor'])
def correlation(request):
    return request.param