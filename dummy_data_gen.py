import csv

import numpy as np
import tqdm

np.random.seed()


A1 = np.random.randint(-100, 100) + np.random.rand()
A2 = np.random.randint(-100, 100) + np.random.rand()
A3 = np.random.randint(-100, 100) + np.random.rand()
A4 = np.random.randint(-100, 100) + np.random.rand()
A5 = np.random.randint(-100, 100) + np.random.rand()
N = int(10e4)


def res(x1, x2, x3, x4, x5):
    return x1**5 * A1 + x4 * A4 + np.float_power(x2, 2.57429343) * A2 + x3**2 * A3 + \
        x4 * A4 + np.float_power(x5, 1.543) * A5 + np.random.normal(0, 1e-3)


if __name__ == '__main__':
    results = []

    for i in tqdm.trange(N):
        d = {
            'id': i + 1,
            'feat1': np.random.uniform(1, 50),
            'feat2': np.random.uniform(1, 50),
            'feat3': np.random.uniform(1, 50),
            'feat4': np.random.uniform(1, 50),
            'feat5': np.random.uniform(1, 50)
        }

        d['ret'] = res(d['feat1'], d['feat2'],
                       d['feat3'], d['feat4'], d['feat5'])

        results.append(d)

    with open('out.csv', 'w+') as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=['id', 'feat1', 'feat2',
                        'feat3', 'feat4', 'feat5', 'ret']
        )
        writer.writeheader()
        writer.writerows(results)
