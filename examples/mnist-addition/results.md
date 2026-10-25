MNIST addition, exact BP, 5 epoch(s) over all 60k training images, batch 32 sums, lr 0.001, seed 0, accelerator mps.

| N | training sums | test sums | digit accuracy | sum accuracy | digit acc.^(2N) | training time |
|---|---|---|---|---|---|---|
| 1 | 30000 | 5000 | 0.9901 | 0.9802 | 0.9803 | 84s |
| 2 | 15000 | 2500 | 0.9889 | 0.9560 | 0.9563 | 66s |
| 4 | 7500 | 1250 | 0.9882 | 0.9104 | 0.9094 | 44s |
| 15 | 2000 | 333 | 0.9860 | 0.6517 | 0.6548 | 27s |
