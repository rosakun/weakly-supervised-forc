## Overview

This repository contains code for testing various models to weakly label a dataset and for the actual labeling process. This work is part of the 2025 Field-of-Research Classification (FoRC) Shared Task which is co-located at SNLP 2025.

## Shared Task Background

The FoRC shared task concerns itself with the automatic classification of scientific research papers by their field of research. We focus on classifying computational linguistics papers, taken from the ACL Anthology, by at least one of a list of [181 hierarchically organized labels](https://github.com/DFKI-NLP/Taxonomy4CL).
The 2025 iteration adds a weakly labeled dataset of over 40,000 papers to the manually labeled dataset of 1500 papers used for [last year's iteration](https://link.springer.com/chapter/10.1007/978-3-031-65794-8_12). 
More details about the shared task can be found [here](https://nfdi4ds.github.io/nslp2025/docs/forc_shared_task.html). 


## Usage

This code can:
* Convert data from [the ACL Anthology corpus](https://github.com/shauryr/ACL-anthology-corpus) into the same format as the FoRC4CL dataset.
* Pre- and postprocess FoRC4CL-format datasets.
* Train and analyse simple ML models on the FoRC4CL train/test split.
* Weakly label the ACL Anthology corpus using the simple models.
* Train and score transformer models on the FoRC4CL train/test split.


## Results

Results are soon to be published in an overview paper.

## Contributions

Contributions are welcome! Please check our [CodaBench](https://www.codabench.org/competitions/5779/) if you'd like to submit a solution!

## Contact

For any questions, please contact maria.francis@dfki.de or open an issue in this repository.

