# Using Supervised Machine-learning Techniques to Identify Objects Classes in Images with Depth Data


## Coarse Layout of Repository

- use ``run.slum`` to run on Balena, University of Bath's HPC.
- use ``python -h <script_name.py>`` to see the list of parameters that it accepts

### ``data/``
- contains two scripts - ``extract-data.py`` and ``transform_data.py``
- ``extract-data.py`` extracts data from the MatLab ``.mat`` data file, containing the NYU Dataset
- ``transform_data.py`` converts data into feature vector formats, e.g. extract patches within an image

### ``model/``
- contains three scripts - ``test_model.py``, ``model.py`` and ``prediction.py``
- ``test_model.py`` is used to test that the required packages are in place
- ``model.py`` runs AdaBoost, SVM or Random Forest according to command line arguments
- ``prediction.py`` runs predictions given a model, provides a precision_recall report and the ability to construct an image based on the prediction


### ``dissertation/``
- contains final dissertation write-up done with LaTeX
