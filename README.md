

The pipeline generates reports, visualizations, and alerts in the `outputs/` folder.

Also  unzip the data set please 

\## Setup \& Installation

Open your terminal in the project folder and run these commands:


python -m venv venv

source venv/bin/activate (if on Linux)

venv\\Scripts\\activate (if on PowerShell)

pip install -r requirements.txt

python building_hybrid_pipeline_v2.py


\##OR

You can just open the .ipynb file and run the entirety of the code, but we prefer that you use the terminal since that is how we worked on our project.


\##Project Structure



.

├── building\_hybrid\_pipeline(v1).py

├── building\_hybrid\_pipeline\_v2.py

├── outputs

│   ├── alerts\_log.csv

│   ├── all\_scores.csv

│   ├── comparison\_agreement.png

│   ├── comparison\_report.csv

│   ├── comparison\_roc.png

│   ├── comparison\_summary.txt

│   ├── filtered\_rules.csv

│   └── static\_rules.csv

├── pattern\_stability\_analysis.py

├── \_\_pycache\_\_

│   └── unsupervised\_comparison.cpython-313.pyc

├── readMe.md

├── requirements.txt

├── sensitivity\_analysis.py

├── simulation\_data\_multi\_prev\_test.csv

├── simulation\_data\_multi\_prev\_train.csv

├── smhi-july-23-29-2018.csv

├── unsupervised\_comparison.py

└── venv




