:: setting up a virtual environment
python -m venv venv

:: activate the virtual environment
CALL .\venv\Scripts\activate.bat

:: install requirements from requirements file
python -m pip install --upgrade pip
:: Would need to enforce python version to use requirements
:: python -m pip install -r requirements.txt 
python -m pip install pytest matplotlib pandas plotly open-source-risk-engine jupyter

PAUSE
