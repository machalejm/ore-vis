@echo off
:: load virtual environment
CALL .\venv\Scripts\activate.bat

python -m jupyter lab

PAUSE