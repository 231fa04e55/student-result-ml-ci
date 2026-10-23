python -m pip install -r requirements.txt
python train_model.py
python -m unittest discover -v
python quality_gate.py
