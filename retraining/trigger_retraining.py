import subprocess

def trigger_retraining():

    print("Retraining pipeline triggered")

    subprocess.run(["python","main.py"])