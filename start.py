import os

while True:
    key = input()
    if key == 's':
        break

os.system('python3.9 detect.py --weights best_1.onnx --device cpu --source rs')