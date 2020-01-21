import  json
from tkinter import filedialog


inputPath = filedialog.askopenfilename()
jsonStr = open(inputPath).read()
data_json = json.loads(jsonStr)

print("len:",len(data_json))