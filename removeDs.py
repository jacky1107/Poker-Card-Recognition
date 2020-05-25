import os

c = os.getcwd()
print(os.listdir(c))
for data in os.listdir(c):
    apth = os.path.join(c, data)
    if "DS_Store" in data:
        os.remove(apth)