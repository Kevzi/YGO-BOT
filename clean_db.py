import os
import time

file_path = 'data/ygo.db'
if os.path.exists(file_path):
    for i in range(10):
        try:
            os.remove(file_path)
            print("Successfully deleted ygo.db")
            break
        except Exception as e:
            print(f"Attempt {i+1} failed: {e}")
            time.sleep(1)
else:
    print("ygo.db does not exist")
