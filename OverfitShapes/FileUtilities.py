import threading
import os
from glob import glob

def parallelForEachFile(dirpath, function, extensions, parallel_batch_size):
    threads = []
    
    files = []
    for extension in extensions:
        files += [y for x in os.walk(dirpath) for y in glob(os.path.join(x[0], '*'+extension))]

    print("Found " + str(len(files)) + " files")

    for i in range(len(files)):
        x = threading.Thread(target=function, args=(files[i],))
        x.start()
        threads += [x]

        if (i % parallel_batch_size == 0):
            for thread in threads:
                thread.join()
            threads = []
    
    for thread in threads:
        thread.join()
