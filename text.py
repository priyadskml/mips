import pandas as pd
import numpy as np


users = []
values = np.zeros((480189, 17770), dtype='float32')
df = pd.DataFrame(data=values)

for x in range(0, 17770):
    name = "%07d" % (x+1,)
    name = (str)(name)
    name = "training_set/mv_"+name+".txt"
    f = open(name, "r")
    p = f.read()
    lines = p.split('\n')
    movie_=lines.pop(0)
    movie_=(int)(movie_[0:-1])
    if (movie_==x+1):
        print(x+1)
    for l in lines:
        if (l != ''):
            line1=l.split(',')
            print(line1[0])
            if (int)(line1[0]) not in users:
                users.append((int)(line1[0]))
        
            df[x][users.index((int)(line1[0]))] = (float)(line1[1])
    f.close()

    
    
from sparse import io
print len(users)



io.mmwrite("Ratings_nf", df, field='real', precision=4)


