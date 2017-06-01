import sys
import numpy as np

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print "RUN: python {} <alpha> <input> <train> <valid>".format(sys.argv[0])
    else:
        f_train = open(sys.argv[3], "w")
        f_valid = open(sys.argv[4], "w")

        n_lines = sum(1 for line in open(sys.argv[2], "r"))
        alpha = float(sys.argv[1])
        
        indicies = set(np.random.choice(n_lines, int(n_lines * alpha)))
        with open(sys.argv[2], "r") as f_read:
            for line_i, line in enumerate(f_read):
                f_write = f_train if line_i not in indicies else f_valid
                f_write.write(line)

        f_train.close()
        f_valid.close()
