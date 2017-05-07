import sys

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print "RUN: python {} <input_file> <output_file>".format(sys.argv[0])
    else:
        f_write = open(sys.argv[2], "w")
        f_write.write("StringId,Mark\n")

        with open(sys.argv[1], "r") as f_read:
            for line_i, line in enumerate(f_read):
                line = float(line.strip())
                if line > 5.0:
                    line = 5.0
                elif line < 1.0:
                    line = 1.0
                f_write.write("{},{}\n".format(line_i+1, line))

        f_write.close()
