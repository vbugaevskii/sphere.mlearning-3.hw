import sys
import time
import datetime


def make_timestamp(s):
    return int(time.mktime(datetime.datetime.strptime(s, "%Y-%m-%d").timetuple()))


def modify_string(line):
    user, item, time = line.strip().split(',')
    time = str(make_timestamp(time))
    return ','.join([user, item, time, "0"])


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print "RUN: python {} <input_file> <output_file>".format(sys.argv[0])
    else:
        f_write = open(sys.argv[2], "w")

        with open(sys.argv[1], "r") as f_read:
            for line in f_read:
                f_write.write(modify_string(line))
                f_write.write("\n")

        f_write.close()
