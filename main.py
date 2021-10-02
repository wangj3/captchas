# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import re
import csv
import shutil

import operator
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s

def alphanum_key(s):
    return [tryint(c) for c in re.split('([0-9]+)', s)]

def sort_nicely(l):
    return sorted(l, key=alphanum_key)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # file = ["1c2ard.txt", "card.txt" , "1c1ard.txt", "52card.txt", "badcard.txt"]
    # print(sort_nicely(file))

    # src = r'stufftest.txt'
    # dst =  r'stufftest.csv'
    # shutil.copyfile(src, dst)
    reader = csv.reader(open("stufftest.csv"), delimiter=";")
    sortedlist = sorted(reader, key=lambda row: row[0])
    l=[]
    for plate in sortedlist:
        plate[0]=plate[0].replace(" ", "")
        plate[0]=plate[0].replace("'", "")
        l.append(plate)

    print(l)
    f = open('stuff.csv', 'w')
    with f:
        write = csv.writer(f)
        write.writerows(l)

    with open('stuff.csv', "r+", encoding="utf-8") as csv_file:
        content = csv_file.read()

    with open('stuff.csv', "w+", encoding="utf-8") as csv_file:
        csv_file.write(content.replace('"', ''))

    # Open the file with a context manager
    # with open("stuff.csv", "a+") as myfile:
    #     # Convert all of the items in lst to strings (for str.join)
    #     lst = map(str, sortedlist)
    #     # Join the items together with commas
    #     line = ",".join(lst)+"\n"
    #     # Write to the file
    #     myfile.write(line)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
