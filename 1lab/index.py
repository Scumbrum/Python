
# task 1
with open('text3.txt', 'r') as fp:
    data = fp.read()
    sliceData = data[:30]
    print(len(sliceData))
    print(sliceData.count("a"))
    print(sliceData.find("a"))
    print(sliceData.index("was"))
    print(sliceData.upper())
    print(sliceData.lower())
    print(sliceData.title())
    print(sliceData.capitalize())
    print("*".join(sliceData))
    print(sliceData.isalnum())
    print(sliceData.isalpha())
    print(sliceData.isdigit())
    print(sliceData.isspace())
    print(sliceData.endswith("r"))
    print(sliceData.startswith("R"))
    fp.close()

# task 2

import re

def remove_vowels_and_digits(string):
    regex = re.compile(r'[^aeiouAEIOU124567890]')
    result = ""
    for char in string:
        if regex.match(char):
            result += char

    return result

with open('text3.txt', 'r') as fp:
    data = fp.read()
    print(data)
    print("<========>")
    print(remove_vowels_and_digits(data))
    fp.close()
