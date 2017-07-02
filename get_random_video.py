# todo: Check for number of comments before returning

import urllib
from urllib import request
from random import randint


def random_str(str_size):
    res = ""
    for i in range(str_size):
        x = randint(0,25)
        c = chr(ord('a')+x)
        res += c
    return res


def find_watch(text, pos):
    start = text.find("watch?v=", pos)
    if start < 0:
        return None, None
    end = text.find(" ", start)
    if end < 0:
        return None, None

    if end-start > 200:
        return None, None

    return text[start+8:end-1], start


def find_instance_links():
    base_url = 'https://www.youtube.com/results?search_query='
    url = base_url+random_str(3)

    r = urllib.request.urlopen(url).read().decode('utf-8')

    links = {}

    pos = 0
    while True:
        link, pos = find_watch(r,pos)
        if link == None or pos == None:
            break
        pos += 1

        if ";" in link:
            continue
        links[link] = 1

    items_list = list(links.items())

    list_size = len(items_list)
    selected = randint(int(list_size/2), list_size-1)
    return items_list[selected][0]


def get_random_videos(number_of_videos):
    ids = []
    for _ in range(number_of_videos):
        ids.append(find_instance_links())

    return ids

print(get_random_videos(10))
