# todo: add support for overwriting files (/w saving of files only at the end)


import os
import os.path

import youtube_common
from googleapiclient import errors

def findkeys(node, kv):
    if isinstance(node, list):
        for i in node:
            for x in findkeys(i, kv):
               yield x
    elif isinstance(node, dict):
        if kv in node:
            yield node[kv]
        for j in node.values():
            for x in findkeys(j, kv):
                yield x


def save_youtube_comments(videoId):

    if os.path.exists('comments/' + videoId + '.txt'):
        print(videoId + ' already exists!')
    else:
        file = open('comments/' + videoId + '.txt', 'w')

        try:
            comments = youtube_common.comment_threads_list_by_video_id(youtube_common.service,
                                                                       part='snippet,replies',
                                                                       videoId=videoId)

            last_page_token = (comments.get('nextPageToken', None))
            comments = list(findkeys(comments, 'textDisplay'))

            file.write('\n'.join(comments))

            while last_page_token is not None:
                comments = youtube_common.comment_threads_list_by_video_id(youtube_common.service,
                                                                           part='snippet,replies',
                                                                           videoId=videoId,
                                                                           pageToken=last_page_token)

                last_page_token = (comments.get('nextPageToken', None))
                comments = list(findkeys(comments, 'textDisplay'))

                file.write('\n'.join(comments))

            file.close()

            print(videoId + ' finished!')

        except errors.HttpError as err:
            print('Error! Deleting file: ' + 'comments/' + videoId + '.txt')
            os.remove('comments/' + videoId + '.txt')