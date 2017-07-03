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


def save_youtube_comments(video_id):

    if os.path.exists('comments_raw/' + video_id + '.txt'):
        print(video_id + ' already exists!')
    else:

        try:
            comments = youtube_common.comment_threads_list_by_video_id(youtube_common.service,
                                                                       part='snippet,replies',
                                                                       videoId=video_id)

            last_page_token = (comments.get('nextPageToken', None))
            comments = list(findkeys(comments, 'textDisplay'))

            if (len(comments) == 0):
                print('Len 0 for video ' + video_id)
            else:

                file = open('comments_raw/' + video_id + '.txt', 'w')
                file.write('\n'.join(comments).encode('ascii', 'ignore').decode('unicode_escape'))

                while last_page_token is not None:
                    comments = youtube_common.comment_threads_list_by_video_id(youtube_common.service,
                                                                           part='snippet,replies',
                                                                           videoId=video_id,
                                                                           pageToken=last_page_token)

                    last_page_token = (comments.get('nextPageToken', None))
                    comments = list(findkeys(comments, 'textDisplay'))

                    file.write('\n'.join(comments).encode('ascii', 'ignore').decode('unicode_escape'))

                file.close()

                print(video_id + ' finished!')

        except errors.HttpError as err:
            print('Error!' + err.__str__())

