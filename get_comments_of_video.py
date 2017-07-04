import os
import os.path
import re

import html
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


def sanitize_comment(given_comment):
    given_comment = html.unescape(given_comment)  # Remove HTML characters
    given_comment = re.sub('<[^<]+>', "", given_comment)  # Remove XML
    given_comment = re.sub(r'[^\x00-\x7f]', r' ', given_comment)  # Only ascii
    given_comment = re.sub(' +', ' ',given_comment)  # Remove double spaces
    given_comment = re.sub(u'(?imu)^\s*\n', u'', given_comment)  # Remove empty lines
    given_comment = given_comment.replace(' \n', '\n').replace('\n ', '\n')  # Remove leading and trailing spaces

    return given_comment


# Returns status: [already_exists, no_comments, saved, disabled_comments, other_http_error]
def save_youtube_comments(video_id, skip_existing=True):

    if os.path.exists('comments_raw/' + video_id + '.txt') and skip_existing:
        return [1, 0, 0, 0, 0]
    else:

        try:
            comments = youtube_common.comment_threads_list_by_video_id(youtube_common.service,
                                                                       part='snippet,replies',
                                                                       videoId=video_id)

            last_page_token = (comments.get('nextPageToken', None))
            comments = list(findkeys(comments, 'textDisplay'))

            if len(comments) == 0:
                return [0, 1, 0, 0, 0]
            else:

                file = open('comments_raw/' + video_id + '.txt', 'w')
                file.write(sanitize_comment('\n'.join(comments)))

                while last_page_token is not None:
                    comments = youtube_common.comment_threads_list_by_video_id(youtube_common.service,
                                                                               part='snippet,replies',
                                                                               videoId=video_id,
                                                                               pageToken=last_page_token)

                    last_page_token = (comments.get('nextPageToken', None))
                    comments = list(findkeys(comments, 'textDisplay'))

                    file.write(sanitize_comment('\n'.join(comments)))

                file.close()

                return [0, 0, 1, 0, 0]

        except errors.HttpError as err:
            if err.__str__().find('disabled comments'):
                return [0, 0, 0, 1, 0]
            else:
                print('Error!' + err.__str__())
                return [0, 0, 0, 0, 1]
