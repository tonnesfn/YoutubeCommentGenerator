import math
import os
import os.path
import re
import sys
import unicodedata
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
    given_comment = re.sub(' +', ' ',given_comment)  # Remove double spaces
    given_comment = given_comment.replace(' \n', '\n').replace('\n ', '\n')  # Remove leading and trailing spaces
    given_comment = unicodedata.normalize('NFKD', given_comment).encode('ascii', 'ignore').decode('utf-8')  # Filter out unicode
    given_comment = re.sub(u'(?imu)^\s*\n', u'', given_comment)  # Remove empty lines

    return given_comment


def print_counter(counter):
    if counter != 0:
        print('\b' * 8, end='')
    print('{}'.format(counter).ljust(8), end='')
    sys.stdout.flush()

    return counter + 1


# Returns status: [already_exists, no_comments, saved, disabled_comments, other_http_error]
def save_youtube_comments(video_id, skip_existing=True):
    counter = 0;

    if os.path.exists('data/raw_comments/' + video_id + '.txt') and skip_existing:
        return [1, 0, 0, 0, 0]
    else:

        try:
            counter = print_counter(counter)

            comments = youtube_common.comment_threads_list_by_video_id(youtube_common.service,
                                                                       part='snippet,replies',
                                                                       videoId=video_id)

            last_page_token = (comments.get('nextPageToken', None))
            comments = list(findkeys(comments, 'textDisplay'))

            if len(comments) == 0:
                return [0, 1, 0, 0, 0]
            else:

                file = open('data/raw_comments/' + video_id + '.txt', 'w')
                file.write(sanitize_comment('\n'.join(comments)))

                while last_page_token is not None:
                    counter = print_counter(counter)

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


def videos_list_most_popular(service, **kwargs):
    kwargs = youtube_common.remove_empty_kwargs(**kwargs)  # See full sample for function
    results = service.videos().list(
      **kwargs
    ).execute()

    return results


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


def get_top_videos(number_of_videos):

    results = videos_list_most_popular(youtube_common.service,
                                       part='snippet,contentDetails,statistics',
                                       chart='mostPopular',
                                       regionCode='US',
                                       videoCategoryId='',
                                       maxResults=50)

    ids = list(findkeys(results, 'id'))

    for _ in range(int(math.floor((number_of_videos/50)))-1):
        last_page_token = (results.get('nextPageToken', None))

        results = videos_list_most_popular(youtube_common.service,
                                           part='snippet,contentDetails,statistics',
                                           chart='mostPopular',
                                           regionCode='US',
                                           videoCategoryId='',
                                           maxResults=50,
                                           pageToken=last_page_token)

        ids = ids + list(findkeys(results, 'id'))

    return ids


# todo: make this into an argument passed at program launch
skip_existing = True

number_of_top_videos = 200

top_videos_to_get = get_top_videos(number_of_top_videos)

# no_comments, already_exists, saved, disabled_comments, other_http_error
statuses = []

if skip_existing:
    print('Starting crawling of top {} comments and skipping comments from existing videos'.format(number_of_top_videos))
else:
    print('Starting crawling of top {} comments and replacing existing comments'.format(number_of_top_videos))

for i in range(len(top_videos_to_get)):
    print('\r    - Saving comments from video {} of {}: page '.format(i+1, len(top_videos_to_get)), end='')
    current_status = save_youtube_comments(top_videos_to_get[i], skip_existing)

    if len(statuses) == 0:
        statuses = current_status
    else:
        statuses = [sum(x) for x in zip(statuses, current_status)]

# Returns status: [already_exists, no_comments, saved, disabled_comments, other_http_error]
print('Comments from {}/{} videos downloaded'.format(statuses[2],sum(statuses)))
print('  {} were already downloded'.format(statuses[0]))
print('  {} had no comments'.format(statuses[1]))
print('  {} had disabled comments'.format(statuses[3]))
print('  {} other http errors'.format(statuses[4]))
