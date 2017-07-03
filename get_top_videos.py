import math

import youtube_common

def videos_list_most_popular(service, **kwargs):
  kwargs = youtube_common.remove_empty_kwargs(**kwargs) # See full sample for function
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

        last_page_token = (results.get('nextPageToken', None))

        ids = ids + list(findkeys(results, 'id'))

    return ids
