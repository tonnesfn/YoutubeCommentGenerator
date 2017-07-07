import get_top_videos
import get_comments_of_video

# todo: make this into an argument passed at program launch
skip_existing = False

number_of_top_videos = 200

top_videos_to_get = get_top_videos.get_top_videos(number_of_top_videos)

# no_comments, already_exists, saved, disabled_comments, other_http_error
statuses = []

if skip_existing:
    print('Starting crawling of top {} comments and skipping comments from existing videos'.format(number_of_top_videos))
else:
    print('Starting crawling of top {} comments and replacing existing comments'.format(number_of_top_videos))

for i in range(len(top_videos_to_get)):
    print('\r    - Saving comments from video {} of {}: page '.format(i+1, len(top_videos_to_get)), end='')
    current_status = get_comments_of_video.save_youtube_comments(top_videos_to_get[i], skip_existing)

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
