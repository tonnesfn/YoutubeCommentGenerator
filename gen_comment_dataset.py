import get_top_videos
import get_comments_of_video

number_of_top_videos = 200

top_videos_to_get = get_top_videos.get_top_videos(number_of_top_videos)

# no_comments, already_exists, saved, disabled_comments, other_http_error
statuses = []

for i in range(len(top_videos_to_get)):
    current_status = get_comments_of_video.save_youtube_comments(top_videos_to_get[i])

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
