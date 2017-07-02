import get_top_videos
import get_comments_of_video

number_of_top_videos = 1000

top_videos_to_get = get_top_videos.get_top_videos(number_of_top_videos)

print(len(top_videos_to_get))

for i in range(len(top_videos_to_get)):
    get_comments_of_video.save_youtube_comments(top_videos_to_get[i])