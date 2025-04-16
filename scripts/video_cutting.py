import json
import os
import re
from tqdm import tqdm
import subprocess
import concurrent.futures

def time_to_float(time_str):
    parts = re.split(r'[:,]', time_str)
    hours = float(parts[0])
    minutes = float(parts[1])
    seconds = float(parts[2])
    milliseconds = float(parts[3])
    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000

srt_root = "/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/Donut_Whisper/Video_Film/SRT_kurz"
video_root = "/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/Donut_Whisper/Video_Film/Video_kurz"
output_dir = "/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/Donut_Whisper/Video_Film/Cut2"

all_srt = {}
srt_files = os.listdir(srt_root)
for srt_file in srt_files:
    with open(os.path.join(srt_root, srt_file), 'r') as file:
        content = file.read()
        entries = content.strip().split('\n\n')
        result = []
        for entry in entries:
            lines = entry.split('\n')
            if len(lines) >= 3:
                time_range = lines[1].split(' --> ')
                start_time = time_to_float(time_range[0])
                end_time = time_to_float(time_range[1])
                pattern = r'<font color="#eba862">(.*?)</font>'
                match = re.search(pattern, ''.join(lines[2:]))
                if match:
                    subtitle = match.group(1)
                    result.append((start_time, end_time, subtitle))
    all_srt[srt_file] = result

for k in all_srt.keys():
    merge_v = []
    v = all_srt[k]
    start, end, text = v[0][0], v[0][1], v[0][2]
    cnt = 1
    for i in range(1, len(v)):
        if not text.endswith(".") and not text.endswith("?") and not text.endswith("!"):
            text += " " + v[i][2]
            end = v[i][1]
            cnt += 1
        else:
            merge_v.append((start, end, text, cnt))
            start, end, text = v[i][0], v[i][1], v[i][2]
            cnt = 1
    merge_v.append((start, end, text, cnt))
    all_srt[k] = merge_v

video_lst = []
time_lst = []
for k, v in all_srt.items():
    video_lst += [os.path.join(video_root, k.replace(".srt", ".mkv"))] * len(v)
    time_lst += v

def process(i):
    start_time, end_time, text, cnt = time_lst[i]
    video_path = video_lst[i]
    video_name = os.path.basename(video_path).replace(".mkv", "")
    os.makedirs(os.path.join(output_dir, "video", video_name), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "audio", video_name), exist_ok=True)

    output_video_path = os.path.join(output_dir, "video", video_name, f"{video_name}-{round(start_time)}-{round(end_time)}.mp4")
    output_audio_path = os.path.join(output_dir, "audio", video_name, f"{video_name}-{round(start_time)}-{round(end_time)}.wav")

    video_cmd = [
        'ffmpeg',
        '-y',
        '-ss', str(start_time),
        '-to', str(end_time),
        '-i', video_path,
        '-c:v', 'libx264',
        '-c:a', 'aac',
        output_video_path
    ]
    # 重定向标准输出和标准错误输出到空设备，避免显示信息
    subprocess.run(video_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    # 提取音频并转换为 16kHz 单声道
    audio_cmd = [
        'ffmpeg',
        '-y',
        '-ss', str(start_time),
        '-to', str(end_time),
        '-i', video_path,
        '-ar', '16000',
        '-ac', '1',
        output_audio_path
    ]
    # 重定向标准输出和标准错误输出到空设备，避免显示信息
    subprocess.run(audio_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    return output_video_path, output_audio_path, text

print(len(time_lst))
with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
    responses = list(tqdm(executor.map(process, range(len(time_lst)))))

print("Done!")

# res = []
# for i in range(len(time_lst)):
#     res.append({
#         "video": output_video_path,
#         "audio": output_audio_path,
#         "text": text,
#         "image_cnt": cnt
#     })

# with open(output_json, 'w') as fp:
#     json.dump(res, fp, indent=4)

# print(len(res))
# print(output_json)