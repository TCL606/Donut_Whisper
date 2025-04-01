import json
import os

video_root = "/mnt/bn/tiktok-mm-2/aiic/public/data/video/training/how2/how2"
audio_root = "/mnt/bn/tiktok-mm-2/aiic/public/data/audio/training/how2"
text_txt = "/mnt/bn/tiktok-mm-2/aiic/public/data/video/training/how2/how2_annotation/train/text.id.en"
seg_txt = "/mnt/bn/tiktok-mm-2/aiic/public/data/video/training/how2/how2_annotation/train/segments"
output_json = "/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/Donut_Whisper/jsons/how2_train.json"

vfiles = os.listdir(video_root)
vfdic = dict(zip(vfiles, [1] * len(vfiles)))

afiles = os.listdir(audio_root)
afdic = dict(zip(afiles, [1] * len(afiles)))

v_dic = {}
with open(seg_txt, 'r') as fp:
    line = fp.readline().strip()
    while line != "":
        lst = line.split()
        v_dic[lst[0]] = lst
        line = fp.readline().strip()

data = []
with open(text_txt, 'r') as fp:
    line = fp.readline().strip()
    while line != "":
        lst = line.split(" ", 1)
        vid = v_dic[lst[0]][1]
        timestamps = [float(v_dic[lst[0]][2]), float(v_dic[lst[0]][3])]

        if vid + ".mp4" in vfdic and vid + ".wav" in afdic:
            data.append({
                "audio": os.path.join(audio_root, vid + ".wav"),
                "video": os.path.join(video_root, vid + ".mp4"),
                "timestamps": timestamps,
                "text": lst[1]
            })

        line = fp.readline().strip()

with open(output_json, 'w') as fp:
    json.dump(data, fp, indent=4)

print(len(data))
print(output_json)
