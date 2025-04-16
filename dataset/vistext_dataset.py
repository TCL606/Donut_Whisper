from torch.utils.data import Dataset
import numpy as np
import torch
import os
import json
from torch.nn.utils.rnn import pad_sequence
import soundfile as sf
from decord import VideoReader, cpu
import cv2
from transformers import DefaultDataCollator
import random
import bytedtos
import io

def get_tos_client():
    # try:
    ak = 'D05VZ7IA3E3LYJW6VUUV'
    bucket_name = 'tiktok-maas-us'
    tos_psm = 'toutiao.tos.tosapi'
    tos_cluster = 'default'
    tos_idc = 'maliva'
    cli = bytedtos.Client(
        bucket_name,
        ak,
        service=tos_psm,
        cluster=tos_cluster,
        idc=tos_idc,
        timeout=60,
        connect_timeout=10,
        connection_pool_size=16,
    )
    #     obj = cli.get_object("academic_source/Charades/RW587.mp4")
    #     print("Use VA TOS")
    # except:
    #     cli = bytedtos.Client('tiktok-maas-be1a', 'AG0KU2PWT1FBA1R8VME4', idc='be1a', timeout=60, connect_timeout=10)
    #     obj = cli.get_object("academic_source/Charades/RW587.mp4")
    #     print("Use BE TOS")
    return cli

class VistextDataset(Dataset):
    def __init__(self, data_path, image_processor, wav_processor):
        self.data_path = data_path
        self.image_processor = image_processor
        self.wav_processor = wav_processor
        with open(self.data_path, 'r') as fp:
            self.data = json.load(fp)
        self.cli = get_tos_client()

    def __len__(self):
        return len(self.data)
    
    @property
    def lengths(self):
        length_list = []
        for sample in self.data:
            length_list.append(len(sample["text"]))
        return length_list

    def __getitem__(self, i):
        try:
            source = self.data[i]

            if "audio" in source:
                audio_file = source["audio"]            
                audio, sr = sf.read(audio_file)
            else:
                audio_file = source["tos_audio"]
                for _ in range(10):
                    try:
                        resp = self.cli.get_object(audio_file)
                        assert len(resp.data) == int(resp.headers['Content-Length'])
                        audio_data = resp.data
                    except:
                        continue
                    break
                audio, sr = sf.read(io.BytesIO(audio_data))

            if len(audio.shape) >= 2:
                audio = audio[:, 0]

            text = source["text"]

            if "video" in source or "tos_video" in source:
                if "video" in source:
                    video_file = source["video"]
                    vr = VideoReader(video_file, ctx=cpu(i % 8), num_threads=1)
                else:
                    video_file = source["tos_video"]
                    for _ in range(10):
                        try:
                            resp = self.cli.get_object(video_file)
                            assert len(resp.data) == int(resp.headers['Content-Length'])
                            video_data = resp.data
                        except:
                            continue
                        break
                    vr = VideoReader(io.BytesIO(video_data), ctx=cpu(i % 8), num_threads=1)

                if "image_cnt" in source:
                    num_images = source["image_cnt"]
                else:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.8
                    thickness = 2
                    width = vr[0].shape[1]

                    flag = False
                    for _ in range(2):
                        words = text.split()
                        lines = []
                        current_line = ""
                        for word in words:
                            test_text = current_line + " " + word if current_line else word
                            text_size, _ = cv2.getTextSize(test_text, font, font_scale, thickness)
                            text_width = text_size[0]
                            if text_width <= width:
                                current_line = test_text
                            else:
                                if current_line:
                                    lines.append(current_line)
                                current_line = word
                        if current_line:
                            lines.append(current_line)
                        num_images = len(lines)
                        if num_images > 20:
                            font_scale = 0.5
                            thickness = 1
                        else:
                            flag = True
                            break   
                    
                    try:
                        assert flag                     
                    except Exception as e:
                        print(f"GGG: Num Images: {num_images}, Text: {text}")
                        raise e

                if "timestamps" in source:
                    start, end = source["timestamps"]
                    # audio = audio[int(start * sr): int(end * sr)]

                    total_frame_num = len(vr)
                    ori_fps = vr.get_avg_fps()
                    start = round(start * ori_fps)
                    end = round(end * ori_fps)
                    end = min(end, total_frame_num - 1)

                    frame_idx = np.linspace(start, end, num_images + 2, dtype=int).tolist()[1:-1]
                    images = vr.get_batch(frame_idx).asnumpy()
                    data_id = "['{}', '{}', '{}']".format(audio_file, video_file, source["timestamps"])

                else:
                    total_frame_num = len(vr)
                    frame_idx = np.linspace(0, total_frame_num - 1, num_images + 2, dtype=int).tolist()[1:-1]
                    images = vr.get_batch(frame_idx).asnumpy()
                    data_id = "['{}', '{}', '{}']".format(audio_file, video_file, None)

                images_with_text = []
                if "image_cnt" not in source:
                    for i in range(num_images):
                        new_image = images[i]
                        new_image = new_image / 255 * 2 - 1

                        current_text = lines[i]

                        text_size, baseline = cv2.getTextSize(current_text, font, font_scale, thickness)
                        text_width = text_size[0]
                        text_height = text_size[1]
                        text_x = int((width - text_width) / 2)
                        text_y = new_image.shape[0] - 30

                        padding = 5
                        x1 = max(text_x - padding, 0)
                        y1 = text_y + baseline - text_height - 2 * padding
                        x2 = min(text_x + text_width + padding, width)
                        y2 = text_y + baseline

                        new_image[y1:y2, x1:x2] = -1.0

                        cv2.putText(new_image, current_text, (text_x, text_y), font, font_scale, (1, 1, 1), thickness)

                        new_image = (new_image + 1) / 2 * 255
                        new_image = new_image.astype(np.uint8)
                        images_with_text.append(new_image)

                        # from PIL import Image
                        # image = Image.fromarray(new_image)
                        # image.save("/opt/tiger/tmp.png")
                        # breakpoint()
                else:
                    for i in range(num_images):
                        new_image = images[i]
                        images_with_text.append(new_image)

                images = []
                for img in images_with_text:
                    images.append(self.image_processor(img)["pixel_values"][0])
                images = np.stack(images)

            else:
                data_id = "['{}', '{}', '{}']".format(audio_file, None, None)

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2
                width = 1280

                words = text.split()
                lines = []
                current_line = ""
                for word in words:
                    test_text = current_line + " " + word if current_line else word
                    text_size, _ = cv2.getTextSize(test_text, font, font_scale, thickness)
                    text_width = text_size[0]
                    if text_width <= width:
                        current_line = test_text
                    else:
                        if current_line:
                            lines.append(current_line)
                        current_line = word
                if current_line:
                    lines.append(current_line)
                num_images = len(lines)

                images_with_text = []
                for i in range(num_images):
                    new_image = np.zeros((960, 1280, 3), dtype=np.uint8)
                    new_image = new_image / 255 * 2 - 1

                    current_text = lines[i]

                    text_size, baseline = cv2.getTextSize(current_text, font, font_scale, thickness)
                    text_width = text_size[0]
                    text_height = text_size[1]
                    text_x = int((width - text_width) / 2)
                    text_y = new_image.shape[0] - 30

                    padding = 5
                    x1 = max(text_x - padding, 0)
                    y1 = text_y + baseline - text_height - 2 * padding
                    x2 = min(text_x + text_width + padding, width)
                    y2 = text_y + baseline

                    new_image[y1:y2, x1:x2] = -1.0

                    cv2.putText(new_image, current_text, (text_x, text_y), font, font_scale, (1, 1, 1), thickness)

                    new_image = (new_image + 1) / 2 * 255
                    new_image = new_image.astype(np.uint8)
                    images_with_text.append(new_image)

                    # from PIL import Image
                    # image = Image.fromarray(new_image)
                    # image.save("/opt/tiger/tmp.png")
                    # breakpoint()

                images = []
                for img in images_with_text:
                    images.append(self.image_processor(img)["pixel_values"][0])
                images = np.stack(images)

            spectrogram = self.wav_processor(audio, sampling_rate=sr, return_tensors="pt")["input_features"].squeeze()

            batch = dict(
                audios=audio,
                spectrograms=spectrogram,
                images=images,
                texts=text,
                data_ids=data_id,
            )
            return batch

        except Exception as e:
            print(f'GGGG {i}. Line: {e.__traceback__.tb_lineno}, Exception:', e)
            return self.__getitem__(random.choice(range(len(self))))

class VistextDataCollator(DefaultDataCollator):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def __call__(self, samples):
        audios = [torch.from_numpy(s["audios"]).to(torch.float16) for s in samples]
        spectrograms = [s["spectrograms"] for s in samples]
        spectrograms = torch.stack(spectrograms).to(torch.float16)

        images = [torch.from_numpy(s["images"]) for s in samples]
        images_len = [it.size(0) for it in images]
        images = torch.cat(images, dim=0).to(torch.float16)
    
        texts = [s['texts'] for s in samples]
        # labels = [torch.tensor(self.tokenizer.encode(t), dtype=torch.int64) for t in texts]
        # labels = pad_sequence(labels, batch_first=True, padding_value=self.tokenizer.eot)
        
        # labels = self.tokenizer(texts, return_tensors="pt", padding=True).input_ids
        labels = [self.tokenizer(s['texts'], return_tensors="pt").input_ids.squeeze() for s in samples]
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        # print(labels)
        data_ids = [s["data_ids"] for s in samples]

        batch = {"audios": audios, "spectrograms": spectrograms, "images": images, "labels": labels, "texts": texts, "data_ids": data_ids, "images_len": images_len}
        return batch

class AudioProcessor(object):
    def __init__(self, n_mels):
        self.n_mels = n_mels

    def __call__(self, audio):
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio.astype(np.float32), n_mels=self.n_mels)
        return mel

if __name__ == "__main__":
    from transformers import DonutProcessor
    image_processor = DonutProcessor.from_pretrained("/mnt/bn/tiktok-mm-4/aiic/public/model/donut-base-finetuned-cord-v2")
    
    from transformers import WhisperTokenizer, WhisperModel, WhisperFeatureExtractor

    whisper_model = WhisperModel.from_pretrained("/mnt/bn/tiktok-mm-4/aiic/public/model/whisper-base")
    wav_processor = WhisperFeatureExtractor.from_pretrained("/mnt/bn/tiktok-mm-4/aiic/public/model/whisper-base")
    tokenizer = WhisperTokenizer.from_pretrained("/mnt/bn/tiktok-mm-4/aiic/public/model/whisper-base", multilingual=False, language="en", task='transcribe')
    dataset = VistextDataset("/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/Donut_Whisper/jsons/how2_val.json", image_processor, wav_processor)

    col = VistextDataCollator(tokenizer)
    items = col([dataset[0], dataset[1]])

    breakpoint()