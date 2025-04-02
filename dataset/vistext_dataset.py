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

class VistextDataset(Dataset):
    def __init__(self, data_path, image_processor, wav_processor):
        self.data_path = data_path
        self.image_processor = image_processor
        self.wav_processor = wav_processor
        with open(self.data_path, 'r') as fp:
            self.data = json.load(fp)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        source = self.data[i]
        if "tos_key" in source:
            pass
        else:
            audio_file = source["audio"]
            # video_file = source["video"]
            text = source["text"]
            
            audio, sr = sf.read(audio_file)
            if len(audio.shape) >= 2:
                audio = audio[:, 0]

            # vr = VideoReader(video_file, ctx=cpu(i % 8), num_threads=1)

            # if "timestamps" in source:
            #     start, end = source["timestamps"]
            #     audio = audio[int(start * sr): int(end * sr)]

            #     total_frame_num = len(vr)
            #     ori_fps = vr.get_avg_fps()
            #     start = round(start * ori_fps)
            #     end = round(end * ori_fps)
            #     end = min(end, total_frame_num - 1)
            #     vidx = (start + end) // 2
                
            #     image = vr[vidx].asnumpy()
            #     image = self.image_processor(image)["pixel_values"][0]

            #     data_id = "['{}', '{}', '{}', '{}']".format(audio_file, video_file, vidx, source["timestamps"])
            # else:
            #     total_frame_num = len(vr)
            #     vidx = (total_frame_num - 1) // 2
            #     image = vr[vidx].asnumpy()
            #     image = np.array(image)
            #     image = self.image_processor(image)["pixel_values"][0]
            #     data_id = "['{}', '{}', '{}']".format(audio_file, video_file, vidx)


        # spectrogram = self.wav_processor(audio)
        spectrogram = self.wav_processor(audio, sampling_rate=sr, return_tensors="pt")["input_features"].squeeze()

        # H, W = image.shape[1], image.shape[2]
        # bottom_part = image[:, H - 128:, :]
        # bottom_part = np.transpose(bottom_part, (1, 2, 0))

        data_id = audio_file
        bottom_part = np.ones((60, 1920, 3), dtype=np.float32) * -1.0
        cv2.putText(bottom_part, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        image = np.transpose(bottom_part, (2, 0, 1))
        image = (image + 1) / 2 * 255
        image = image.astype(np.uint8)
        image = self.image_processor(image)["pixel_values"][0]

        # from PIL import Image
        # image = (image + 1) / 2 * 255
        # image = image.astype(np.uint8).transpose(1, 2, 0)
        # image = Image.fromarray(image)
        # image.save("/opt/tiger/tmp.png")

        batch = dict(
            audios=audio,
            spectrograms=spectrogram,
            images=image,
            texts=text,
            data_ids=data_id,
        )
        return batch

class VistextDataCollator(DefaultDataCollator):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def __call__(self, samples):
        audios = [torch.from_numpy(s["audios"]).to(torch.float16) for s in samples]
        spectrograms = [s["spectrograms"] for s in samples]
        spectrograms = torch.stack(spectrograms).to(torch.float16)

        images = [torch.from_numpy(s["images"]) for s in samples]
        images = torch.stack(images).to(torch.float16)
        
        texts = [s['texts'] for s in samples]
        # labels = [torch.tensor(self.tokenizer.encode(t), dtype=torch.int64) for t in texts]
        # labels = pad_sequence(labels, batch_first=True, padding_value=self.tokenizer.eot)
        
        # labels = self.tokenizer(texts, return_tensors="pt", padding=True).input_ids
        labels = [self.tokenizer(s['texts'], return_tensors="pt").input_ids.squeeze() for s in samples]
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        # print(labels)
        data_ids = [s["data_ids"] for s in samples]

        batch = {"audios": audios, "spectrograms": spectrograms, "images": images, "labels": labels, "texts": texts, "data_ids": data_ids}
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
    a = col([dataset[0], dataset[1]])