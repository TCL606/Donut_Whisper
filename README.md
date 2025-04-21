# README

## 准备

下载 whisper_base model：https://huggingface.co/openai/whisper-base

下载 donut model：https://huggingface.co/naver-clova-ix/donut-base-finetuned-cord-v2

## 数据格式

储存在 json 文件中，为一个列表，每个元素为一个字典，可包含以下键值对：

~~~json
[
  {
      "audio": "...",
      "video": "...",
      "text": "xxx.",
      "timestamps": [start, end],
      "image_cnt": "3",
  },
  ...
]
~~~

audio 为 audio 路径，video 为 video 路径，text 为训练 label。timestamps 可选，若指定 start 和 end 的时间点，则自动在 video 中对应时间采帧。image_cnt 可选，代表在视频中采多少帧。

## 训练

~~~bash
bash run.sh \
	--whisper_path $WHISPER_PATH \
	--image_model_path $IMAGE_MODEL_PATH \
	--train_data $TRAIN_DATA \
	--eval_data $EVAL_DATA \
	--epochs 10 \
	--train_bs 16 \
	--eval_bs 4 \
	--eval_accumulation_steps 20 \
	--output_name $OUTPUT_NAME \
	--model_type donut_whisper
~~~

## 测试

~~~bash
bash run.sh \
	--whisper_path $WHISPER_PATH \
	--image_model_path $IMAGE_MODEL_PATH \
	--test_data $TEST_DATA \
	--do_test \
	--ckpt $CKPT \
	--model_type donut_whisper \
	--eval_bs 4
~~~

