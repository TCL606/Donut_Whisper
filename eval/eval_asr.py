import json
from wer import WER
from whisper_normalizer import EnglishTextNormalizer

json_file = "/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/Donut_Whisper/output/test/whisper_base_film/results_final.json" # /mnt/bn/tiktok-mm-4/aiic/users/tangchangli/Donut_Whisper/output/test/donut_only_film_ctn11k_250415_film_11k/results_final.json" # output/test/donut_whisper_film_ctn11k_250415_film_11k/results_final.json" # output/test/donut_whisper_multiImg_how2vd_LS960_comV_250414_film_11k/results_final.json" # output/test/donut_only_multiImg_how2vd_LS960_comV_250414_film_11k/results_final.json" # output/test/donut_only_multiImg_how2vd_LS960_comV_250414_film_11k/results_final.json" # output/test/donut_whisper_multiImg_how2vd_LS960_comV_250414_LStestclean_11k/results_final.json" # output/test/donut_only_multiImg_how2vd_LS960_comV_250414_LStestclean_11k/results_final.json" # output/test/donut_only_multiImg_how2vd_LS960_comV_250414_how2val_11k/results_final.json" # output/test/donut_whisper_multiImg_how2vd_LS960_comV_250414_how2val_11k/results_final.json" # output/test/whisper_base_m3avTest/results_final.json" # "/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/Donut_Whisper/output/test/whisper_base_shotval/results_final.json" # "/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/Donut_Whisper/output/test/donut_only_how2_LS960_comV_ep5_how2val_13k/results_final.json" # output/test/whisper_only_how2_LS960_comV_ep5_how2val_13k/results_final.json" # output/test/debug/results_final.json" # Donut_Whisper/output/test/debug/results_final.json" # output/test/donut_whisper_how2_LS960_comV_ep5_how2val_13k/results_final.json" # output/test/donut_whisper_how2_LS960_comV_ep5_how2val_7k/results_final.json" # output/test/donut_whisper_how2_LS960_comV_ep5_how2val_3k/results_final.json" # output/test/donut_whisper_ep20_lr1e5_how2val_9k5/results_final.json" # output/test/donut_whisper_ep20_how2val_6k5/results_final.json" # output/test/donut_whisper_attn_ep20_how2val_5k5/results_final.json" # output/test/donut-base-how2val/results_final.json" # "/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/Donut_Whisper/output/test/donut_whisper_attn_how2val_2k/results_final.json" # output/test/whisper_base_how2val/results_final.json" # "/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/Donut_Whisper/output/test/run_base_donut_10epo_how2val_1k/results_final.json"

with open(json_file, 'r') as fp:
    data = json.load(fp)

criterion = WER()
std = EnglishTextNormalizer()

refs = [std(it["ref"]) for it in data]
preds = [std(it["pred"].split("</s>")[0].replace("<s>", "")) for it in data]

subs, dels, ins, total = criterion._compute(predictions=preds, references=refs)

print(f"{len(preds)} samples, Sub: {subs}, Del: {dels}, Ins: {ins}, Total: {total}")
print(f"WER: {(subs + dels + ins)  / total * 100:.2f}")
