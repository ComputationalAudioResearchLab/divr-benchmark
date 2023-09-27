import re
from pathlib import Path
from typing import List


class Top:
    def run(self, output_file: str, log_files: List[str]):
        with open(output_file, "w") as output:
            for log_file_path in log_files:
                with open(log_file_path, "r") as log_file:
                    log_data = log_file.read()
                    tpr, tnr, acc = self.eval_metrics(log_data)
                    output.write(f"{acc:2.2f},{tpr:2.2f},{tnr:2.2f}: {log_file_path}\n")

    def eval_metrics(self, log_data: str):
        val_matches = re.search(
            r"Val >>(.|\n)*confusion >\n.*(\[\[(\[|\]| |\d|\n)*\]\])", log_data
        )
        tp, fp, fn, tn = re.findall(r"(\d+)", val_matches[2])
        tp, fp, fn, tn = int(tp), int(fp), int(fn), int(tn)
        tpr = tp / (tp + fn) * 100
        tnr = tn / (tn + fp) * 100
        acc = (tp + tn) / (tp + fp + tn + fn) * 100
        return tpr, tnr, acc


if __name__ == "__main__":
    curdir = Path(__file__).resolve().parent
    top = Top()
    top.run(
        output_file=f"{curdir}/output.log",
        log_files=[
            # SVD aiu to SVD aiu
            "/home/storage/data/nn_latents_full/16000/nn/results/svd_aiu.data2vec_base_960.1000_0.001_2_1024.log",
            "/home/storage/data/nn_latents_full/16000/nn/results/svd_aiu.unispeech_sat_large.1000_0.001_5_512.log",
            "/home/storage/data/nn_latents_full/16000/nn/results/svd_aiu.wav2vec_large.1000_0.0001_10_512.log",
            "/home/storage/data/nn_latents_full/16000/nn/results/svd_aiu.modified_cpc.1000_0.001_2_1024.log",
            "/home/storage/data/baseline_latents/16000/nn/results/svd_aiu.mfcc_mu.1000_0.0001_2_1024.log",
            # SVD aiu to Voiced
            "/home/storage/data/cross_tester/svd_to_voiced_faster/results/nn_latents_full.16000.nn.svd_aiu.data2vec_base_960.1000_0.001_2_1024.log",
            "/home/storage/data/cross_tester/svd_to_voiced_faster/results/nn_latents_full.16000.nn.svd_aiu.unispeech_sat_large.1000_0.0001_10_512.log",
            "/home/storage/data/cross_tester/svd_to_voiced/results/nn_latents_full.16000.nn.svd_aiu.wav2vec_large.1000_0.001_2_1024.log",
            "/home/storage/data/cross_tester/svd_to_voiced/results/nn_latents_full.16000.nn.svd_aiu.modified_cpc.1000_0.0001_5_512.log",
            "/home/storage/data/cross_tester/svd_to_voiced/results/baseline_latents.16000.nn.svd_aiu.mfcc_mu.1000_0.001_2_1024.log",
            # SVD a to SVD a
            "/home/storage/data/nn_latents_full/16000/nn/results/svd_a.data2vec_base_960.1000_0.0001_2_1024.log",
            "/home/storage/data/nn_latents_full/16000/nn/results/svd_a.unispeech_sat_large.1000_0.0001_5_512.log",
            "/home/storage/data/nn_latents_full/16000/nn/results/svd_a.wav2vec_large.100_0.0001_5_1024.log",
            "/home/storage/data/nn_latents_full/16000/nn/results/svd_a.modified_cpc.100_0.001_10_512.log",
            "/home/storage/data/baseline_latents/16000/nn/results/svd_a.mfcc_mu.100_0.0001_10_1024.log",
            # SVD a to Voiced
            "/home/storage/data/cross_tester/svd_to_voiced_faster/results/nn_latents_full.16000.svm.svd_a.data2vec_base_960.1.0_5_poly.log",
            "/home/storage/data/cross_tester/svd_to_voiced_faster/results/nn_latents_full.16000.nn.svd_a.unispeech_sat_large.1000_0.001_10_1024.log",
            "/home/storage/data/cross_tester/svd_to_voiced/results/nn_latents_full.16000.nn.svd_a.wav2vec_large.100_0.001_2_512.log",
            "/home/storage/data/cross_tester/svd_to_voiced/results/nn_latents_full.16000.nn.svd_a.modified_cpc.1000_0.0001_5_512.log",
            "/home/storage/data/cross_tester/svd_to_voiced/results/baseline_latents.16000.nn.svd_a.mfcc_mu.100_0.001_2_1024.log",
            # #### SVD to SVD
            # "/home/storage/data/nn_latents_full/16000/nn/results/svd_aiu.wav2vec_large.1000_0.0001_10_512.log",
            # "/home/storage/data/nn_latents_full/16000/nn/results/svd_aiu.modified_cpc.1000_0.001_2_1024.log",
            # "/home/storage/data/nn_latents_full/16000/nn/results/svd_aiu.wav2vec_large.1000_0.0001_10_1024.log",
            # "/home/storage/data/nn_latents_full/16000/nn/results/svd_aiu.wav2vec_large.100_0.001_2_512.log",
            # "/home/storage/data/nn_latents_full/16000/nn/results/svd_aiu.modified_cpc.100_0.001_10_1024.log",
            # "/home/storage/data/nn_latents_full/16000/nn/results/svd_a.data2vec_base_960.1000_0.0001_2_1024.log",
            # "/home/storage/data/nn_latents_full/16000/nn/results/svd_a.data2vec_base_960.100_0.001_5_512.log",
            # "/home/storage/data/nn_latents/16000/nn/results/svd_aiu.modified_cpc.1000_0.001_2_512.log",
            # "/home/storage/data/baseline_latents/16000/nn/results/svd_aiu.mfcc_mu.1000_0.0001_2_1024.log",
            # "/home/storage/data/baseline_latents/16000/nn/results/svd_aiu.mfcc_mu.100_0.001_10_1024.log",
            # #### SVD to Voiced
            # "/home/storage/data/cross_tester/svd_to_voiced/results/nn_latents.16000.nn.svd_a.hubert_base_robust_mgr.100_0.001_10_512.log",
            # "/home/storage/data/cross_tester/svd_to_voiced/results/baseline_latents.24000.nn.svd_a.mel_mu.100_0.001_2_512.log",
            # "/home/storage/data/cross_tester/svd_to_voiced/results/baseline_latents.24000.nn.svd_a.mel_mu_mel_std.100_0.001_10_512.log",
            # "/home/storage/data/cross_tester/svd_to_voiced/results/nn_latents.24000.nn.svd_a.wav2vec_large.1000_0.001_1_1024.log",
            # "/home/storage/data/cross_tester/svd_to_voiced/results/baseline_latents.24000.nn.svd_a.mel_mu.100_0.0001_10_512.log",
            # "/home/storage/data/cross_tester/svd_to_voiced/results/nn_latents.24000.nn.svd_aiu.apc_960hr.100_0.0001_5_512.log",
            # "/home/storage/data/cross_tester/svd_to_voiced/results/nn_latents.16000.svm.svd_aiu.modified_cpc.1.0_5_poly.log",
            # "/home/storage/data/cross_tester/svd_to_voiced/results/baseline_latents.16000.nn.svd_aiu.mel_mu_mel_std_mfcc_mu_mfcc_std.1000_0.0001_10_1024.log",
            # "/home/storage/data/cross_tester/svd_to_voiced/results/nn_latents.24000.nn.svd_a.wav2vec_large.1000_0.001_1_512.log",
            # ### Voiced to SVD A
            # "/home/storage/data/cross_tester/voiced_to_svd_a/results/nn_latents.24000.nn.voiced.xls_r_2b.100_0.001_5_512.log",
            # "/home/storage/data/cross_tester/voiced_to_svd_a/results/baseline_latents.16000.nn.voiced.mel_mu_mel_std.100_0.0001_10_1024.log",
            # "/home/storage/data/cross_tester/voiced_to_svd_a/results/baseline_latents.16000.nn.voiced.mel_std.100_0.001_10_512.log",
            # "/home/storage/data/cross_tester/voiced_to_svd_a/results/baseline_latents.16000.nn.voiced.mel_mu_mel_std_mfcc_mu_mfcc_std.1000_0.001_5_1024.log",
            # "/home/storage/data/cross_tester/voiced_to_svd_a/results/nn_latents.24000.nn.voiced.hubert_base_robust_mgr.100_0.001_1_1024.log",
            # "/home/storage/data/cross_tester/voiced_to_svd_a/results/nn_latents.24000.nn.voiced.hubert_base_robust_mgr.100_0.001_1_512.log",
            # "/home/storage/data/cross_tester/voiced_to_svd_a/results/nn_latents.24000.nn.voiced.hubert_base_robust_mgr.1000_0.0001_1_1024.log",
            # ### Voiced to SVD I
            # "/home/storage/data/cross_tester/voiced_to_svd_i/results/nn_latents.24000.nn.voiced.distilhubert_base.1000_0.0001_10_512.log",
            # "/home/storage/data/cross_tester/voiced_to_svd_i/results/nn_latents.24000.nn.voiced.distilhubert_base.100_0.0001_5_1024.log",
            # "/home/storage/data/cross_tester/voiced_to_svd_i/results/nn_latents.24000.nn.voiced.distilhubert_base.1000_0.0001_10_1024.log",
            # "/home/storage/data/cross_tester/voiced_to_svd_i/results/nn_latents.24000.nn.voiced.hubert_base.1000_0.001_10_1024.log",
            # "/home/storage/data/cross_tester/voiced_to_svd_i/results/nn_latents.16000.nn.voiced.distilhubert_base.100_0.001_10_1024.log",
            # "/home/storage/data/cross_tester/voiced_to_svd_i/results/nn_latents.24000.nn.voiced.distilhubert_base.1000_0.001_10_1024.log",
            # "/home/storage/data/cross_tester/voiced_to_svd_i/results/nn_latents.24000.nn.voiced.hubert_base.100_0.0001_5_512.log",
            # "/home/storage/data/cross_tester/voiced_to_svd_i/results/nn_latents.24000.nn.voiced.distilhubert_base.100_0.001_10_1024.log",
            # "/home/storage/data/cross_tester/voiced_to_svd_i/results/nn_latents.24000.nn.voiced.hubert_base.1000_0.0001_10_1024.log",
            # ### Voiced to SVD U
            # "/home/storage/data/cross_tester/voiced_to_svd_u/results/baseline_latents.16000.nn.voiced.mel_mu_mel_std.1000_0.0001_2_1024.log",
            # "/home/storage/data/cross_tester/voiced_to_svd_u/results/baseline_latents.16000.nn.voiced.mel_mu_mel_std.100_0.001_2_512.log",
            # "/home/storage/data/cross_tester/voiced_to_svd_u/results/nn_latents.24000.nn.voiced.distilhubert_base.1000_0.0001_10_512.log",
            # "/home/storage/data/cross_tester/voiced_to_svd_u/results/baseline_latents.16000.nn.voiced.mel_mu_mel_std.1000_0.001_2_512.log",
            # "/home/storage/data/cross_tester/voiced_to_svd_u/results/baseline_latents.16000.nn.voiced.mel_mu_mel_std.100_0.0001_2_1024.log",
            # "/home/storage/data/cross_tester/voiced_to_svd_u/results/nn_latents.24000.nn.voiced.hubert_base.100_0.001_10_1024.log",
            # "/home/storage/data/cross_tester/voiced_to_svd_u/results/baseline_latents.24000.nn.voiced.mel_std.100_0.0001_2_1024.log",
            # ### Voiced to SVD AIU
            # # "/home/storage/data/cross_tester/voiced_to_svd_aiu/results/baseline_latents.24000.nn.voiced.mfcc_mu.1000_0.001_2_512.log",
            # "/home/storage/data/cross_tester/voiced_to_svd_aiu/results/nn_latents.16000.nn.voiced.wav2vec_large.100_0.001_5_512.log",
            # "/home/storage/data/cross_tester/voiced_to_svd_aiu/results/nn_latents.16000.nn.voiced.wav2vec_large.100_0.0001_2_512.log",
            # "/home/storage/data/cross_tester/voiced_to_svd_aiu/results/baseline_latents.16000.nn.voiced.mel_mu_mel_std.1000_0.0001_2_1024.log",
            # "/home/storage/data/cross_tester/voiced_to_svd_aiu/results/nn_latents.24000.nn.voiced.wav2vec_large.100_0.0001_2_512.log",
            # "/home/storage/data/cross_tester/voiced_to_svd_aiu/results/nn_latents.24000.nn.voiced.xls_r_2b.1000_0.001_10_512.log",
        ],
    )
