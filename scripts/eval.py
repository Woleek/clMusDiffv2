import cdpam
import glob
import os
import librosa as lb
import numpy as np
import tqdm
import torch
from visqol import visqol_lib_py
from visqol.pb2 import visqol_config_pb2
from visqol.pb2 import similarity_result_pb2
from audioldm_eval import EvaluationHelper
import pickle
import json

# CDPAM
cdpam_dist = cdpam.CDPAM()

# VISQOLA
config = visqol_config_pb2.VisqolConfig()
config.audio.sample_rate = 48000
config.options.use_speech_scoring = False
svr_model_path = "libsvm_nu_svr_model.txt"
config.options.svr_model_path = os.path.join(
    os.path.dirname(visqol_lib_py.__file__), "model", svr_model_path)
api = visqol_lib_py.VisqolApi()
api.Create(config)

# SI-SNR
def si_snr(estimate, reference, epsilon=1e-8):
    estimate = torch.tensor(estimate)
    reference = torch.tensor(reference)
    
    min_length = min(estimate.size(0), reference.size(0))
    estimate = estimate[:min_length]
    reference = reference[:min_length]
    
    estimate = estimate - estimate.mean()
    reference = reference - reference.mean()
    reference_pow = reference.pow(2).mean(axis=0, keepdim=True)
    mix_pow = (estimate * reference).mean(axis=0, keepdim=True)
    scale = mix_pow / (reference_pow + epsilon)

    reference = scale * reference
    error = estimate - reference

    reference_pow = reference.pow(2)
    error_pow = error.pow(2)

    reference_pow = reference_pow.mean(axis=0)
    error_pow = error_pow.mean(axis=0)

    si_snr = 10 * torch.log10(reference_pow) - 10 * torch.log10(error_pow)
    return si_snr.item()

# AudioLDM eval
evaluator = EvaluationHelper(16000, 'cuda')



# eval
base_path = 'datasets/Img2Spec/eval/'
audio_ds = glob.glob(base_path + 'audio_eval/*.wav')
audio_ds.sort()
audio_gen_new = glob.glob(base_path + 'audio_eval_gen/*.wav')
audio_gen_new.sort()
audio_gen_old = glob.glob(base_path + 'audio_eval_old/*.wav')
audio_gen_old.sort()
audio_map = glob.glob(base_path + 'audio_eval_map/*.wav')
audio_map.sort()

results = {
    "new": {"cdpam": [], "visqola": [], "si_snr": [], "sf": [], "zcr": [], "rmse": [], 'fd': [], "fad": [], "kl": [], "is": []},
    "old": {"cdpam": [], "visqola": [], "si_snr": [], "sf": [], "zcr": [], "rmse": [], 'fd': [], "fad": [], "kl": [], "is": []},
    "map": {"cdpam": [], "visqola": [], "si_snr": [], "sf": [], "zcr": [], "rmse": [], 'fd': [], "fad": [], "kl": [], "is": []},
    "ds": {"sf": [], "zcr": [], "rmse": []}
}

assert len(audio_ds) == len(audio_gen_new) == len(audio_gen_old) == len(audio_map)
for path_ds, path_new, path_old, path_map in tqdm.tqdm(list(zip(audio_ds, audio_gen_new, audio_gen_old, audio_map))):
    # CDPAM
    sample_ds = cdpam.load_audio(path_ds)
    sample_new = cdpam.load_audio(path_new)
    sample_old = cdpam.load_audio(path_old)
    sample_map = cdpam.load_audio(path_map)
    
    cdpam_res_new = cdpam_dist.forward(sample_ds, sample_new)[0].item()
    results["new"]["cdpam"].append(cdpam_res_new)
    cdpam_res_old = cdpam_dist.forward(sample_ds, sample_old)[0].item()
    results["old"]["cdpam"].append(cdpam_res_old)
    cdpam_res_map = cdpam_dist.forward(sample_ds, sample_map)[0].item()
    results["map"]["cdpam"].append(cdpam_res_map)
    
    # VISQOLA
    sample_ds, _ = lb.load(path_ds, sr=22050, dtype=np.float64)
    sample_new, _ = lb.load(path_new, sr=22050, dtype=np.float64)
    sample_old, _ = lb.load(path_old, sr=22050, dtype=np.float64)
    sample_map, _ = lb.load(path_map, sr=22050, dtype=np.float64)
    
    visqol_res_new = api.Measure(sample_ds, sample_new).moslqo
    results["new"]["visqola"].append(visqol_res_new)
    visqol_res_old = api.Measure(sample_ds, sample_old).moslqo
    results["old"]["visqola"].append(visqol_res_old)
    visqol_res_map = api.Measure(sample_ds, sample_map).moslqo
    results["map"]["visqola"].append(visqol_res_map)
    
    # Si-SNR
    si_snr_res_new = si_snr(sample_new, sample_ds)
    results["new"]["si_snr"].append(si_snr_res_new)
    si_snr_res_old = si_snr(sample_old, sample_ds)
    results["old"]["si_snr"].append(si_snr_res_old)
    si_snr_res_map = si_snr(sample_map, sample_ds)
    results["map"]["si_snr"].append(si_snr_res_map)
    
    # SF
    sf_res_new = lb.feature.spectral_flatness(y=sample_new).mean()
    results["new"]["sf"].append(sf_res_new)
    sf_res_old = lb.feature.spectral_flatness(y=sample_old).mean()
    results["old"]["sf"].append(sf_res_old)
    sf_res_map = lb.feature.spectral_flatness(y=sample_map).mean()
    results["map"]["sf"].append(sf_res_map)
    sf_res_ds = lb.feature.spectral_flatness(y=sample_ds).mean()
    results["ds"]["sf"].append(sf_res_ds)
    
    # ZCR
    zcr_res_new = lb.feature.zero_crossing_rate(y=sample_new).mean()
    results["new"]["zcr"].append(zcr_res_new)
    zcr_res_old = lb.feature.zero_crossing_rate(y=sample_old).mean()
    results["old"]["zcr"].append(zcr_res_old)
    zcr_res_map = lb.feature.zero_crossing_rate(y=sample_map).mean()
    results["map"]["zcr"].append(zcr_res_map)
    zcr_res_ds = lb.feature.zero_crossing_rate(y=sample_ds).mean()
    results["ds"]["zcr"].append(zcr_res_ds)
    
    # RMSE
    rmse_res_new = lb.feature.rms(y=sample_new)
    results["new"]["rmse"].append(rmse_res_new)
    rmse_res_old = lb.feature.rms(y=sample_old)
    results["old"]["rmse"].append(rmse_res_old)
    rmse_res_map = lb.feature.rms(y=sample_map)
    results["map"]["rmse"].append(rmse_res_map)
    rmse_res_ds = lb.feature.rms(y=sample_ds)
    results["ds"]["rmse"].append(rmse_res_ds)

# SAVE 1
with open('results/results_1.pkl', 'wb') as f:
    pickle.dump(results, f)

ldm_res_new = evaluator.main(
    base_path + 'audio_eval_gen',
    base_path + 'audio_eval'
)
results["new"]["fd"] = ldm_res_new['frechet_distance']
results["new"]["fad"] = ldm_res_new['frechet_audio_distance']
results["new"]["kl"] = ldm_res_new['kullback_leibler_divergence_softmax']
results["new"]["is"] = ldm_res_new['inception_score_mean']

ldm_res_old = evaluator.main(
    base_path + 'audio_eval_old',
    base_path + 'audio_eval'
)
results["old"]["fd"] = ldm_res_old['frechet_distance']
results["old"]["fad"] = ldm_res_old['frechet_audio_distance']
results["old"]["kl"] = ldm_res_old['kullback_leibler_divergence_softmax']
results["old"]["is"] = ldm_res_old['inception_score_mean']

ldm_res_map = evaluator.main(
    base_path + 'audio_eval_map',
    base_path + 'audio_eval'
)
results["map"]["fd"] = ldm_res_map['frechet_distance']
results["map"]["fad"] = ldm_res_map['frechet_audio_distance']
results["map"]["kl"] = ldm_res_map['kullback_leibler_divergence_softmax']
results["map"]["is"] = ldm_res_map['inception_score_mean']

# SAVE 2
with open('results/results_2.pkl', 'wb') as f:
    pickle.dump(results, f)
    
for key, metrics_dict in results.items():
    for metric, values in metrics_dict.items():
        if values and isinstance(values, list):
            results[key][metric] = np.mean(values)
            
# SAVE 3
with open('results/results_3.pkl', 'wb') as f:
    pickle.dump(results, f)
    

# SAVE RESULTS
def convert_numpy_types(obj):
    if isinstance(obj, np.generic):
        return obj.item()  # Convert numpy type to Python native type
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

with open('results/new.json', 'w') as f:
    json.dump(results["new"], f, default=convert_numpy_types)
with open('results/old.json', 'w') as f:
    json.dump(results["old"], f, default=convert_numpy_types)
with open('results/map.json', 'w') as f:
    json.dump(results["map"], f, default=convert_numpy_types)
with open('results/ds.json', 'w') as f:
    json.dump(results["ds"], f, default=convert_numpy_types)