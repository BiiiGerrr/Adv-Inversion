import torch
import numpy as np
import random
# from omegaconf import OmegaConf
# from ldm.models.diffusion.ddim import DDIMSampler
# from ldm.util import instantiate_from_config
from fr_model import IRSE_50, MobileFaceNet, IR_152, InceptionResnetV1


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_fr_model(name):
    if name == 'IRSE50':
        model = IRSE_50()
        model.load_state_dict(torch.load('pretrained_models/irse50.pth', map_location=torch.device('cpu')))
    elif name == 'MobileFace':
        model = MobileFaceNet(512)
        model.load_state_dict(torch.load('pretrained_modelsmobile_face.pth', map_location=torch.device('cpu')))
    elif name == 'IR152':
        model = IR_152([112, 112])
        model.load_state_dict(torch.load('pretrained_modelsir152.pth', map_location=torch.device('cpu')))
    elif name == 'FaceNet':
        model = InceptionResnetV1(num_classes=8631)
        model.load_state_dict(torch.load('pretrained_modelsfacenet.pth', map_location=torch.device('cpu')))
    else:
        raise ValueError(f'Invalid model name: {name}')

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model.cuda()


th_dict = {'IR152': (0.094632, 0.166788, 0.227922), 'IRSE50': (0.144840, 0.241045, 0.312703),
           'FaceNet': (0.256587, 0.409131, 0.591191), 'MobileFace': (0.183635, 0.301611, 0.380878)}


def asr_calculation(cos_sim_scores_dict):
    # Iterate each image pair's simi-score from "simi_scores_dict" and compute the attacking success rate
    for key, values in cos_sim_scores_dict.items():
        th01, th001, th0001 = th_dict[key]
        total = len(values)
        success01 = 0
        success001 = 0
        success0001 = 0
        for v in values:
            if v > th01:
                success01 += 1
            if v > th001:
                success001 += 1
            if v > th0001:
                success0001 += 1
        print(key, " attack success(far@0.1) rate: ", success01 / total)
        print(key, " attack success(far@0.01) rate: ", success001 / total)
        print(key, " attack success(far@0.001) rate: ", success0001 / total)


def asr_calculation_list(cos_sim_scores_dict_list):
    # Iterate over each set of cosine similarity scores (for different processing methods)
    for i, cos_sim_scores_dict in enumerate(cos_sim_scores_dict_list):
        print(f"Results for processing method {i + 1}:")

        # Iterate over each defense method (raw, jpeg, squeezed, smoothed)
        for defense_method, values_dict in cos_sim_scores_dict.items():
            print(f"\nDefense Method: {defense_method.capitalize()}")

            # Iterate each model's scores in the dictionary
            for model_name, values in values_dict.items():
                th01, th001, th0001 = th_dict[defense_method]
                total = len(values)
                success01 = 0
                success001 = 0
                success0001 = 0

                for v in values:
                    if v > th01:
                        success01 += 1
                    if v > th001:
                        success001 += 1
                    if v > th0001:
                        success0001 += 1

                # Print attack success rates for each threshold
                print(f"Model {model_name}:")
                print(f"  Attack success (far@0.1) rate: {success01 / total:.4f}")
                print(f"  Attack success (far@0.01) rate: {success001 / total:.4f}")
                print(f"  Attack success (far@0.001) rate: {success0001 / total:.4f}")
            print("-" * 40)  # Separator between different models for this defense method
        print("=" * 40)  # Separator between different defense methods


def asr_calculation_txt(cos_sim_scores_dict, output_path=None):
    # Open the output file if the path is provided
    if output_path:
        file = open(output_path, "w")
    else:
        file = None

    # Iterate each image pair's simi-score from "simi_scores_dict" and compute the attacking success rate
    for key, values in cos_sim_scores_dict.items():
        th01, th001, th0001 = th_dict[key]
        total = len(values)
        success01 = 0
        success001 = 0
        success0001 = 0
        for v in values:
            if v > th01:
                success01 += 1
            if v > th001:
                success001 += 1
            if v > th0001:
                success0001 += 1

        # Format the output strings
        output01 = f"{key} attack success(far@0.1) rate: {success01 / total}"
        output001 = f"{key} attack success(far@0.01) rate: {success001 / total}"
        output0001 = f"{key} attack success(far@0.001) rate: {success0001 / total}"

        # Print to console
        print(output01)
        print(output001)
        print(output0001)

        # Write the same to the file if file is open
        if file:
            file.write(output01 + "\n")
            file.write(output001 + "\n")
            file.write(output0001 + "\n")

        # Close the file if it was opened
    if file:
        file.close()


def asr_calculation_dogging_txt(cos_sim_scores_dict, output_path=None):
    # Open the output file if the path is provided
    if output_path:
        file = open(output_path, "w")
    else:
        file = None

    # Iterate each image pair's simi-score from "simi_scores_dict" and compute the attacking success rate
    for key, values in cos_sim_scores_dict.items():
        th01, th001, th0001 = th_dict[key]
        total = len(values)
        success01 = 0
        success001 = 0
        success0001 = 0
        for v in values:
            if v < th01:
                success01 += 1
            if v < th001:
                success001 += 1
            if v < th0001:
                success0001 += 1

        # Format the output strings
        output01 = f"{key} attack success(far@0.1) rate: {success01 / total}"
        output001 = f"{key} attack success(far@0.01) rate: {success001 / total}"
        output0001 = f"{key} attack success(far@0.001) rate: {success0001 / total}"

        # Print to console
        print(output01)
        print(output001)
        print(output0001)

        # Write the same to the file if file is open
        if file:
            file.write(output01 + "\n")
            file.write(output001 + "\n")
            file.write(output0001 + "\n")

        # Close the file if it was opened
    if file:
        file.close()