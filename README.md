# Area for research: 
1. CLIP in point-cloud/3D. 
2. Open-Vocabulary Object Detection (OVD)
3. efficient CLIP training (better use of computation or data)
4. applying CLIP models in narrow fields; such as Human Object Interaction detection, crowd counting...etc 
# Papers from [CVPR2023](https://cvpr2023.thecvf.com/Conferences/2023/AcceptedPapers): 
(might missed some papers)

## pretraining CLIP models: 
|Title | Description | Code |
| ----------- | ----------- | ----------- |
|[DisCo-CLIP: A Distributed Contrastive Loss for Memory Efficient CLIP Training](https://arxiv.org/abs/2304.08480)| Reducing memory consumption through decomposing the gradient| [code](https://github.com/IDEA-Research/DisCo-CLIP) 
|[Scaling Language-Image Pre-training via Masking](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Scaling_Language-Image_Pre-Training_via_Masking_CVPR_2023_paper.pdf)| by adding masked image modelling to the image branch of clip it improved speed, memory, and performance| [code](https://github.com/facebookresearch/flip) |
|[Non-Contrastive Learning Meets Language-Image Pre-Training](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhou_Non-Contrastive_Learning_Meets_Language-Image_Pre-Training_CVPR_2023_paper.pdf)| added the loss introduced in SwAv (based on cluster assignment agreement) in addition to the contrastive loss of CLIP. interestingly, if non-Contrastive loss is used alone the zero-shot performance is bad but when used with contrastive loss (0.7*swav + 0.3*contrastive) it over perform the contrastive loss. Additionally, it helped the need for data (trained on 35-million only) and small batch size (4096 combared to 32K)|[code](https://github.com/shallowtoil/xclip)

## Finetuning CLIP models: 
|Title | Description | Code |
| ----------- | ----------- | ----------- |
|[Learning to Name Classes for Vision and Language Models](https://openaccess.thecvf.com/content/CVPR2023/papers/Parisot_Learning_To_Name_Classes_for_Vision_and_Language_Models_CVPR_2023_paper.pdf)| created a learnable token embedding for the class names in otherwise frozen clip model, reduce the need for prompt engineering| NA| 
|[Multimodality Helps Unimodality: Cross-Modal Few-Shot Learning with Multimodal Models](https://openaccess.thecvf.com/content/CVPR2023/papers/Lin_Multimodality_Helps_Unimodality_Cross-Modal_Few-Shot_Learning_With_Multimodal_Models_CVPR_2023_paper.pdf)| when fine-tuning the model with linear classifier it is useful to train it from multi modality| NA|
|[MaPLe: Multi-modal Prompt Learning](https://openaccess.thecvf.com/content/CVPR2023/papers/Khattak_MaPLe_Multi-Modal_Prompt_Learning_CVPR_2023_paper.pdf)| learnable prompts on both the image and text branches, image prompt are derived from a linear layer that takes the text prompt as input| [code](https://github.com/muzairkhattak/multimodal-prompt-learning)|

## CLIP in video: 
|Title | Description | Code |
| ----------- | ----------- | ----------- |
|[Fine-Tuned CLIP Models Are Efficient Video Learners](https://openaccess.thecvf.com/content/CVPR2023/papers/Rasheed_Fine-Tuned_CLIP_Models_Are_Efficient_Video_Learners_CVPR_2023_paper.pdf)|Adapts clip for videos. Claims that frame level clip embeddings from the videos though processed independently can still show temporal dependencies. Claims that instead of devising certain specific modules to address the temporal dependency in videos, simply fine-tuning ViFiCLIP can generalise to good performance. They do temporal pooling meaning pool embeddings from T frames and use that embedding in the contrastive learning process. This is probably why the embeddings are consistent with image based CLIP.|[code](https://github.com/muzairkhattak/ViFi-CLIP)| 
|[Vita-CLIP: Video and text adaptive CLIP via Multimodal Prompting](https://arxiv.org/abs/2304.03307)|Performs prompt learning on the video data to better fine tune image based CLIP model for videos. Same authors as of ViFi CLIP (above) Need to look into how the prompts are actually learned.|[code](https://github.com/TalalWasim/Vita-CLIP)| 

## Crowd Counting: 
|Title | Description | Code |
| ----------- | ----------- | ----------- |
|[CrowdCLIP: Unsupervised Crowd Counting via Vision-Language Model ](https://openaccess.thecvf.com/content/CVPR2023/papers/Liang_CrowdCLIP_Unsupervised_Crowd_Counting_via_Vision-Language_Model_CVPR_2023_paper.pdf)|crowd counting with clip. fine-tune clip for the counting task using ranking loss. Does not use labels of people counts as ground truth for training. uses a sequential prompting setting to filter parts that only contain people heads for counting|[code](https://github.com/dk-liang/CrowdCLIP)|

## Generative: 
|Title | Description | Code |
| ----------- | ----------- | ----------- |
|[ShapeClipper: Scalable 3D Shape Learning from Single-View Images via Geometric and CLIP-based Consistency](https://zixuanh.com/projects/shapeclipper.html)|...|...|
|[CLIP-Sculptor: Zero-Shot Generation of High-Fidelity and Diverse Shapes From Natural Language](https://openaccess.thecvf.com/content/CVPR2023/papers/Sanghi_CLIP-Sculptor_Zero-Shot_Generation_of_High-Fidelity_and_Diverse_Shapes_From_Natural_CVPR_2023_paper.pdf)|...|...|
|[CLIP2Protect: Protecting Facial Privacy using Text-Guided Makeup via Adversarial Latent Search](https://openaccess.thecvf.com/content/CVPR2023/papers/Shamshad_CLIP2Protect_Protecting_Facial_Privacy_Using_Text-Guided_Makeup_via_Adversarial_Latent_CVPR_2023_paper.pdf)|...|...|
|[Local 3D Editing via 3D Distillation of CLIP Knowledge](https://openaccess.thecvf.com/content/CVPR2023/papers/Hyung_Local_3D_Editing_via_3D_Distillation_of_CLIP_Knowledge_CVPR_2023_paper.pdf)|...|...|

## Continual learning: 
|Title | Description | Code |
| ----------- | ----------- | ----------- |
|[AttriCLIP: A Non-Incremental Learner for Incremental Knowledge Learning](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_AttriCLIP_A_Non-Incremental_Learner_for_Incremental_Knowledge_Learning_CVPR_2023_paper.pdf)| Used prompt tuning with CLIP to solve the problem of Continual learning, heavily inspired by CoOp| [code](https://gitee.com/mindspore/models/tree/master/research/cv/AttriCLIP)| 

## 3D and Point-cloud: 
|Title | Description | Code |
| ----------- | ----------- | ----------- |
|[CLIP2: Contrastive Language-Image-Point Pretraining From Real-World Point Cloud Data](https://openaccess.thecvf.com/content/CVPR2023/papers/Zeng_CLIP2_Contrastive_Language-Image-Point_Pretraining_From_Real-World_Point_Cloud_Data_CVPR_2023_paper.pdf)|...|...|

## Detection:  
|Title | Description | Code |
| ----------- | ----------- | ----------- |
|[DetCLIPv2: Scalable Open-Vocabulary Object Detection Pre-training via Word-Region Alignment](https://arxiv.org/abs/2304.04514)|
|[CLIP Is Also an Efficient Segmenter: A Text-Driven Approach for Weakly Supervised Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2023/papers/Lin_CLIP_Is_Also_an_Efficient_Segmenter_A_Text-Driven_Approach_for_CVPR_2023_paper.pdf)|
|[WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation](https://openaccess.thecvf.com/content/CVPR2023/papers/Jeong_WinCLIP_Zero-Few-Shot_Anomaly_Classification_and_Segmentation_CVPR_2023_paper.pdf)|
|[HOICLIP: Efficient Knowledge Transfer for HOI Detection with Vision-Language Models](https://arxiv.org/abs/2303.15786)|
