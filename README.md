# Augmented-Dialectal-Speech-Recognition-for-AD-Screening
## Project Introduction

This repository aims to combine Automatic Speech Recognition (ASR) technology with Natural Language Processing (NLP) techniques to recognize the speech of elderly individuals in the context of Alzheimer's Disease (AD) assessment. The project addresses the challenges posed by the scarcity of dialect-labeled data and the unclear pronunciation of elderly people.

### Key Objectives

#### Phase 1: Overcoming Dialect Label Data Scarcity
In the first phase of the project, we tackle the issue of limited dialect-labeled data by leveraging pre-training techniques for the encoder. This process involves utilizing a large corpus of open-source ASR data to pre-train the model. Subsequently, we fine-tune the model using a small dataset of authentic elderly dialect speech. This represents the initial step of our project.

#### Phase 2: Addressing Elderly Pronunciation Variability
In the second phase, we focus on mitigating the challenges presented by the variable and unclear pronunciation of elderly individuals. We harness the question-and-answer format commonly used in AD assessment questionnaires and employ NLP techniques to correct the transcriptions. This refinement process aims to enhance the accuracy of audio recognition.

By combining ASR and NLP technologies, our project seeks to advance the accuracy of speech recognition in the context of AD assessment, ultimately facilitating a more precise and efficient screening process for elderly individuals.

### Data set description

| Data name | Language | Duration/Number | Open Source | Link | Frequency |
|----------|----------|----------|----------|----------|----------|
| LibriSpeech | English | 1000 hours | Yes | [Click Me](https://www.openslr.org/12) | 16k |
| B-NSA Audio | English | 197 | No | [Click Me](https://drive.google.com/drive/folders/1cOpW4RbWA4Qm5BqpvaXG11S1XtvbwgE2) | 16k |
| C-NA | Chinese | 20 hours | No | [Click Me](https://drive.google.com/drive/folders/1wkcJF8DuLq7nHhc6rhU8enA5qQKbfQWXm) | 16k |
| AISHELL-1  | Chinese | 178 hours | Yes | [Click Me](https://www.openslr.org/33/) | 16k |
| C-NSA Audio | Chinese | 306 samples | No | [Click Me](https://drive.google.com/drive/folders/1MEo6OL5VP6DhsJrhrvvYMP7ebiFf7jw4) | 16k |
| B-NSA Text  | English | 197 | No | [Click Me](https://drive.google.com/drive/folders/1bj7YnU64LOfZYCkrCSd-rSfVmyJWNlgF) | 16k |
| Chinese-NSA | Chinese | 100 | No | [Click Me](https://drive.google.com/drive/folders/1vVbvaSkAw7ITi8Y0VB5_7E5mzAIQhnIl) | 16k |
| CQ-NSA Tex | Chinese | 490000 | No | [Click Me](https://drive.google.com/drive/folders/16z5a4VhYN3zj48Q0K46vp_RgvkOJZ7Uk) | 16k |


### Start

#### Stage 1: fine tuning 


* training base model based on aishell-1 dataset: bash run.sh --stage 4 --stop-stage 6

* fine tuning our dataset: bash aiadrun.sh  --stage 4 --stop-stage 6


#### Stage 2: training MRC


* set variables: TF_KERAS=1 python mrc_mlm.py




