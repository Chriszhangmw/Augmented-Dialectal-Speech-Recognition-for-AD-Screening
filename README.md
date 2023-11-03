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

|                  |         |               |         |            |            |        |                 |
|:----------------:|:-------:|:-------------:|:-------:|:----------:|:----------:|:------:|:---------------:|
|                  |         | **Audio**     |         |            | **Text**   |        |                 |
| **English**      | Name    | Duration/Number | Frequency | Open Source | Name       | Number | Open Source     |
|                  |         |                 |           |            |            |        |                 |
|                  | LibriSpeech | 1000 hours | 16K Hz | Yes | B-NSA Text | 384 | No |
|                  | B-NSA Audio | 394 | 16K Hz | No | | | |
| **Chinese**      | C-NA | 20 hours | 16K Hz | No | Chinese-NSA | 100 | No |
|                  | AISHELL-1 | 178 hours | 16K Hz | Yes | CQ-NSA Text | 490,000 | |
|                  | C-NSA Audio | 306 samples | 16K Hz | No | | | |


