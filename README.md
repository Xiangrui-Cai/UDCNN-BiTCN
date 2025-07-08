# UDCNN-BiTCN
UDCNN-BiTCN: Novel Brain-Computer Interface Application: Precise Decoding of Urination and Defecation Motor Attempts in Spinal Cord Injury Patients

Author: Xiangrui Cai, Chuan Xue, Lei Cao*, Jie Jia*, Ziqi Guo, Haoran Xu, Shuyang Zhang, Chunjiang Fan.

# About UDCNN-BiTCN
1. This paper proposes a Bidirectional Temporal Convolutional Network model (UDCNN-BiTCN) for decoding urination and defecation movement intentions, which represents the first attempt to decode urination and defecation movement intention tasks.
   
2. UDCNN-BiTCN introduces a novel amplitude masking method and the Bi-TCN architecture, and extensive ablation experiments validate the effectiveness of each module.
   
3. UDCNN-BiTCN achieves the best performance in withinsubject and within-task, cross-subject and within-task, as well as within-subject and cross-task settings.

4. We found that SCI patients achieved higher classification accuracy in urination and defecation tasks. In addition, we identified the 0.5–4 Hz (delta band) as the most influential frequency band for classification and explored the underlying reasons.
   
![Model](./Model.png)

# About Datasets
We sincerely apologize that we have not yet made our dataset and the corresponding training models for each subject publicly available. However, we commit to releasing our dataset by the end of 2025 or early 2026.

# Development environment
We implemented our UDCNN-BiTCN model using Python 3.8 and the TensorFlow library on a NVIDIA GeForce RTX 2080 Ti GPU.

· Tensorflow == 2.9

· matplotlib == 3.7.5

· NumPy == 1.23.5

· scikit-learn == 1.1.3

· SciPy == 1.9.3

# References
The manuscript is currently under submission.

If you have any questions about our paper or code, please feel free to contact us at: 202330310112@stu.shmtu.edu.cn

# Other
We would like to express our sincere gratitude to the following papers for providing open-source code, which has been of great significance to our research:

Physics-informed attention temporal convolutional network for EEG-based motor imagery classification.

# Details


