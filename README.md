# **Fair Adaptive Scaling (FAS) – Experiment** 

## **About** 📌  
This project was a small experiment to understand Fair Adaptive Scaling (FAS) in practice and see how it helps reduce bias in classification tasks.  

I worked with 3,000 images:  
-  1,500 male  
-  1,500 female  

After preprocessing, I fine-tuned a ResNet18 model.  

## **Experiment Steps** 🔬  
1. Trained the baseline model on the dataset.  
2. Calculated accuracy scores for individual classes on 1,000 new test images.  
3. Identified the underperforming class (male).  
4. Retrained the model using Fair Adaptive Scaling (FAS) to give more balanced attention to groups.  

## **My Understanding of FAS** 🧠  
FAS does not treat every training sample equally. Instead, it adapts how much attention the model pays to each sample during training.  

- **Individual-level weighting**  
  - Tracks how difficult each sample has been over time (using an exponential moving average of its cross-entropy loss).  
  - Samples that remain hard to classify get a higher weight.  

- **Group-level weighting**  
  - Each group (e.g., male or female) has its own learnable parameter.  
  - If a group overall is underperforming, its parameter increases so the model assigns more weight to that group’s samples.  

In my implementation, I combined these two signals with a parameter **c**:  
- Half weight from the sample’s difficulty.  
- Half weight from the group parameter.  
- The weights were normalized and clipped to stay stable.

## **Results** 📊  

| Model    | Male Accuracy | Female Accuracy | Gap   |
|----------|--------------:|----------------:|-------|
| Baseline | 73.74%        | 87.66%          | 13.92% |
| With FAS | 88.19%        | 87.75%          | 0.44%  |

✅ With FAS, the male accuracy improved significantly, while female accuracy remained stable.  
The performance gap between groups nearly disappeared.  
