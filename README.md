# Awesome Contrastive Learning & Data Augmentation RS Paper & Code

This repository collects the latest research progress of **Contrastive Learning (CL) and Data Augmentation (DA)** in Recommender Systems.
Comments and contributions are welcome.

CF = Collaborative Filtering, SSL = Self-Supervised Learning

- [Survey/Tutorial](#Survey/Tutorial) Total Papers: 3
- [Only Data Augmentation](#Only-Data-Augmentation) Total Papers: 17
- [Graph Models with CL](#Graph-Models-with-CL) Total Papers: 38
- [Sequential Models with CL](#Sequential-Models-with-CL) Total Papers: 27
- [Other Tasks with CL](#Other-Tasks-with-CL) Total Papers: 30


## Survey/Tutorial
1. **Self-Supervised Learning for Recommender Systems A Survey** (Survey)
   
   TKDE 2022, [[PDF]](https://arxiv.org/pdf/2203.15876.pdf), [[Code]](https://github.com/Coder-Yu/SELFRec)

2. **Self-Supervised Learning in Recommendation: Fundamentals and Advances** (Tutorial)
   
   WWW 2022, [[Web]](https://ssl-recsys.github.io/)
   
3. **Tutorial: Self-Supervised Learning for Recommendation: Foundations, Methods and Prospects** (Tutorial)
   
   DASFAA 2023, [[Web]](https://junliang-yu.github.io/publications/)


## Only Data Augmentation

1. **Enhancing Collaborative Filtering with Generative Augmentation** (CF + GAN + DA)
   
    KDD 2019, [[PDF]](https://arxiv.org/pdf/2207.02643.pdf)

2. **Future Data Helps Training Modeling Future Contexts for Session-based Recommendation** (Session + DA)

    WWW 2020, [[PDF]](https://arxiv.org/pdf/1906.04473.pdf), [[Code]](https://github.com/fajieyuan/WWW2020-grec)

3. **Augmenting Sequential Recommendation with Pseudo-Prior Items via Reversely Pre-training Transformer** (Sequential + DA)
   
    SIGIR 2021, [[PDF]](https://arxiv.org/pdf/2105.00522.pdf), [[Code]](https://github.com/DyGRec/ASReP)

4. **Self-Knowledge Distillation with Bidirectional Chronological Augmentation of Transformer for Sequential Recommendation** (Sequential + DA)
   
    arXiv 2022, [[PDF]](https://arxiv.org/pdf/2112.06460.pdf), [[Code]](https://github.com/juyongjiang/BiCAT)

5. **Counterfactual Data-Augmented Sequential Recommendation** (Sequential + Counterfactual + DA)
   
    SIGIR 2021, [[PDF]](https://arxiv.org/pdf/2207.02643.pdf)

6. **CauseRec: Counterfactual User Sequence Synthesis for Sequential Recommendation** (Sequential + Counterfactual + DA)
   
    SIGIR 2021, [[PDF]](https://arxiv.org/pdf/2109.05261.pdf)

7. **Explicit Counterfactual Data Augmentation for Recommendation** (Sequential + Counterfactual + DA)
   
    WSDM 2023

8. **Effective and Efficient Training for Sequential Recommendation using Recency Sampling** (Sequential + DA)
   
    RecSys 2022, [[PDF]](https://arxiv.org/pdf/2207.02643.pdf)

9. **Data Augmentation Strategies for Improving Sequential Recommender Systems** (Sequential + DA)

    arXiv 2022, [[PDF]](https://arxiv.org/pdf/2203.14037.pdf), [[Code]](https://github.com/saladsong/DataAugForSeqRec)

10. **Learning to Augment for Casual User Recommendation** (Sequential + DA)

    WWW 2022, [[PDF]](https://arxiv.org/pdf/2204.00926.pdf)

11. **Recency Dropout for Recurrent Recommender Systems** (RNN + DA)
   
     arXiv 2022, [[PDF]](https://arxiv.org/pdf/2201.11016.pdf)

12. **Improved Recurrent Neural Networks for Session-based Recommendations** (RNN + DA)

     DLRS 2016, [[PDF]](https://arxiv.org/pdf/1606.08117.pdf)

13. **Bootstrapping User and Item Representations for One-Class Collaborative Filtering** (CF + Graph + DA)

     SIGIR 2021, [[PDF]](https://arxiv.org/pdf/2105.06323.pdf), [[Code]](https://github.com/donalee/BUIR)

14. **MixGCF: An Improved Training Method for Graph Neural Network-based Recommender Systems** (Graph + DA)

     KDD 2021, [[PDF]](http://keg.cs.tsinghua.edu.cn/jietang/publications/KDD21-Huang-et-al-MixGCF.pdf), [[Code]](https://github.com/huangtinglin/MixGCF)

15. **Improving Recommendation Fairness via Data Augmentation** (Fairness + DA)

     WWW 2023, [[PDF]](https://arxiv.org/pdf/2302.06333.pdf), [[Code]](https://github.com/newlei/FDA)

16. **Fairly Adaptive Negative Sampling for Recommendations** (Fairness + DA)

     WWW 2023, [[PDF]](https://arxiv.org/pdf/2302.08266.pdf)

17. **Creating Synthetic Datasets for Collaborative Filtering Recommender  Systems using Generative Adversarial Networks** (CF + DA)

     arXiv 2023, [[PDF]](https://arxiv.org/pdf/2303.01297.pdf)


## Graph Models with CL

1. **Self-supervised Graph Learning for Recommendation** (Graph + CL + DA)
   
    SIGIR 2021, [[PDF]](https://arxiv.org/pdf/2010.10783.pdf), [[Code]](https://github.com/wujcan/SGL-Torch)

2. **Contrastive Graph Structure Learning via Information Bottleneck for Recommendation** (Graph + CL)
   
    NeurIPS 2022, [[PDF]](https://openreview.net/pdf?id=lhl_rYNdiH6), [[Code]](https://github.com/weicy15/CGI)

3. **Are graph augmentations necessary? simple graph contrastive learning for recommendation** (Graph + CL)
   
    SIGIR 2022, [[PDF]](https://arxiv.org/pdf/2112.08679.pdf), [[Code]](https://github.com/Coder-Yu/SELFRec/blob/main/model/graph/SimGCL.py)

4. **XSimGCL: Towards Extremely Simple Graph Contrastive Learning for Recommendation** (Graph + CL)
   
    TKDE 2022, [[PDF]](https://arxiv.org/pdf/2209.02544.pdf), [[Code]](https://github.com/Coder-Yu/SELFRec/blob/main/model/graph/XSimGCL.py)

5. **Intent-aware Multi-source Contrastive Alignment for Tag-enhanced Recommendation** (Graph + CL + DA)
   
    arXiv 2022, [[PDF]](https://arxiv.org/pdf/2211.06370.pdf)

6. **DisenPOI: Disentangling Sequential and Geographical Influence for Point-of-Interest Recommendation** (POI Rec, Graph + CL + DA)

    WSDM 2023, [[PDF]](https://arxiv.org/pdf/2210.16591.pdf), [[Code]](https://github.com/Fang6ang/DisenPOI)

7. **An MLP-based Algorithm for Efficient Contrastive Graph Recommendations** (Short paper, Graph + CL + DA)

    SIGIR 2022, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3477495.3531874)

8. **A Review-aware Graph Contrastive Learning Framework for Recommendation** (Graph + CL + DA)

    SIGIR 2022, [[PDF]](https://arxiv.org/pdf/2204.12063.pdf), [[Code]](https://github.com/JarenceSJ/ReviewGraph)

9. **Simple Yet Effective Graph Contrastive Learning for Recommendation** (Graph + CL + DA)

    ICLR 2023, [[PDF]](https://arxiv.org/pdf/2302.08191.pdf), [[Code]](https://github.com/HKUDS/LightGCL)

10. **Contrastive Meta Learning with Behavior Multiplicity for Recommendation** (Graph + CL + DA)

    WSDM 2022, [[PDF]](https://arxiv.org/pdf/2202.08523.pdf), [[Code]](https://github.com/weiwei1206/CML)

11. **Disentangled Contrastive Learning for Social Recommendation** (Short paper, Graph + CL + DA)

    CIKM 2022, [[PDF]](https://arxiv.org/pdf/2208.08723.pdf)

12. **Improving Knowledge-aware Recommendation with Multi-level Interactive Contrastive Learning** (Graph + CL)

    CIKM 2022, [[PDF]](https://arxiv.org/pdf/2208.10061.pdf), [[Code]](https://github.com/CCIIPLab/KGIC)

13. **Multi-level Cross-view Contrastive Learning for Knowledge-aware Recommender System** (Graph + CL)

    SIGIR 2022, [[PDF]](https://arxiv.org/pdf/2204.08807.pdf), [[Code]](https://github.com/CCIIPLab/MCCLK)

14. **Knowledge Graph Contrastive Learning for Recommendation** (Graph + DA + CL)

    SIGIR 2022, [[PDF]](https://arxiv.org/pdf/2205.00976.pdf), [[Code]](https://github.com/yuh-yang/KGCL-SIGIR22)

15. **Temporal Knowledge Graph Reasoning with Historical Contrastive Learning** (Graph + CL)

    IJCAI 2022, [[PDF]](https://arxiv.org/pdf/2211.10904.pdf), [[Code]](https://github.com/xyjigsaw/CENET)

16. **Self-Supervised Multi-Channel Hypergraph Convolutional Network for Social Recommendation** (Graph + SSL)

    WWW 2021, [[PDF]](https://arxiv.org/pdf/2101.06448.pdf), [[Code]](https://github.com/Coder-Yu/QRec)

17. **SAIL: Self-Augmented Graph Contrastive Learning** (Graph + CL)

    AAAI 2022, [[PDF]](https://arxiv.org/pdf/2009.00934.pdf)

18. **Predictive and Contrastive: Dual-Auxiliary Learning for Recommendation** (Graph + CL)

    arXiv 2022, [[PDF]](https://arxiv.org/pdf/2203.03982.pdf)

19. **Socially-Aware Self-Supervised Tri-Training for Recommendation** (Graph + CL)

    KDD 2021, [[PDF]](https://arxiv.org/pdf/2106.03569.pdf), [[Code]](https://github.com/Coder-Yu/QRec)

20. **Predictive and Contrastive: Dual-Auxiliary Learning for Recommendation** (Graph + CL)

    arXiv 2022, [[PDF]](https://arxiv.org/pdf/2203.03982.pdf)

21. **Multi-Behavior Dynamic Contrastive Learning for Recommendation** (Graph + CL)

    arXiv 2022, [[PDF]](https://arxiv.org/pdf/2203.03982.pdf)

22. **Self-Augmented Recommendation with Hypergraph Contrastive Collaborative Filtering** (Graph + CL)

    SIGIR 2022, [[PDF]](https://arxiv.org/pdf/2204.12200), [[Code]](https://github.com/akaxlh/HCCF)

23. **Improving Graph Collaborative Filtering with Neighborhood-enriched Contrastive Learning** (Graph + CF + CL)

    WWW 2022, [[PDF]](https://arxiv.org/pdf/2202.06200.pdf), [[Code]](https://github.com/RUCAIBox/NCL)

24. **Semi-deterministic and Contrastive Variational Graph Autoencoder for Recommendation** (Graph + CL)

    CIKM 2021, [[PDF]](https://dl.acm.org/doi/10.1145/3459637.3482390), [[Code]](https://github.com/syxkason/SCVG)

25. **Hypergraph Contrastive Collaborative Filtering** (Graph + CF + CL + DA)

    SIGIR 2022, [[PDF]](https://arxiv.org/pdf/2204.12200.pdf), [[Code]]( https://github.com/akaxlh/HCCF)

26. **Graph Structure Aware Contrastive Knowledge Distillation for Incremental Learning in Recommender Systems** (Short paper, Graph + CL)

    CIKM 2021, [[PDF]](https://dl.acm.org/doi/10.1145/3459637.3482117), [[Code]](https://github.com/syxkason/SCVG)

27. **Double-Scale Self-Supervised Hypergraph Learning for Group Recommendation** (Group Rec, Graph + CL + DA)

    CIKM 2021, [[PDF]](https://arxiv.org/abs/2109.04200), [[Code]](https://github.com/0411tony/HHGR)

28. **Self-Supervised Hypergraph Transformer for Recommender Systems** (Graph + SSL)

    KDD 2022, [[PDF]](https://arxiv.org/pdf/2207.14338.pdf), [[Code]](https://github.com/akaxlh/SHT)

29. **Episodes Discovery Recommendation with Multi-Source Augmentations** (Graph + DA + CL)

     arXiv 2023, [[PDF]](https://arxiv.org/pdf/2301.01737.pdf)

30. **Poincaré Heterogeneous Graph Neural Networks for Sequential Recommendation** (Graph + Sequential + CL)

     TOIS 2023, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3568395)
    
31. **Adversarial Learning Data Augmentation for Graph Contrastive Learning in Recommendation** (Graph + DA + CL)

     DASFAA 2023, [[PDF]](https://arxiv.org/pdf/2302.02317.pdf)
    
32. **SimCGNN: Simple Contrastive Graph Neural Network for Session-based Recommendation** (Graph + CL)

     arXiv 2023, [[PDF]](https://arxiv.org/pdf/2302.03997.pdf)
    
33. **MA-GCL: Model Augmentation Tricks for Graph Contrastive Learning** (Graph + DA + CL)

     AAAI 2023, [[PDF]](https://arxiv.org/pdf/2212.07035.pdf), [[Code]](https://github.com/GXM1141/MA-GCL)
    
34. **Self-Supervised Hypergraph Convolutional Networks for Session-based Recommendation** (Graph + Session + CL)

     AAAI 2021, [[PDF]](https://arxiv.org/pdf/2012.06852.pdf), [[Code]](https://github.com/xiaxin1998/DHCN)
    
35. **Self-Supervised Graph Co-Training for Session-based Recommendation** (Graph + Session + CL)

     CIMK 2021, [[PDF]](https://arxiv.org/pdf/2108.10560.pdf), [[Code]](https://github.com/xiaxin1998/COTREC)

36. **Heterogeneous Graph Contrastive Learning for Recommendation** (Graph + CL)

     WSDM 2023, [[PDF]](https://arxiv.org/pdf/2303.00995.pdf), [[Code]](https://github.com/HKUDS/HGCL)

37. **Automated Self-Supervised Learning for Recommendation** (Graph + DA + CL)

     WWW 2023, [[PDF]](https://arxiv.org/pdf/2303.07797.pdf), [[Code]](https://github.com/HKUDS/AutoCF)

37. **Graph-less Collaborative Filtering** (Graph + CL)

     WWW 2023, [[PDF]](https://arxiv.org/pdf/2303.08537.pdf), [[Code]](https://github.com/HKUDS/SimRec)


## Sequential Models with CL

1. **Uniform Sequence Better: Time Interval Aware Data Augmentation for Sequential Recommendation** (Sequential + CL + DA)

    AAAI 2023, [[PDF]](https://arxiv.org/pdf/2212.08262.pdf), [[Code]](https://github.com/KingGugu/TiCoSeRec)

2. **Contrastive Learning for Sequential Recommendation** (Sequential + CL + DA)
   
    SIGIR 2021, [[PDF]](https://arxiv.org/pdf/2010.14395.pdf), [[Code]](https://github.com/RUCAIBox/RecBole-DA/blob/master/recbole/model/sequential_recommender/cl4srec.py)

3. **Contrastive Self-supervised Sequential Recommendation with Robust Augmentation** (Sequential + CL + DA)
   
    SIGIR 2021, [[PDF]](https://arxiv.org/pdf/2108.06479.pdf), [[Code]](https://github.com/YChen1993/CoSeRec)

4. **Learnable Model Augmentation Self-Supervised Learning for Sequential Recommendation** (Sequential + CL + DA)

    arXiv 2022, [[PDF]](https://arxiv.org/pdf/2204.10128.pdf)

5. **S3-Rec: Self-Supervised Learning for Sequential Recommendation with Mutual Information Maximization** (Sequential + CL + DA)

    CIKM 2020, [[PDF]](https://arxiv.org/pdf/2008.07873.pdf), [[Code]](https://github.com/RUCAIBox/CIKM2020-S3Rec)

6. **Contrastive Curriculum Learning for Sequential User Behavior Modeling via Data Augmentation** (Sequential + CL + DA)

    CIKM 2021, [[PDF]](https://www.atailab.cn/seminar2022Spring/pdf/2021_CIKM_Contrastive%20Curriculum%20Learning%20for%20Sequential%20User%20Behavior%20Modeling%20via%20Data%20Augmentation.pdf) , [[Code]](https://github.com/RUCAIBox/Contrastive-Curriculum-Learning)

7. **Contrastive Learning for Representation Degeneration Problem in Sequential Recommendation** (Sequential + CL + DA)

    WSDM 2022, [[PDF]](https://arxiv.org/pdf/2110.05730.pdf), [[Code]](https://github.com/RuihongQiu/DuoRec)

8. **Memory Augmented Multi-Instance Contrastive Predictive Coding for Sequential Recommendation** (Sequential + CL + DA)

   arXiv 2021, [[PDF]](https://arxiv.org/pdf/2109.00368.pdf)

9. **Contrastive Learning with Bidirectional Transformers for Sequential Recommendation** (Sequential + CL + DA)

   CIKM 2022, [[PDF]](https://arxiv.org/pdf/2208.03895.pdf), [[Code]](https://github.com/hw-du/CBiT/tree/master)

10. **ContrastVAE: Contrastive Variational AutoEncoder for Sequential Recommendation** (Sequential + CL + DA)

    CIKM 2022, [[PDF]](https://arxiv.org/pdf/2209.00456.pdf), [[Code]](https://github.com/YuWang-1024/ContrastVAE)

11. **Temporal Contrastive Pre-Training for Sequential Recommendation** (Sequential + CL + DA)

    CIKM 2022, [[PDF]](https://dl.acm.org/doi/10.1145/3511808.3557468), [[Code]](https://github.com/ChangxinTian/TCP-SRec)

12. **Multi-level Contrastive Learning Framework for Sequential Recommendation** (Graph + Sequential + CL)

    CIKM 2022, [[PDF]](https://arxiv.org/pdf/2208.13007.pdf)

13. **Equivariant Contrastive Learning for Sequential Recommendation** (Sequential + DA + CL)

    arXiv 2022, [[PDF]](https://arxiv.org/pdf/2211.05290.pdf)

14. **Explanation Guided Contrastive Learning for Sequential Recommendation** (Sequential + DA + CL)

    CIKM 2022, [[PDF]](https://arxiv.org/pdf/2209.01347.pdf), [[Code]](https://github.com/demoleiwang/EC4SRec)

15. **Intent Contrastive Learning for Sequential Recommendation** (Sequential + DA + CL)

    WWW 2022, [[PDF]](https://arxiv.org/pdf/2202.02519.pdf), [[Code]](https://github.com/salesforce/ICLRec)

16. **Dual Contrastive Network for Sequential Recommendation** (Short paper, Sequential + CL)

    SIGIR 2022, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3477495.3531918)

17. **Dual Contrastive Network for Sequential Recommendation with User and Item-Centric Perspectives** (Sequential + CL)

    arXiv 2022, [[PDF]](https://arxiv.org/pdf/2209.08446.pdf)

18. **Enhancing Sequential Recommendation with Graph Contrastive Learning** (Sequential + Graph + CL + DA)

    IJCAI 2022, [[PDF]](https://arxiv.org/pdf/2205.14837.pdf), [[Code]](https://github.com/sdu-zyx/GCL4SR)

19. **Disentangling Long and Short-Term Interests for Recommendation** (Sequential + Graph + CL)

    WWW 2022, [[PDF]](https://arxiv.org/pdf/2202.13090.pdf), [[Code]](https://github.com/tsinghua-fib-lab/CLSR)

20. **Hyperbolic Hypergraphs for Sequential Recommendation** (Sequential + Graph + CL + DA)

    CIKM 2021, [[PDF]](https://arxiv.org/pdf/2108.08134.pdf), [[Code]](https://github.com/Abigale001/h2seqrec)
    
21. **Mutual Wasserstein Discrepancy Minimization for Sequential  Recommendation** (Sequential + CL)

    WWW 2023, [[PDF]](https://arxiv.org/pdf/2301.12197.pdf), [[Code]](https://github.com/zfan20/MStein)
    
22. **Dual-interest Factorization-heads Attention for Sequential Recommendation** (Sequential + CL)

    WWW 2023, [[PDF]](https://arxiv.org/pdf/2302.03965.pdf), [[Code]](https://github.com/tsinghua-fib-lab/WWW2023-DFAR)

23. **GUESR: A Global Unsupervised Data-Enhancement with Bucket-Cluster  Sampling for Sequential Recommendation** (Sequential + DA + CL)

    arXiv 2023, [[PDF]](https://arxiv.org/pdf/2303.00243.pdf)

24. **Self-Supervised Interest Transfer Network via Prototypical Contrastive  Learning for Recommendation** (Sequential + CL)

    AAAI 2023, [[PDF]](https://arxiv.org/pdf/2302.14438.pdf), [[Code]](https://github.com/fanqieCoffee/SITN-Supplement)

25. **A Self-Correcting Sequential Recommender** (Sequential + DA + SSL)

    WWW 2023, [[PDF]](https://arxiv.org/pdf/2303.02297.pdf), [[Code]](https://github.com/TempSDU/STEAM)

26. **User Retention-oriented Recommendation with Decision Transformer** (Sequential + CL)

    WWW 2023, [[PDF]](https://arxiv.org/pdf/2303.06347.pdf), [[Code]](https://github.com/kesenzhao/DT4Rec)

27. **Debiased Contrastive Learning for Sequential Recommendation** (Sequential + DA + CL)

    WWW 2023, [[PDF]](https://arxiv.org/pdf/2303.11780.pdf), [[Code]](https://github.com/kesenzhao/DT4Rec)


## Other Tasks with CL

1. **CL4CTR: A Contrastive Learning Framework for CTR Prediction** (CTR + CL)

    WSDM 2023, [[PDF]](https://arxiv.org/pdf/2212.00522.pdf), [[Code]](https://github.com/cl4ctr/cl4ctr)

2. **CCL4Rec: Contrast over Contrastive Learning for Micro-video Recommendation** (Micro-video + CL)

    arXiv 2022, [[PDF]](https://arxiv.org/pdf/2208.08024.pdf)

3. **Re4: Learning to Re-contrast, Re-attend, Re-construct for Multi-interest Recommendation** (Multi-interest + CL)

    WWW 2022, [[PDF]](https://arxiv.org/pdf/2208.08011.pdf), [[Code]](https://github.com/DeerSheep0314/Re4-Learning-to-Re-contrast-Re-attend-Re-construct-for-Multi-interest-Recommendation)

4. **Interventional Recommendation with Contrastive Counterfactual Learning for Better Understanding User Preferences** (Counterfactual + DA + CL)

    arXiv 2022, [[PDF]](https://arxiv.org/pdf/2208.06746.pdf)

5. **Multi-granularity Item-based Contrastive Recommendation** (Industry + CL)

    arXiv 2022, [[PDF]](https://arxiv.org/pdf/2207.01387.pdf)

6. **Improving Micro-video Recommendation via Contrastive Multiple Interests** (Short paper, Micro-video + CL)

    SIGIR 2022, [[PDF]](https://arxiv.org/pdf/2205.09593.pdf)

7. **Exploiting Negative Preference in Content-based Music Recommendation with Contrastive Learning** (Music Rec + CL)

    RecSys 2022, [[PDF]](https://arxiv.org/pdf/2103.09410.pdf), [[Code]](https://github.com/Spijkervet/CLMR)

8. **Self-supervised Learning for Large-scale Item Recommendations** (Industry + CL + DA)

    CIKM 2021, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3459637.3481952)

9. **CrossCBR: Cross-view Contrastive Learning for Bundle Recommendation** (Bundle Rec + CL)

    KDD 2023, [[PDF]](https://arxiv.org/pdf/2206.00242.pdf), [[Code]](https://github.com/mysbupt/CrossCBR)

10. **Contrastive Learning for Cold-start Recommendation** (Short paper, Cold Start + CL)

    ACM MM (ACM International Conference on Multimedia) 2021, [[PDF]](https://arxiv.org/pdf/2107.05315v1.pdf), [[Code]](https://github.com/weiyinwei/CLCRec)

11. **Socially-aware Dual Contrastive Learning for Cold-Start Recommendation** (Short paper, Cold Start + CL)

    SIGIR 2022, [[PDF]](https://dl.acm.org/doi/10.1145/3477495.3531780)

12. **Multi-modal Graph Contrastive Learning for Micro-video Recommendation** (Short paper, Cold Start + Graph + CL)

    SIGIR 2022, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3477495.3531780)

13. **Self-supervised Learning for Multimedia Recommendation** (Multimedia Rec + Graph + DA + CL)

    TMM (IEEE Transactions on Multimedia) 2022, [[PDF]](https://arxiv.org/pdf/2107.05315v1.pdf), [[Code]](https://github.com/zltao/SLMRec/)

14. **SelfCF: A Simple Framework for Self-supervised Collaborative Filtering** (CF + Graph + DA + CL)

    ACM MM (ACM International Conference on Multimedia) 2021, [[PDF]](https://arxiv.org/pdf/2107.03019.pdf), [[Code]](https://github.com/enoche/SelfCF)

15. **Trading Hard Negatives and True Negatives:A Debiased Contrastive Collaborative Filtering Approach** (CF + CL)

    IJCAI 2022, [[PDF]](https://arxiv.org/pdf/2204.11752.pdf)

16. **The World is Binary: Contrastive Learning for Denoising Next Basket Recommendation** (Next Basket + CL)

    SIGIR 2021, [[PDF]](https://dl.acm.org/doi/10.1145/3404835.3462836)

17. **MIC: Model-agnostic Integrated Cross-channel Recommender** (Industry + CL + DA)

     CIKM 2022, [[PDF]](https://arxiv.org/pdf/2110.11570.pdf)

18. **A Contrastive Sharing Model for Multi-Task Recommendation** (Multi-Task + CL)

     WWW 2022, [[PDF]](https://dl.acm.org/doi/10.1145/3485447.3512043)

19. **C2-CRS: Coarse-to-Fine Contrastive Learning for Conversational Recommender System** (Conversational Rec + CL)

     WSDM 2022, [[PDF]](https://arxiv.org/pdf/2201.02732.pdf), [[Code]](https://github.com/RUCAIBox/WSDM2022-C2CRS)

20. **Contrastive Cross-domain Recommendation in Matching** (Cross-domain Rec + DA + CL)

     KDD 2022, [[PDF]](https://arxiv.org/pdf/2112.00999.pdf), [[Code]](https://github.com/lqfarmer/CCDR)

21. **Contrastive Cross-Domain Sequential Recommendation** (Cross-domain Rec + Sequential + CL)

     CIKM 2022, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3511808.3557262), [[Code]](https://github.com/cjx96/C2DSR)

22. **Prototypical Contrastive Learning and Adaptive Interest Selection for Candidate Generation in Recommendations** (Short Paper, Industry + CL + DA)

     CIKM 2022, [[PDF]](https://arxiv.org/pdf/2211.12893.pdf), [[Code]](https://github.com/cjx96/C2DSR)

23. **Spatio-Temporal Contrastive Learning Enhanced GNNs for Session-based Recommendation** (GNN + CL)

     TOIS 2022, under review, [[PDF]](https://arxiv.org/pdf/2209.11461v2.pdf)
    
24. **Disentangled Causal Embedding With Contrastive Learning For Recommender System** (Causal + CL)

     arXiv 2023, [[PDF]](https://arxiv.org/pdf/2302.03248.pdf), [[Code]](https://github.com/somestudies/DCCL)

25. **Contrastive Collaborative Filtering for Cold-Start Item
    Recommendation** (CF + Cold Start +  CL)

     WWW 2023, [[PDF]](https://arxiv.org/pdf/2302.02151.pdf), [[Code]](https://github.com/zzhin/CCFCRec)
    
26. **Cross-domain recommendation via user interest alignment** (Cross-domain Rec + CL)

     WWW 2023, [[PDF]](https://arxiv.org/pdf/2301.11467.pdf), [[Code]](https://github/anonymous/COAST)
    
27. **Multi-Modal Self-Supervised Learning for Recommendation** (Multi-Modal Rec + CL)

     WWW 2023, [[PDF]](https://arxiv.org/pdf/2302.10632.pdf), [[Code]](https://github.com/HKUDS/MMSSL)
    
28. **Efficient On-Device Session-Based Recommendation** (Session + DA + CL)

     TOIS 2023, [[PDF]](https://arxiv.org/pdf/2209.13422.pdf), [[Code]](https://github.com/xiaxin1998/EODRec)

29. **On-Device Next-Item Recommendation with Self-Supervised Knowledge Distillation** (Session + DA + CL)

     SIGIR 2022, [[PDF]](https://arxiv.org/pdf/2204.11091.pdf), [[Code]](https://github.com/xiaxin1998/OD-Rec)

30. **Modality Matches Modality: Pretraining Modality-Disentangled Item Representations for Recommendation** (Multi-Modal Rec + CL)

     WWW 2022, [[PDF]](https://web.archive.org/web/20220428140054id_/https://dl.acm.org/doi/pdf/10.1145/3485447.3512079), [[Code]](https://github.com/hantengyue/PAMD)