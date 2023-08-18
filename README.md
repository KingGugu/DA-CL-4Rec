# Awesome Contrastive Learning & Data Augmentation RS Paper & Code

This repository collects the latest research progress of **Contrastive Learning (CL) and Data Augmentation (DA)** in Recommender Systems.
Comments and contributions are welcome.

CF = Collaborative Filtering, SSL = Self-Supervised Learning

- [Survey/Tutorial](#Survey/Tutorial) Total Papers: 4
- [Only Data Augmentation](#Only-Data-Augmentation) Total Papers: 24
- [Graph Models with CL](#Graph-Models-with-CL) Total Papers: 53
- [Sequential Models with CL](#Sequential-Models-with-CL) Total Papers: 51
- [Other Tasks with CL](#Other-Tasks-with-CL) Total Papers: 53


## Survey/Tutorial
1. **Contrastive Self-supervised Learning in Recommender Systems: A Survey** (Survey)
   
   arXiv 2023, [[PDF]](https://arxiv.org/pdf/2303.09902.pdf)

2. **Self-Supervised Learning for Recommender Systems A Survey** (Survey)
   
   TKDE 2022, [[PDF]](https://arxiv.org/pdf/2203.15876.pdf), [[Code]](https://github.com/Coder-Yu/SELFRec)

3. **Self-Supervised Learning in Recommendation: Fundamentals and Advances** (Tutorial)
   
   WWW 2022, [[Web]](https://ssl-recsys.github.io/)
   
4. **Tutorial: Self-Supervised Learning for Recommendation: Foundations, Methods and Prospects** (Tutorial)
   
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

7. **Effective and Efficient Training for Sequential Recommendation using Recency Sampling** (Sequential + DA)
   
    RecSys 2022, [[PDF]](https://arxiv.org/pdf/2207.02643.pdf)

8. **Data Augmentation Strategies for Improving Sequential Recommender Systems** (Sequential + DA)

    arXiv 2022, [[PDF]](https://arxiv.org/pdf/2203.14037.pdf), [[Code]](https://github.com/saladsong/DataAugForSeqRec)

9. **Learning to Augment for Casual User Recommendation** (Sequential + DA)

    WWW 2022, [[PDF]](https://arxiv.org/pdf/2204.00926.pdf)

10. **Recency Dropout for Recurrent Recommender Systems** (RNN + DA)
   
     arXiv 2022, [[PDF]](https://arxiv.org/pdf/2201.11016.pdf)

11. **Improved Recurrent Neural Networks for Session-based Recommendations** (RNN + DA)

     DLRS 2016, [[PDF]](https://arxiv.org/pdf/1606.08117.pdf)

12. **Bootstrapping User and Item Representations for One-Class Collaborative Filtering** (CF + Graph + DA)

     SIGIR 2021, [[PDF]](https://arxiv.org/pdf/2105.06323.pdf), [[Code]](https://github.com/donalee/BUIR)

13. **MixGCF: An Improved Training Method for Graph Neural Network-based Recommender Systems** (Graph + DA)

     KDD 2021, [[PDF]](http://keg.cs.tsinghua.edu.cn/jietang/publications/KDD21-Huang-et-al-MixGCF.pdf), [[Code]](https://github.com/huangtinglin/MixGCF)

14. **Improving Recommendation Fairness via Data Augmentation** (Fairness + DA)

     WWW 2023, [[PDF]](https://arxiv.org/pdf/2302.06333.pdf), [[Code]](https://github.com/newlei/FDA)

15. **Fairly Adaptive Negative Sampling for Recommendations** (Fairness + DA)

     WWW 2023, [[PDF]](https://arxiv.org/pdf/2302.08266.pdf)

16. **Creating Synthetic Datasets for Collaborative Filtering Recommender  Systems using Generative Adversarial Networks** (CF + DA)

     arXiv 2023, [[PDF]](https://arxiv.org/pdf/2303.01297.pdf)

17. **Graph Collaborative Signals Denoising and Augmentation for Recommendation** (CF + DA)

     SIGIR 2023, [[PDF]](https://arxiv.org/pdf/2304.03344.pdf), [[Code]](https://github.com/zfan20/GraphDA)

18. **Data Augmented Sequential Recommendation based on Counterfactual Thinking** (Sequential + DA)

    TKDE 2022, [[PDF]](https://ieeexplore.ieee.org/abstract/document/9950302)

19. **Data Augmented Sequential Recommendation based on Counterfactual Thinking** (CRT + DA)

    arXiv 2023, [[PDF]](https://arxiv.org/pdf/2305.19531.pdf)

20. **Improving Conversational Recommendation Systems via Counterfactual Data Simulation** (Conversational Rec + DA)

    KDD 2023, [[PDF]](https://arxiv.org/pdf/2306.02842.pdf), [[Code]](https://github.com/RUCAIBox/CFCRS)

21. **Disentangled Variational Auto-encoder Enhanced by Counterfactual Data for Debiasing Recommendation** (Debias Rec + DA)

    arXiv 2023, [[PDF]](https://arxiv.org/pdf/2306.15961.pdf)

22. **Domain Disentanglement with Interpolative Data Augmentation for Dual-Target Cross-Domain Recommendation** (Cross-domain + DA)

    RecSys 2023, [[PDF]](https://arxiv.org/pdf/2307.13910.pdf)

23. **Scaling Session-Based Transformer Recommendations using Optimized Negative Sampling and Loss Functions** (Session + DA)

    RecSys 2023, [[PDF]](https://arxiv.org/pdf/2307.14906.pdf), [[Code]](https://github.com/otto-de/TRON)

24. **Intrinsically Motivated Reinforcement Learning based Recommendation with Counterfactual Data Augmentation** (RL Rec + DA)

     arXiv 2022, [[PDF]](https://arxiv.org/pdf/2209.08228.pdf)


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

38. **Graph-less Collaborative Filtering** (Graph + CL)

     WWW 2023, [[PDF]](https://arxiv.org/pdf/2303.08537.pdf), [[Code]](https://github.com/HKUDS/SimRec)

39. **Disentangled Contrastive Collaborative Filtering** (Graph + CL)

     SIGIR 2023, [[PDF]](https://arxiv.org/pdf/2305.02759.pdf), [[Code]](https://github.com/HKUDS/DCCF)

40. **Knowledge-refined Denoising Network for Robust Recommendation** (Graph + CL)

     SIGIR 2023, [[PDF]](https://arxiv.org/pdf/2304.14987.pdf), [[Code]](https://github.com/xj-zhu98/KRDN)

41. **Disentangled Graph Contrastive Learning for Review-based Recommendation** (Graph + CL)

     IJCAI 2023, [[PDF]](https://arxiv.org/pdf/2209.01524.pdf)

42. **Adaptive Graph Contrastive Learning for Recommendation** (Graph + CL)

     SIGIR 2023, [[PDF]](https://arxiv.org/abs/2305.10837), [[Code]](https://github.com/ZzMeei/AdaptiveGCL)

43. **Knowledge Enhancement for Contrastive Multi-Behavior Recommendation** (Graph + CL)

     WSDM 2023, [[PDF]](https://arxiv.org/pdf/2301.05403.pdf), [[Code]](https://github.com/HKUDS/SSLRec)

44. **Contrastive Meta Learning with Behavior Multiplicity for Recommendation** (Graph + CL)

     WSDM 2022, [[PDF]](https://arxiv.org/pdf/2202.08523.pdf), [[Code]](https://github.com/weiwei1206/CML)

45. **Graph Transformer for Recommendation** (Graph + CL)

     SIGIR 2023, [[PDF]](https://arxiv.org/pdf/2306.02330.pdf), [[Code]](https://github.com/HKUDS/GFormer)

46. **PANE-GNN: Unifying Positive and Negative Edges in Graph Neural Networks for Recommendation** (Graph + CL)

     CIKM 2023, [[PDF]](https://arxiv.org/pdf/2306.04095.pdf)

47. **Knowledge Graph Self-Supervised Rationalization for Recommendation** (Graph + CL)

     KDD 2023, [[PDF]](https://arxiv.org/pdf/2307.02759.pdf), [[Code]](https://github.com/HKUDS/KGRec)

48. **Enhanced Graph Learning for Collaborative Filtering via Mutual Information Maximization** (Graph + CL)

     SIGIR 2021, [[PDF]](https://dl.acm.org/doi/abs/10.1145/3404835.3462928)

49. **Generative-Contrastive Graph Learning for Recommendation** (Graph + CL)

     SIGIR 2023, [[PDF]](https://arxiv.org/pdf/2307.05100.pdf)

50. **AdaMCL: Adaptive Fusion Multi-View Contrastive Learning for Collaborative Filtering** (Graph + CL)

     SIGIR 2023, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3539618.3591632), [[Code]](https://github.com/PasaLab/AdaMCL)

51. **Candidate–aware Graph Contrastive Learning for Recommendation** (Graph + CL)

     SIGIR 2023, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3539618.3591647), [[Code]](https://github.com/WeiHeCnSH/CGCL-Pytorch-master)

52. **Multi-View Graph Convolutional Network for Multimedia Recommendation** (Graph + CL)

     MM 2023, [[PDF]](https://arxiv.org/ftp/arxiv/papers/2308/2308.03588.pdf), [[Code]](https://github.com/demonph10/MGCN)

53. **Uncertainty-aware Consistency Learning for Cold-Start Item Recommendation** (Graph + CL)

     SIGIR 2023, [[PDF]](https://arxiv.org/pdf/2308.03470.pdf)


## Sequential Models with CL

1. **Uniform Sequence Better: Time Interval Aware Data Augmentation for Sequential Recommendation** (Sequential + CL + DA)

    AAAI 2023, [[PDF]](https://arxiv.org/pdf/2212.08262.pdf), [[Code]](https://github.com/KingGugu/TiCoSeRec)

2. **Contrastive Learning for Sequential Recommendation** (Sequential + CL + DA)
   
    ICDE 2022, [[PDF]](https://arxiv.org/pdf/2010.14395.pdf), [[Code]](https://github.com/RUCAIBox/RecBole-DA/blob/master/recbole/model/sequential_recommender/cl4srec.py)

3. **Contrastive Self-supervised Sequential Recommendation with Robust Augmentation** (Sequential + CL + DA)
   
    arXiv 2021, [[PDF]](https://arxiv.org/pdf/2108.06479.pdf), [[Code]](https://github.com/YChen1993/CoSeRec)

4. **Learnable Model Augmentation Self-Supervised Learning for Sequential Recommendation** (Sequential + CL + DA)

    arXiv 2022, [[PDF]](https://arxiv.org/pdf/2204.10128.pdf)

5. **S3-Rec: Self-Supervised Learning for Sequential Recommendation with Mutual Information Maximization** (Sequential + CL + DA)

    CIKM 2020, [[PDF]](https://arxiv.org/pdf/2008.07873.pdf), [[Code]](https://github.com/RUCAIBox/CIKM2020-S3Rec)

6. **Contrastive Curriculum Learning for Sequential User Behavior Modeling via Data Augmentation** (Sequential + CL + DA)

    CIKM 2021, [[PDF]](https://www.atailab.cn/seminar2022Spring/pdf/2021_CIKM_Contrastive%20Curriculum%20Learning%20for%20Sequential%20User%20Behavior%20Modeling%20via%20Data%20Augmentation.pdf) , [[Code]](https://github.com/RUCAIBox/Contrastive-Curriculum-Learning)

7. **Contrastive Learning for Representation Degeneration Problem in Sequential Recommendation** (Sequential + CL + DA)

    WSDM 2022, [[PDF]](https://arxiv.org/pdf/2110.05730.pdf), [[Code]](https://github.com/RuihongQiu/DuoRec)

8. **Memory Augmented Multi-Instance Contrastive Predictive Coding for Sequential Recommendation** (Sequential + CL + DA)

   ICDM 2021, [[PDF]](https://arxiv.org/pdf/2109.00368.pdf)

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

23. **GUESR: A Global Unsupervised Data-Enhancement with Bucket-Cluster Sampling for Sequential Recommendation** (Sequential + DA + CL)

    arXiv 2023, [[PDF]](https://arxiv.org/pdf/2303.00243.pdf)

24. **Self-Supervised Interest Transfer Network via Prototypical Contrastive  Learning for Recommendation** (Sequential + CL)

    AAAI 2023, [[PDF]](https://arxiv.org/pdf/2302.14438.pdf), [[Code]](https://github.com/fanqieCoffee/SITN-Supplement)

25. **A Self-Correcting Sequential Recommender** (Sequential + DA + SSL)

    WWW 2023, [[PDF]](https://arxiv.org/pdf/2303.02297.pdf), [[Code]](https://github.com/TempSDU/STEAM)

26. **User Retention-oriented Recommendation with Decision Transformer** (Sequential + CL)

    WWW 2023, [[PDF]](https://arxiv.org/pdf/2303.06347.pdf), [[Code]](https://github.com/kesenzhao/DT4Rec)

27. **Debiased Contrastive Learning for Sequential Recommendation** (Sequential + DA + CL)

    WWW 2023, [[PDF]](https://arxiv.org/pdf/2303.11780.pdf), [[Code]](https://github.com/HKUDS/DCRec)

28. **Learning Vector-Quantized Item Representation for Transferable Sequential Recommenders** (Sequential + CL)

    WWW 2023, [[PDF]](https://arxiv.org/pdf/2210.12316.pdf), [[Code]](https://github.com/RUCAIBox/VQ-Rec)

29. **Multimodal Pre-training Framework for Sequential Recommendation via Contrastive Learning** (Multi-Modal + Sequential + CL)

    arXiv 2023, [[PDF]](https://arxiv.org/pdf/2303.11879.pdf)

30. **Sequential Recommendation with Diffusion Models** (Diffsion + Sequential + CL)

    arXiv 2023, [[PDF]](https://arxiv.org/pdf/2304.04541.pdf)

31. **Triple Sequence Learning for Cross-domain Recommendation** (Sequential + CL)

    arXiv 2023, [[PDF]](https://arxiv.org/pdf/2304.05027.pdf)

32. **Contrastive Cross-Domain Sequential Recommendation** (Cross-domain + Sequential + CL)

    CIMK 2022, [[PDF]](https://arxiv.org/pdf/2304.03891.pdf), [[Code]](https://github.com/cjx96/C2DSR)

33. **Adversarial and Contrastive Variational Autoencoder for Sequential Recommendation** (VAE + Sequential + CL)

    WWW 2021, [[PDF]](https://arxiv.org/pdf/2103.10693.pdf), [[Code]](https://github.com/ACVAE/ACVAE-PyTorch)

34. **Meta-optimized Contrastive Learning for Sequential Recommendation** (Meta + Sequential + CL)

    SIGIR 2023, [[PDF]](https://arxiv.org/pdf/2304.07763.pdf), [[Code]](https://github.com/QinHsiu/MCLRec)

35. **Frequency Enhanced Hybrid Attention Network for Sequential Recommendation** (Sequential + CL)

    SIGIR 2023, [[PDF]](https://arxiv.org/pdf/2304.09184.pdf), [[Code]](https://github.com/sudaada/FEARec)

36. **Self-Supervised Multi-Modal Sequential Recommendation** (Multi-Moda + Sequential + CL)

    arXiv 2023, [[PDF]](https://arxiv.org/pdf/2304.13277.pdf), [[Code]](https://github.com/kz-song/MMSRec)

37. **Conditional Denoising Diffusion for Sequential Recommendation** (Diffusion + Sequential + CL)

    arXiv 2023, [[PDF]](https://arxiv.org/pdf/2304.11433.pdf)

38. **Ensemble Modeling with Contrastive Knowledge Distillation for Sequential Recommendation** (Diffusion + Sequential + CL)

    SIGIR 2023, [[PDF]](https://arxiv.org/pdf/2304.14668.pdf), [[Code]](https://github.com/hw-du/EMKD)

39. **Multi-view Multi-behavior Contrastive Learning in Recommendation** (Sequential + Graph + CL)

    DASFAA 2022, [[PDF]](https://arxiv.org/pdf/2203.10576.pdf), [[Code]](https://github.com/wyqing20/MMCLR)

40. **Denoising Multi-modal Sequential Recommenders with Contrastive Learning** (Sequential + CL)

    arXiv 2023, [[PDF]](https://arxiv.org/pdf/2305.01915.pdf)

41. **Multi-view Multi-behavior Contrastive Learning in Recommendation** (Sequential + Graph + CL)

    SIGIR 2022, [[PDF]](https://arxiv.org/pdf/2305.04619.pdf), [[Code]](https://github.com/HKUDS/MAERec)

42. **Contrastive Enhanced Slide Filter Mixer for Sequential Recommendation** (Sequential + CL)

    ICDE 2023, [[PDF]](https://arxiv.org/pdf/2305.04322.pdf), [[Code]](https://github.com/sudaada/SLIME4Rec)

43. **Contrastive State Augmentations for Reinforcement Learning-Based Recommender Systems** (Sequential + DA + CL)

    SIGIR 2023, [[PDF]](https://arxiv.org/pdf/2305.11081.pdf), [[Code]](https://github.com/HN-RS)

44. **When Search Meets Recommendation: Learning Disentangled Search Representation for Recommendation** (Sequential + CL)

    SIGIR 2023, [[PDF]](https://arxiv.org/pdf/2305.10822.pdf), [[Code]](https://github.com/Ethan00Si/SESREC-SIGIR-2023)

45. **Text Is All You Need: Learning Language Representations for Sequential Recommendation** (Sequential + CL)

    KDD 2023, [[PDF]](https://arxiv.org/pdf/2305.13731.pdf)

46. **Text Is All You Need: Learning Language Representations for Sequential Recommendation** (Sequential + CL)

    TOIS 2022, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3522673), [[Code]](https://github.com/THUwangcy/ReChorus/tree/TOIS22)

47. **Robust Reinforcement Learning Objectives for Sequential Recommender Systems** (Sequential + CL)

    arXiv 2023, [[PDF]](https://arxiv.org/pdf/2305.18820.pdf), [[Code]](https://github.com/melfm/sasrec-ccql)

48. **AdaptiveRec: Adaptively Construct Pairs for Contrastive Learning in Sequential Recommendation** (Sequential + CL)

    PMLR 2023, [[PDF]](https://arxiv.org/pdf/2307.05469.pdf)

49. **Fisher-Weighted Merge of Contrastive Learning Models in Sequential Recommendation** (Sequential + CL)

    PMLR 2023, [[PDF]](https://arxiv.org/pdf/2307.05476.pdf)

50. **Hierarchical Contrastive Learning with Multiple Augmentation for Sequential Recommendation** (Sequential + DA + CL)

    arXiv 2023, [[PDF]](https://arxiv.org/pdf/2308.03400.pdf)

51. **Poisoning Self-supervised Learning Based Sequential Recommendations** (Sequential + Attack + DA + CL)

    SIGIR 2023, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3539618.3591751), [[Code]](https://github.com/CongGroup/Poisoning-SSL-based-RS)


## Other Tasks with CL

1. **CL4CTR: A Contrastive Learning Framework for CTR Prediction** (CTR + CL)

    WSDM 2023, [[PDF]](https://arxiv.org/pdf/2212.00522.pdf), [[Code]](https://github.com/cl4ctr/cl4ctr)

2. **CCL4Rec: Contrast over Contrastive Learning for Micro-video Recommendation** (Micro Video + CL)

    arXiv 2022, [[PDF]](https://arxiv.org/pdf/2208.08024.pdf)

3. **Re4: Learning to Re-contrast, Re-attend, Re-construct for Multi-interest Recommendation** (Multi Interest + CL)

    WWW 2022, [[PDF]](https://arxiv.org/pdf/2208.08011.pdf), [[Code]](https://github.com/DeerSheep0314/Re4-Learning-to-Re-contrast-Re-attend-Re-construct-for-Multi-interest-Recommendation)

4. **Interventional Recommendation with Contrastive Counterfactual Learning for Better Understanding User Preferences** (Counterfactual + DA + CL)

    arXiv 2022, [[PDF]](https://arxiv.org/pdf/2208.06746.pdf)

5. **Multi-granularity Item-based Contrastive Recommendation** (Industry + CL)

    arXiv 2022, [[PDF]](https://arxiv.org/pdf/2207.01387.pdf)

6. **Improving Micro-video Recommendation via Contrastive Multiple Interests** (Short paper, Micro Video + CL)

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

13. **Self-supervised Learning for Multimedia Recommendation** (Multi Media + Graph + DA + CL)

    TMM (IEEE Transactions on Multimedia) 2022, [[PDF]](https://arxiv.org/pdf/2107.05315v1.pdf), [[Code]](https://github.com/zltao/SLMRec/)

14. **SelfCF: A Simple Framework for Self-supervised Collaborative Filtering** (CF + Graph + DA + CL)

    ACM MM (ACM International Conference on Multimedia) 2021, [[PDF]](https://arxiv.org/pdf/2107.03019.pdf), [[Code]](https://github.com/enoche/SelfCF)

15. **Trading Hard Negatives and True Negatives:A Debiased Contrastive Collaborative Filtering Approach** (CF + CL)

    IJCAI 2022, [[PDF]](https://arxiv.org/pdf/2204.11752.pdf)

16. **The World is Binary: Contrastive Learning for Denoising Next Basket Recommendation** (Next Basket + CL)

    SIGIR 2021, [[PDF]](https://dl.acm.org/doi/10.1145/3404835.3462836)

17. **MIC: Model-agnostic Integrated Cross-channel Recommender** (Industry + CL + DA)

     CIKM 2022, [[PDF]](https://arxiv.org/pdf/2110.11570.pdf)

18. **A Contrastive Sharing Model for Multi-Task Recommendation** (Multi Task + CL)

     WWW 2022, [[PDF]](https://dl.acm.org/doi/10.1145/3485447.3512043)

19. **C2-CRS: Coarse-to-Fine Contrastive Learning for Conversational Recommender System** (Conversational Rec + CL)

     WSDM 2022, [[PDF]](https://arxiv.org/pdf/2201.02732.pdf), [[Code]](https://github.com/RUCAIBox/WSDM2022-C2CRS)

20. **Contrastive Cross-domain Recommendation in Matching** (Cross-domain Rec + DA + CL)

     KDD 2022, [[PDF]](https://arxiv.org/pdf/2112.00999.pdf), [[Code]](https://github.com/lqfarmer/CCDR)

21. **Contrastive Cross-Domain Sequential Recommendation** (Cross Domain + Sequential + CL)

     CIKM 2022, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3511808.3557262), [[Code]](https://github.com/cjx96/C2DSR)

22. **Prototypical Contrastive Learning and Adaptive Interest Selection for Candidate Generation in Recommendations** (Short Paper, Industry + CL + DA)

     CIKM 2022, [[PDF]](https://arxiv.org/pdf/2211.12893.pdf), [[Code]](https://github.com/cjx96/C2DSR)

23. **Spatio-Temporal Contrastive Learning Enhanced GNNs for Session-based Recommendation** (GNN + CL)

     TOIS 2022, under review, [[PDF]](https://arxiv.org/pdf/2209.11461v2.pdf)
    
24. **Disentangled Causal Embedding With Contrastive Learning For Recommender System** (Causal + CL)

     arXiv 2023, [[PDF]](https://arxiv.org/pdf/2302.03248.pdf), [[Code]](https://github.com/somestudies/DCCL)

25. **Contrastive Collaborative Filtering for Cold-Start Item Recommendation** (CF + Cold Start +  CL)

     WWW 2023, [[PDF]](https://arxiv.org/pdf/2302.02151.pdf), [[Code]](https://github.com/zzhin/CCFCRec)
    
26. **Cross-domain recommendation via user interest alignment** (Cross Domain Rec + CL)

     WWW 2023, [[PDF]](https://arxiv.org/pdf/2301.11467.pdf), [[Code]](https://github/anonymous/COAST)
    
27. **Multi-Modal Self-Supervised Learning for Recommendation** (Multi Modal Rec + CL)

     WWW 2023, [[PDF]](https://arxiv.org/pdf/2302.10632.pdf), [[Code]](https://github.com/HKUDS/MMSSL)
    
28. **Efficient On-Device Session-Based Recommendation** (Session + DA + CL)

     TOIS 2023, [[PDF]](https://arxiv.org/pdf/2209.13422.pdf), [[Code]](https://github.com/xiaxin1998/EODRec)

29. **On-Device Next-Item Recommendation with Self-Supervised Knowledge Distillation** (Session + DA + CL)

     SIGIR 2022, [[PDF]](https://arxiv.org/pdf/2204.11091.pdf), [[Code]](https://github.com/xiaxin1998/OD-Rec)

30. **Modality Matches Modality: Pretraining Modality-Disentangled Item Representations for Recommendation** (Multi Modal Rec + CL)

     WWW 2022, [[PDF]](https://web.archive.org/web/20220428140054id_/https://dl.acm.org/doi/pdf/10.1145/3485447.3512079), [[Code]](https://github.com/hantengyue/PAMD)

31. **End-to-End Personalized Next Location Recommendation via Contrastive User Preference Modeling** (POI Rec + CL)

    arXiv 2023, [[PDF]](https://arxiv.org/abs/2303.12507)

32. **Bootstrap Latent Representations for Multi-modal Recommendation** (Multi Modal Rec + CL)

     WWW 2023, [[PDF]](https://arxiv.org/abs/2207.05969), [[Code]](https://github.com/enoche/BM3)

33. **Simplifying Content-Based Neural News Recommendation: On User Modeling and Training Objectives** (News Rec + CL)

     SIGIR 2023, [[PDF]](https://arxiv.org/abs/2304.03112), [[Code]](https://github.com/andreeaiana/simplifying_nnr)

34. **Hierarchically Fusing Long and Short-Term User Interests for Click-Through Rate Prediction in Product Search** (CTR + CL)

     CIKM 2022, [[PDF]](https://arxiv.org/abs/2304.02089)

35. **Cross-Domain Recommendation to Cold-Start Users via Variational Information Bottleneck** (Cross Domain + CL)

     ICDE 2022, [[PDF]](https://arxiv.org/abs/2304.02089), [[Code]](https://github.com/cjx96/CDRIB)

36. **DisenCDR: Learning Disentangled Representations for Cross-Domain Recommendation** (Cross Domain + CL)

     SIGIR 2022, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3477495.3531967), [[Code]](https://github.com/cjx96/DisenCDR)

37. **Towards Universal Cross-Domain Recommendation** (Cross domain + CL)

     WSDM 2023, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3539597.3570366), [[Code]](https://github.com/cjx96/UniCDR)

38. **Dual-Ganularity Contrastive Learning for Session-based Recommendation** (Session + CL)

    arXiv 2023, [[PDF]](https://arxiv.org/pdf/2304.08873.pdf)

39. **Discreetly Exploiting Inter-session Information for Session-based Recommendation** (Session Rec + CL)

    arXiv 2023, [[PDF]](https://arxiv.org/pdf/2304.08894.pdf)

40. **PerCoNet: News Recommendation with Explicit Persona and Contrastive Learning** (News Rec + CL)

    arXiv 2023, [[PDF]](https://arxiv.org/pdf/2304.07923.pdf)

41. **Hierarchical and Contrastive Representation Learning for Knowledge-aware Recommendation** (Knowledge Aware + CL)

    ICME 2023, [[PDF]](https://arxiv.org/pdf/2304.07506.pdf)

42. **Attention-guided Multi-step Fusion: A Hierarchical Fusion Network for Multimodal Recommendation** (Multi Modal + CL)

    SIGIR 2023, [[PDF]](https://arxiv.org/pdf/2304.11979.pdf)

43. **PerFedRec++: Enhancing Personalized Federated Recommendation with Self-Supervised Pre-Training** (Fed Rec + CL)

    arXiv 2023, [[PDF]](https://arxiv.org/abs/2305.06622)

44. **UniTRec: A Unified Text-to-Text Transformer and Joint Contrastive Learning Framework for Text-based Recommendation** (Text Based Rec + CL)

    ACL 2023, [[PDF]](https://arxiv.org/pdf/2305.15756.pdf), [[Code]](https://github.com/Veason-silverbullet/UniTRec)

45. **Attention-guided Multi-step Fusion: A Hierarchical Fusion Network for Multimodal Recommendation** (Multi Behavior + CL)

    SIGIR 2023, [[PDF]](https://arxiv.org/pdf/2305.18238v1.pdf), [[Code]](https://github.com/Scofield666/MBSSL)

46. **Learning Similarity among Users for Personalized Session-Based Recommendation from hierarchical structure of User-Session-Item** (Session Rec + CL)

    arXiv 2023, [[PDF]](https://arxiv.org/pdf/2306.03040.pdf)

47. **Securing Visually-Aware Recommender Systems: An Adversarial Image Reconstruction and Detection Framework** (Visually Rec + CL)

    arXiv 2023, [[PDF]](https://arxiv.org/pdf/2306.07992.pdf)

48. **Disentangled Contrastive Learning for Cross-Domain Recommendation** (Cross Domain + CL)

    DASFAA 2023, [[PDF]](https://link.springer.com/chapter/10.1007/978-3-031-30672-3_11)

49. **ContentCTR: Frame-level Live Streaming Click-Through Rate Prediction with Multimodal Transformer** (CTR + CL)

     arXiv 2023, [[PDF]](https://arxiv.org/pdf/2306.14392.pdf)

50. **ContentCTR: Frame-level Live Streaming Click-Through Rate Prediction with Multimodal Transformer** (CVR + CL)

     SIGIR 2023, [[PDF]](https://arxiv.org/pdf/2307.05974.pdf), [[Code]](https://github.com/DongRuiHust/CL4CVR)

51. **Language-Enhanced Session-Based Recommendation with Decoupled Contrastive Learning** (Session Rec + CL)

     KDD 2023, [[PDF]](https://arxiv.org/pdf/2307.10650.pdf), [[Code]](https://github.com/gaozhanfire//KDDCup2023)

52. **Multi-view Hypergraph Contrastive Policy Learning for Conversational Recommendation** (Conversational Rec + CL)

     SIGIR 2023, [[PDF]](https://arxiv.org/pdf/2307.14024.pdf), [[Code]](https://github.com/Snnzhao/MH)

53. **Gaussian Graph with Prototypical Contrastive Learning in E-Commerce Bundle Recommendation** (Bundle Rec + CL)

     arXiv 2023, [[PDF]](https://arxiv.org/abs/2307.13468), [[Code]](https://github.com/Snnzhao/MH)