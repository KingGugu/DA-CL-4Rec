# Awesome Contrastive Learning & Data Augmentation RS Paper & Code

This repository collects the latest research progress of **Contrastive Learning (CL) and Data Augmentation (DA)** in Recommender Systems.
Comments and contributions are welcome.

CF = Collaborative Filtering, SSL = Self-Supervised Learning

- [Survey/Tutorial/Framework](#Survey/Tutorial/Framework) Total Papers: 8
- [Only Data Augmentation](#Only-Data-Augmentation) Total Papers: 53
- [Graph Models with CL](#Graph-Models-with-CL) Total Papers: 149
- [Sequential Models with CL](#Sequential-Models-with-CL) Total Papers: 121
- [Other Tasks with CL](#Other-Tasks-with-CL) Total Papers: 163


## Survey/Tutorial/Framework
1. **Contrastive Self-supervised Learning in Recommender Systems: A Survey** (Survey)
   
   TOIS 2023, [[PDF]](https://arxiv.org/pdf/2303.09902.pdf)

2. **Self-Supervised Learning for Recommender Systems A Survey** (Survey + Framework)
   
   TKDE 2023, [[PDF]](https://ieeexplore.ieee.org/document/10144391), [[Code]](https://github.com/Coder-Yu/SELFRec)

3. **Self-Supervised Learning in Recommendation: Fundamentals and Advances** (Tutorial)
   
   WWW 2022, [[Web]](https://ssl-recsys.github.io/)
   
4. **Tutorial: Self-Supervised Learning for Recommendation: Foundations, Methods and Prospects** (Tutorial)
   
   DASFAA 2023, [[Web]](https://junliang-yu.github.io/publications/)

5. **SSLRec: A Self-Supervised Learning Framework for Recommendation** (Framework)
   
   WSDM 2024, [[PDF]](https://arxiv.org/pdf/2308.05697.pdf), [[Code]](https://github.com/HKUDS/SSLRec)

6. **A Comprehensive Survey on Self-Supervised Learning for Recommendation** (Survey)
   
   arXiv 2024, [[PDF]](https://arxiv.org/pdf/2404.03354.pdf), [[Code]](https://github.com/HKUDS/Awesome-SSLRec-Papers)

7. **Towards Graph Contrastive Learning: A Survey and Beyond** (Survey)
   
   arXiv 2024, [[PDF]](https://arxiv.org/pdf/2405.11868)

8. **Data Augmentation for Sequential Recommendation: A Survey** (Survey)
   
   arXiv 2024, [[PDF]](https://arxiv.org/pdf/2409.13545) 


## Only Data Augmentation

1. **Enhancing Collaborative Filtering with Generative Augmentation** (CF + GAN + DA)
   
    KDD 2019, [[PDF]](https://dl.acm.org/doi/abs/10.1145/3292500.3330873)

2. **Future Data Helps Training: Modeling Future Contexts for Session-based Recommendation** (Session + DA)

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

19. **Multi-Epoch Learning for Deep Click-Through Rate Prediction Models** (CRT + DA)

    arXiv 2023, [[PDF]](https://arxiv.org/pdf/2305.19531.pdf)

20. **Improving Conversational Recommendation Systems via Counterfactual Data Simulation** (Conversational Rec + DA)

    KDD 2023, [[PDF]](https://arxiv.org/pdf/2306.02842.pdf), [[Code]](https://github.com/RUCAIBox/CFCRS)

21. **Disentangled Variational Auto-encoder Enhanced by Counterfactual Data for Debiasing Recommendation** (Debias Rec + DA)

    arXiv 2023, [[PDF]](https://arxiv.org/pdf/2306.15961.pdf)

22. **Domain Disentanglement with Interpolative Data Augmentation for Dual-Target Cross-Domain Recommendation** (Cross-Domain + DA)

    RecSys 2023, [[PDF]](https://arxiv.org/pdf/2307.13910.pdf)

23. **Scaling Session-Based Transformer Recommendations using Optimized Negative Sampling and Loss Functions** (Session + DA)

    RecSys 2023, [[PDF]](https://arxiv.org/pdf/2307.14906.pdf)

24. **Intrinsically Motivated Reinforcement Learning based Recommendation with Counterfactual Data Augmentation** (RL Rec + DA)

     arXiv 2022, [[PDF]](https://arxiv.org/pdf/2209.08228.pdf)

25. **Augmented Negative Sampling for Collaborative Filtering** (CF + DA)

     RecSys 2023, [[PDF]](https://arxiv.org/pdf/2308.05972.pdf), [[Code]](https://github.com/Asa9aoTK/ANS-Recbole)

26. **gSASRec: Reducing Overconfidence in Sequential Recommendation Trained with Negative Sampling** (Sequential + DA)

     RecSys 2023, [[PDF]](https://arxiv.org/pdf/2308.07192.pdf)

27. **Learning from All Sides: Diversified Positive Augmentation via Self-distillation in Recommendation** (DA)

     arXiv 2023, [[PDF]](https://arxiv.org/pdf/2308.07629.pdf)

28. **Counterfactual Graph Augmentation for Consumer Unfairness Mitigation in  Recommender Systems** (Graph + DA)

     CIKM 2023, [[PDF]](https://arxiv.org/abs/2308.12083), [[Code]](https://github.com/jackmedda/RS-BGExplainer/tree/cikm2023)

29. **Bayes-enhanced Multi-view Attention Networks for Robust POI Recommendation** (Graph + DA)

     arXiv 2023, [[PDF]](https://arxiv.org/pdf/2311.00491.pdf)

30. **Diffusion Augmentation for Sequential Recommendation** (Sequential + DA)

     CIKM 2023, [[PDF]](https://arxiv.org/pdf/2309.12858.pdf), [[Code]](https://github.com/liuqidong07/DiffuASR)

31. **Large Language Models as Data Augmenters for Cold-Start Item Recommendation** (DA)

     WWW 2024, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3589335.3651532)

32. **SSDRec: Self-Augmented Sequence Denoising for Sequential Recommendation** (Sequential + DA)

     ICDE 2024, [[PDF]](https://arxiv.org/pdf/2403.04278.pdf), [[Code]](https://github.com/zc-97/SSDRec)

33. **CoRAL: Collaborative Retrieval-Augmented Large Language Models Improve Long-tail Recommendation** (DA)

     arXiv 2024, [[PDF]](https://arxiv.org/pdf/2403.06447.pdf)

34. **ReLLa: Retrieval-enhanced Large Language Models for Lifelong Sequential Behavior Comprehension in Recommendation** (DA)

     WWW 2024, [[PDF]](https://arxiv.org/pdf/2308.11131.pdf), [[Code]](https://github.com/LaVieEnRose365/ReLLa)

35. **Repeated Padding for Sequential Recommendation** (Sequential + DA)

     RecSys 2024, [[PDF]](https://arxiv.org/pdf/2403.06372.pdf), [[Code]](https://github.com/KingGugu/RepPad)

36. **Rethinking sequential relationships: Improving sequential recommenders with inter-sequence data augmentation** (Sequential + DA)

     amazon.science 2024, [[PDF]](https://www.amazon.science/publications/rethinking-sequential-relationships-improving-sequential-recommenders-with-inter-sequence-data-augmentation)

37. **Beyond Relevance: Factor-level Causal Explanation for User Travel Decisions with Counterfactual Data Augmentation** (POI Rec + DA)

     TOIS 2024, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3653673)

38. **TAU: Trajectory Data Augmentation with Uncertainty for Next POI Recommendation** (POI Rec + DA)

     AAAI 2024, [[PDF]](https://ojs.aaai.org/index.php/AAAI/article/download/30265/32257)

39. **Improving Long-Tail Item Recommendation with Graph Augmentation** (Graph + DA)

     CIKM 2023, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3583780.3614929)

40. **Improving Long-Tail Item Recommendation with Graph Augmentation** (Coupon Rec + DA)

     WWW 2024, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3589335.3648306)

41. **Dataset Regeneration for Sequential Recommendation** (Sequential + DA)

     KDD 2024, [[PDF]](https://arxiv.org/pdf/2405.17795), [[Code]](https://anonymous.4open.science/r/KDD2024-86EA/)

42. **Counterfactual Data Augmentation for Debiased Coupon Recommendations Based on Potential Knowledge** (DA)

     WWW 2024, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3589335.3648306)

43. **A Generic Behavior-Aware Data Augmentation Framework for Sequential Recommendation** (Sequential + DA)

     SIGIR 2024, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3626772.3657682), [[Code]](https://github.com/XiaoJing-C/MBASR)

44. **Cross-reconstructed Augmentation for Dual-target Cross-domain Recommendation** (Cross-Domain + DA)

     SIGIR 2024, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3626772.3657902), [[Code]](https://github.com/Double680/CrossAug)

45. **SCM4SR: Structural Causal Model-based Data Augmentation for Robust Session-based Recommendation** (Session Rec + DA)

     SIGIR 2024, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3626772.3657940)

46. **GenRec: A Flexible Data Generator for Recommendations** (DA)

     arXiv 2024, [[PDF]](https://arxiv.org/pdf/2407.16594), [[Code]](https://anonymous.4open.science/r/GenRec-DED3)

47. **Sample Enrichment via Temporary Operations on Subsequences for Sequential Recommendation** (Sequential + DA)

     arXiv 2024, [[PDF]](https://arxiv.org/pdf/2407.17802), [[Code]](https://anonymous.4open.science/r/SETO-code-A026/)

48. **Discovering Collaborative Signals for Next POI Recommendation with Iterative Seq2Graph Augmentation** (POI Rec + DA)

     IJCAI 2021, [[PDF]](https://arxiv.org/abs/2106.15814)

49. **Federated Recommender System Based on Diffusion Augmentation and Guided Denoising** (Fed Rec + DA)

     TOIS 2024, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3688570)

50. **Sliding Window Training - Utilizing Historical Recommender Systems Data for Foundation Models** (Sequential + DA)

     RecSys 2024, [[PDF]](https://arxiv.org/pdf/2409.14517)

51. **PACIFIC: Enhancing Sequential Recommendation via Preference-aware Causal Intervention and Counterfactual Data Augmentation** (Sequential + DA)

     CIKM 2024, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3627673.3679803), [[Code]](https://github.com/ppppeanut/Pacific)

52. **Guided Diffusion-based Counterfactual Augmentation for Robust Session-based Recommendation** (Session + DA)

     RecSys 2024, [[PDF]](https://arxiv.org/pdf/2410.21892)

53. **Privacy-Preserving Synthetic Data Generation for Recommendation Systems** (Privacy-Preserving + DA)

     SIGIR 2022, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3477495.3532044), [[Code]](https://github.com/HuilinChenJN/UPC_SDG)


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

7. **An MLP-based Algorithm for Efficient Contrastive Graph Recommendations** (Graph + CL + DA)

    SIGIR 2022, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3477495.3531874)

8. **A Review-aware Graph Contrastive Learning Framework for Recommendation** (Graph + CL + DA)

    SIGIR 2022, [[PDF]](https://arxiv.org/pdf/2204.12063.pdf), [[Code]](https://github.com/JarenceSJ/ReviewGraph)

9. **Simple Yet Effective Graph Contrastive Learning for Recommendation** (Graph + CL + DA)

    ICLR 2023, [[PDF]](https://arxiv.org/pdf/2302.08191.pdf), [[Code]](https://github.com/HKUDS/LightGCL)

10. **Contrastive Meta Learning with Behavior Multiplicity for Recommendation** (Graph + CL + DA)

    WSDM 2022, [[PDF]](https://arxiv.org/pdf/2202.08523.pdf), [[Code]](https://github.com/weiwei1206/CML)

11. **Disentangled Contrastive Learning for Social Recommendation** (Graph + CL + DA)

    CIKM 2022, [[PDF]](https://arxiv.org/pdf/2208.08723.pdf)

12. **Improving Knowledge-aware Recommendation with Multi-level Interactive Contrastive Learning** (Graph + CL)

    CIKM 2022, [[PDF]](https://arxiv.org/pdf/2208.10061.pdf), [[Code]](https://github.com/CCIIPLab/KGIC)

13. **Multi-level Cross-view Contrastive Learning for Knowledge-aware Recommender System** (Graph + CL)

    SIGIR 2022, [[PDF]](https://arxiv.org/pdf/2204.08807.pdf), [[Code]](https://github.com/CCIIPLab/MCCLK)

14. **Knowledge Graph Contrastive Learning for Recommendation** (Graph + DA + CL)

    SIGIR 2022, [[PDF]](https://arxiv.org/pdf/2205.00976.pdf), [[Code]](https://github.com/yuh-yang/KGCL-SIGIR22)

15. **Self-Supervised Multi-Channel Hypergraph Convolutional Network for Social Recommendation** (Graph + SSL)

    WWW 2021, [[PDF]](https://arxiv.org/pdf/2101.06448.pdf), [[Code]](https://github.com/Coder-Yu/QRec)

16. **SAIL: Self-Augmented Graph Contrastive Learning** (Graph + CL)

    AAAI 2022, [[PDF]](https://arxiv.org/pdf/2009.00934.pdf)

17. **Predictive and Contrastive: Dual-Auxiliary Learning for Recommendation** (Graph + CL)

    arXiv 2022, [[PDF]](https://arxiv.org/pdf/2203.03982.pdf)

18. **Socially-Aware Self-Supervised Tri-Training for Recommendation** (Graph + CL)

    KDD 2021, [[PDF]](https://arxiv.org/pdf/2106.03569.pdf), [[Code]](https://github.com/Coder-Yu/QRec)

19. **Predictive and Contrastive: Dual-Auxiliary Learning for Recommendation** (Graph + CL)

    arXiv 2022, [[PDF]](https://arxiv.org/pdf/2203.03982.pdf)

20. **Multi-Behavior Dynamic Contrastive Learning for Recommendation** (Graph + CL)

    arXiv 2022, [[PDF]](https://arxiv.org/pdf/2203.03982.pdf)

21. **Self-Augmented Recommendation with Hypergraph Contrastive Collaborative Filtering** (Graph + CL)

    SIGIR 2022, [[PDF]](https://arxiv.org/pdf/2204.12200), [[Code]](https://github.com/akaxlh/HCCF)

22. **Improving Graph Collaborative Filtering with Neighborhood-enriched Contrastive Learning** (Graph + CF + CL)

    WWW 2022, [[PDF]](https://arxiv.org/pdf/2202.06200.pdf), [[Code]](https://github.com/RUCAIBox/NCL)

23. **Semi-deterministic and Contrastive Variational Graph Autoencoder for Recommendation** (Graph + CL)

    CIKM 2021, [[PDF]](https://dl.acm.org/doi/10.1145/3459637.3482390), [[Code]](https://github.com/syxkason/SCVG)

24. **Hypergraph Contrastive Collaborative Filtering** (Graph + CF + CL + DA)

    SIGIR 2022, [[PDF]](https://arxiv.org/pdf/2204.12200.pdf), [[Code]]( https://github.com/akaxlh/HCCF)

25. **Graph Structure Aware Contrastive Knowledge Distillation for Incremental Learning in Recommender Systems** (Graph + CL)

    CIKM 2021, [[PDF]](https://dl.acm.org/doi/10.1145/3459637.3482117), [[Code]](https://github.com/syxkason/SCVG)

26. **Double-Scale Self-Supervised Hypergraph Learning for Group Recommendation** (Group Rec, Graph + CL + DA)

    CIKM 2021, [[PDF]](https://arxiv.org/abs/2109.04200), [[Code]](https://github.com/0411tony/HHGR)

27. **Self-Supervised Hypergraph Transformer for Recommender Systems** (Graph + SSL)

    KDD 2022, [[PDF]](https://arxiv.org/pdf/2207.14338.pdf), [[Code]](https://github.com/akaxlh/SHT)

28. **Episodes Discovery Recommendation with Multi-Source Augmentations** (Graph + DA + CL)

     arXiv 2023, [[PDF]](https://arxiv.org/pdf/2301.01737.pdf)

29. **Poincaré Heterogeneous Graph Neural Networks for Sequential Recommendation** (Graph + Sequential + CL)

     TOIS 2023, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3568395)
    
30. **Adversarial Learning Data Augmentation for Graph Contrastive Learning in Recommendation** (Graph + DA + CL)

     DASFAA 2023, [[PDF]](https://arxiv.org/pdf/2302.02317.pdf)
    
31. **SimCGNN: Simple Contrastive Graph Neural Network for Session-based Recommendation** (Graph + CL)

     arXiv 2023, [[PDF]](https://arxiv.org/pdf/2302.03997.pdf)
    
32. **MA-GCL: Model Augmentation Tricks for Graph Contrastive Learning** (Graph + DA + CL)

     AAAI 2023, [[PDF]](https://arxiv.org/pdf/2212.07035.pdf), [[Code]](https://github.com/GXM1141/MA-GCL)
    
33. **Self-Supervised Hypergraph Convolutional Networks for Session-based Recommendation** (Graph + Session + CL)

     AAAI 2021, [[PDF]](https://arxiv.org/pdf/2012.06852.pdf), [[Code]](https://github.com/xiaxin1998/DHCN)
    
34. **Self-Supervised Graph Co-Training for Session-based Recommendation** (Graph + Session + CL)

     CIMK 2021, [[PDF]](https://arxiv.org/pdf/2108.10560.pdf), [[Code]](https://github.com/xiaxin1998/COTREC)

35. **Heterogeneous Graph Contrastive Learning for Recommendation** (Graph + CL)

     WSDM 2023, [[PDF]](https://arxiv.org/pdf/2303.00995.pdf), [[Code]](https://github.com/HKUDS/HGCL)

36. **Automated Self-Supervised Learning for Recommendation** (Graph + DA + CL)

     WWW 2023, [[PDF]](https://arxiv.org/pdf/2303.07797.pdf), [[Code]](https://github.com/HKUDS/AutoCF)

37. **Graph-less Collaborative Filtering** (Graph + CL)

     WWW 2023, [[PDF]](https://arxiv.org/pdf/2303.08537.pdf), [[Code]](https://github.com/HKUDS/SimRec)

38. **Disentangled Contrastive Collaborative Filtering** (Graph + CL)

     SIGIR 2023, [[PDF]](https://arxiv.org/pdf/2305.02759.pdf), [[Code]](https://github.com/HKUDS/DCCF)

39. **Knowledge-refined Denoising Network for Robust Recommendation** (Graph + CL)

     SIGIR 2023, [[PDF]](https://arxiv.org/pdf/2304.14987.pdf), [[Code]](https://github.com/xj-zhu98/KRDN)

40. **Disentangled Graph Contrastive Learning for Review-based Recommendation** (Graph + CL)

     arxiv 2022, [[PDF]](https://arxiv.org/pdf/2209.01524.pdf)

41. **Adaptive Graph Contrastive Learning for Recommendation** (Graph + CL)

     SIGIR 2023, [[PDF]](https://arxiv.org/abs/2305.10837), [[Code]](https://github.com/ZzMeei/AdaptiveGCL)

42. **Knowledge Enhancement for Contrastive Multi-Behavior Recommendation** (Graph + CL)

     WSDM 2023, [[PDF]](https://arxiv.org/pdf/2301.05403.pdf), [[Code]](https://github.com/HKUDS/SSLRec)

43. **Contrastive Meta Learning with Behavior Multiplicity for Recommendation** (Graph + CL)

     WSDM 2022, [[PDF]](https://arxiv.org/pdf/2202.08523.pdf), [[Code]](https://github.com/weiwei1206/CML)

44. **Graph Transformer for Recommendation** (Graph + CL)

     SIGIR 2023, [[PDF]](https://arxiv.org/pdf/2306.02330.pdf), [[Code]](https://github.com/HKUDS/GFormer)

45. **PANE-GNN: Unifying Positive and Negative Edges in Graph Neural Networks for Recommendation** (Graph + CL)

     arxiv 2023, [[PDF]](https://arxiv.org/pdf/2306.04095.pdf)

46. **Knowledge Graph Self-Supervised Rationalization for Recommendation** (Graph + CL)

     KDD 2023, [[PDF]](https://arxiv.org/pdf/2307.02759.pdf), [[Code]](https://github.com/HKUDS/KGRec)

47. **Enhanced Graph Learning for Collaborative Filtering via Mutual Information Maximization** (Graph + CL)

     SIGIR 2021, [[PDF]](https://dl.acm.org/doi/abs/10.1145/3404835.3462928)

48. **Generative-Contrastive Graph Learning for Recommendation** (Graph + CL)

     SIGIR 2023, [[PDF]](https://arxiv.org/pdf/2307.05100.pdf)

49. **AdaMCL: Adaptive Fusion Multi-View Contrastive Learning for Collaborative Filtering** (Graph + CL)

     SIGIR 2023, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3539618.3591632), [[Code]](https://github.com/PasaLab/AdaMCL)

50. **Candidate–aware Graph Contrastive Learning for Recommendation** (Graph + CL)

     SIGIR 2023, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3539618.3591647), [[Code]](https://github.com/WeiHeCnSH/CGCL-Pytorch-master)

51. **Multi-View Graph Convolutional Network for Multimedia Recommendation** (Graph + CL)

     MM 2023, [[PDF]](https://arxiv.org/ftp/arxiv/papers/2308/2308.03588.pdf), [[Code]](https://github.com/demonph10/MGCN)

52. **Uncertainty-aware Consistency Learning for Cold-Start Item Recommendation** (Graph + CL)

     SIGIR 2023, [[PDF]](https://arxiv.org/pdf/2308.03470.pdf)

53. **uCTRL: Unbiased Contrastive Representation Learning via Alignment and Uniformity for Collaborative Filtering** (Graph + CL)

     SIGIR 2023, [[PDF]](https://arxiv.org/pdf/2305.12768.pdf), [[Code]](https://github.com/Jaewoong-Lee/sigir_2023_uCTRL)

54. **Contrastive Box Embedding for Collaborative Reasoning** (Graph + CL)

     SIGIR 2023, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3539618.3591654)

55. **Self-Supervised Dynamic Hypergraph Recommendation based on Hyper-Relational Knowledge Graph** (Graph + CL)

     CIKM 2023, [[PDF]](https://arxiv.org/pdf/2308.07752.pdf)

56. **Contrastive Graph Prompt-tuning for Cross-domain Recommendation** (Graph + CL)

     arXiv 2023, [[PDF]](https://arxiv.org/pdf/2308.10685.pdf)

57. **Dual Intents Graph Modeling for User-centric Group Discovery** (Graph + CL)

     CIKM 2023, [[PDF]](https://arxiv.org/pdf/2308.05013.pdf), [[Code]](https://github.com/WxxShirley/CIKM2023DiRec)

58. **Group Identification via Transitional Hypergraph Convolution with Cross-view Self-supervised Learning** (Graph + CL)

     CIKM 2023, [[PDF]](https://arxiv.org/pdf/2308.08620.pdf), [[Code]](https://github.com/mdyfrank/GTGS)

59. **Multi-Relational Contrastive Learning for Recommendation** (Graph + CL)

     RecSys 2023, [[PDF]](https://arxiv.org/pdf/2309.01103.pdf), [[Code]](https://github.com/HKUDS/RCL)

60. **Multi-behavior Recommendation with SVD Graph Neural Networks** (Graph + CL)

     arXiv 2023, [[PDF]](https://arxiv.org/pdf/2309.06912.pdf)

61. **E-commerce Search via Content Collaborative Graph Neural Network** (Graph + DA + CL)

     KDD 2023, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3580305.3599320), [[Code]](https://github.com/XMUDM/CC-GNN)

62. **Long-tail Augmented Graph Contrastive Learning for Recommendation** (Graph + DA + CL)

     PKDD 2023, [[PDF]](https://arxiv.org/pdf/2309.11177.pdf), [[Code]](https://github.com/im0qianqian/LAGCL)

63. **LMACL: Improving Graph Collaborative Filtering with Learnable Model Augmentation Contrastive Learning** (Graph + CL)

    TKDD 2024, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3657302), [[Code]](https://github.com/LiuHsinx/LMACL)

64. **On the Sweet Spot of Contrastive Views for Knowledge-enhanced Recommendation** (Graph + CL)

     arXiv 2023, [[PDF]](https://arxiv.org/pdf/2309.13384.pdf), [[Code]](https://figshare.com/articles/conference_contribution/SimKGCL/22783382)

65. **Neighborhood-Enhanced Supervised Contrastive Learning for Collaborative Filtering** (Graph + CL)

     TKDE 2023, [[PDF]](https://ieeexplore.ieee.org/document/10255367), [[Code]](https://gitee.com/peijie_hfut/nescl)

66. **Towards Robust Neural Graph Collaborative Filtering via Structure Denoising and Embedding Perturbation** (Graph + CL)

     TOIS 2023, [[PDF]](https://dl.acm.org/doi/full/10.1145/3568396)

67. **TDCGL: Two-Level Debiased Contrastive Graph Learning for Recommendation** (Graph + CL)

     arXiv 2023, [[PDF]](https://browse.arxiv.org/pdf/2310.00569.pdf)

68. **Topology-aware Debiased Self-supervised Graph Learning for Recommendation** (Graph + CL)

     arXiv 2023, [[PDF]](https://arxiv.org/pdf/2310.15858.pdf), [[Code]](https://github.com/malajikuai/TDSGL)

69. **Learning to Denoise Unreliable Interactions for Graph Collaborative Filtering** (Graph + DA + CL)

     SIGIR 2022, [[PDF]](https://dl.acm.org/doi/abs/10.1145/3477495.3531889), [[Code]](https://github.com/ChangxinTian/RGCF)

70. **Contrastive Multi-Level Graph Neural Networks for Session-based Recommendation** (Graph + CL)

     TMM 2023, [[PDF]](https://arxiv.org/pdf/2311.02938.pdf)

71. **An Empirical Study Towards Prompt-Tuning for Graph Contrastive Pre-Training in Recommendations** (Graph + CL)

     NeurIPS 2023, [[PDF]](https://openreview.net/pdf?id=XyAP8ScqLV), [[Code]](https://github.com/Haoran-Young/CPTPP)

72. **An Empirical Study Towards Prompt-Tuning for Graph Contrastive Pre-Training in Recommendations** (Graph + CL)

     ICDM 2021, [[PDF]](https://ieeexplore.ieee.org/document/9678992), [[Code]](https://github.com/Haoran-Young/HMG-CR)

73. **Denoised Self-Augmented Learning for Social Recommendation** (Graph + CL)

     IJCAI 2023, [[PDF]](https://arxiv.org/pdf/2305.12685.pdf), [[Code]](https://github.com/HKUDS/DSL)

74. **Intent-aware Recommendation via Disentangled Graph Contrastive Learning** (Graph + CL)

     IJCAI 2023, [[PDF]](https://www.ijcai.org/proceedings/2023/0260.pdf)

75. **GENET: Unleashing the Power of Side Information for Recommendation via Hypergraph Pre-training** (Graph + CL)

     arXiv 2023, [[PDF]](https://arxiv.org/pdf/2311.13121.pdf)

76. **Graph Pre-training and Prompt Learning for Recommendation** (Graph + CL)

     arXiv 2023, [[PDF]](https://arxiv.org/pdf/2311.16716.pdf)

77. **Robust Basket Recommendation via Noise-tolerated Graph Contrastive Learning** (Graph + CL)

     CIKM 2023, [[PDF]](https://arxiv.org/pdf/2311.16334.pdf), [[Code]](https://github.com/Xinrui17/BNCL)

78. **ID Embedding as Subtle Features of Content and Structure for Multimodal Recommendation** (Graph + Multi-Modal + CL)

     arXiv 2023, [[PDF]](https://arxiv.org/pdf/2311.05956.pdf)

79. **Knowledge Graphs and Pre-trained Language Models enhanced Representation Learning for Conversational Recommender Systems** (Graph + LLM + CL)

     arXiv 2023, [[PDF]](https://arxiv.org/pdf/2312.10967.pdf)

80. **LGMRec: Local and Global Graph Learning for Multimodal Recommendation** (Graph + Multi-Modal + CL)

     AAAI 2024, [[PDF]](https://arxiv.org/pdf/2312.16400.pdf), [[Code]](https://github.com/georgeguo-cn/LGMRec)

81. **RDGCL: Reaction-Diffusion Graph Contrastive Learning for Recommendation** (Graph + CL)

     arXiv 2023, [[PDF]](https://arxiv.org/pdf/2312.16563.pdf)

82. **DiffKG: Knowledge Graph Diffusion Model for Recommendation** (Graph + CL)

     WSDM 2024, [[PDF]](https://arxiv.org/pdf/2312.16890.pdf), [[Code]](https://github.com/HKUDS/DiffKG)

83. **QoS-Aware Graph Contrastive Learning for Web Service Recommendation** (Graph + CL)

     arXiv 2024, [[PDF]](https://arxiv.org/pdf/2401.03162.pdf)

84. **Challenging Low Homophily in Social Recommendation** (Graph + CL)

     WWW 2024, [[PDF]](https://arxiv.org/pdf/2401.14606.pdf)

85. **RecDCL: Dual Contrastive Learning for Recommendation** (Graph + CL)

     WWW 2024, [[PDF]](https://arxiv.org/pdf/2401.15635.pdf), [[Code]](https://github.com/THUDM/RecDCL)

86. **Prerequisite-Enhanced Category-Aware Graph Neural Networks for Course Recommendation** (Graph + CL)

     TKDD 2024, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3643644)

87. **Graph Contrastive Learning With Negative Propagation for Recommendation** (Graph + CL)

     TCSS 2024, [[PDF]](https://ieeexplore.ieee.org/abstract/document/10419035)

88. **General Debiasing for Graph-based Collaborative Filtering via Adversarial Graph Dropout** (Graph + CL)

     WWW 2024, [[PDF]](https://arxiv.org/pdf/2402.13769.pdf), [[Code]](https://github.com/Arthurma71/AdvDrop)

89. **Breaking the Barrier: Utilizing Large Language Models for Industrial Recommendation Systems through an Inferential Knowledge Graph** (Graph + CL)

     arXiv 2024, [[PDF]](https://arxiv.org/pdf/2402.13750.pdf)

90. **FedHCDR: Federated Cross-Domain Recommendation with Hypergraph Signal Decoupling** (Graph + CL)

     SIAM 2024, [[PDF]](https://arxiv.org/pdf/2403.02630.pdf), [[Code]](https://github.com/orion-orion/FedHCDR)

91. **Self-supervised Contrastive Learning for Implicit Collaborative Filtering** (Graph + DA + CL)

     arXiv 2024, [[PDF]](https://arxiv.org/pdf/2403.07265.pdf)

92. **Dual-Channel Multiplex Graph Neural Networks for Recommendation** (Graph + CL)

     arXiv 2024, [[PDF]](https://arxiv.org/pdf/2403.11624.pdf)

93. **Bilateral Unsymmetrical Graph Contrastive Learning for Recommendation** (Graph + CL)

     arXiv 2024, [[PDF]](https://arxiv.org/pdf/2403.15075.pdf)

94. **Knowledge-aware Dual-side Attribute-enhanced Recommendation** (Graph + CL)

     arXiv 2024, [[PDF]](https://arxiv.org/pdf/2403.16037.pdf), [[Code]](https://github.com/TJTP/KDAR)

95. **A Progressively-Passing-then-Disentangling Approach to Recipe Recommendation** (Graph + CL)

     TMM 2024, [[PDF]](https://ieeexplore.ieee.org/abstract/document/10460165/)

96. **Graph Augmentation for Recommendation** (Graph + DA + CL)

     ICDE 2024, [[PDF]](https://arxiv.org/pdf/2403.16656.pdf), [[Code]](https://github.com/HKUDS/GraphAug)

97. **One Backpropagation in Two Tower Recommendation Models** (Graph + CL)

     arXiv 2024, [[PDF]](https://arxiv.org/pdf/2403.18227.pdf)

98. **Improving Content Recommendation: Knowledge Graph-Based Semantic Contrastive Learning for Diversity and Cold-Start Users** (Graph + CL)

     COLING 2024, [[PDF]](https://arxiv.org/pdf/2403.18667.pdf)

99. **Dual Homogeneity Hypergraph Motifs with Cross-view Contrastive Learning for Multiple Social Recommendations** (Graph + Social Rec + CL)

    TKDD 2024, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3653976), [[Code]](https://github.com/chenai2024/DH-HGCNplusplus)

100. **Graph Disentangled Contrastive Learning with Personalized Transfer for Cross-Domain Recommendation** (Graph + CL)

     AAAI 2024, [[PDF]](https://ojs.aaai.org/index.php/AAAI/article/download/28723/29398)

101. **A Directional Diffusion Graph Transformer for Recommendation** (Graph + CL)

     SIGIR 2024, [[PDF]](https://arxiv.org/pdf/2404.03326.pdf)

102. **Heterogeneous Adaptive Preference Learning for Recommendation** (Graph + CL)

     TORS 2024, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3656480), [[Code]](https://github.com/Feifei84/HAPLRec/tree/main/)

103. **Stock Recommendations for Individual Investors: A Temporal Graph Network Approach with Diversification-Enhancing Contrastive Learning** (Graph + CL)

     arXiv 2024, [[PDF]](https://arxiv.org/pdf/2404.07223.pdf), [[Code]](https://anonymous.4open.science/r/IJCAI2024-12F4)

104. **Disentangled Cascaded Graph Convolution Networks for Multi-Behavior Recommendation** (Graph + CL)

     arXiv 2024, [[PDF]](https://arxiv.org/pdf/2404.11519.pdf), [[Code]](https://github.com/JianhuaDongCS/Disen-CGCN)

105. **Enhanced Hierarchical Contrastive Learning for Recommendation** (Graph + CL)

     AAAI 2024, [[PDF]](https://ojs.aaai.org/index.php/AAAI/article/download/28761/29461)

106. **How to Improve Representation Alignment and Uniformity in Graph-based Collaborative Filtering?** (Graph + CL)

     AAAI 2024, [[PDF]](https://zyouyang.github.io/assets/publications/AUPlus.pdf), [[Code]](https://github.com/zyouyang/AUPlus)

107. **PopDCL: Popularity-aware Debiased Contrastive Loss for Collaborative Filtering** (Graph + CL)

     CIKM 2023, [[PDF]](https://www.researchgate.net/profile/Liu-Zhuang/publication/374907265_PopDCL_Popularity-aware_Debiased_Contrastive_Loss_for_Collaborative_Filtering/links/65dda61ce7670d36abe2b0eb/PopDCL-Popularity-aware-Debiased-Contrastive-Loss-for-Collaborative-Filtering.pdf)

108. **Improving Graph Collaborative Filtering with Directional Behavior Enhanced Contrastive Learning** (Graph + CL)

     TKDD 2024, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3663574)

109. **Stochastic Sampling for Contrastive Views and Hard Negative Samples in Graph-based Collaborative Filtering** (Graph + CL)

     arXiv 2024, [[PDF]](https://arxiv.org/pdf/2405.00287)

110. **Learning Social Graph for Inactive User Recommendation** (Graph + CL)

     DASFAA 2024, [[PDF]](https://arxiv.org/pdf/2405.05288), [[Code]](https://github.com/liun-online/LSIR)

111. **Exploring the Individuality and Collectivity of Intents behind Interactions for Graph Collaborative Filtering** (Graph + CL)

     SIGIR 2024, [[PDF]](https://arxiv.org/pdf/2405.09042), [[Code]](https://github.com/BlueGhostYi/BIGCF)

112. **Graph Contrastive Learning with Kernel Dependence Maximization for Social Recommendation** (Graph + CL)

     WWW 2024, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3589334.3645412)

113. **MvStHgL: Multi-view Hypergraph Learning with Spatial-temporal Periodic Interests for Next POI Recommendation** (Graph + POI Rec + CL)

     TOIS 2024, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3664651)

114. **A Vlogger-augmented Graph Neural Network Model for Micro-video Recommendation** (Graph + CL)

     ECML-PKDD 2023, [[PDF]](https://arxiv.org/pdf/2405.18260), [[Code]](https://github.com/laiweijiang/VAGNN)

115. **Knowledge Enhanced Multi-intent Transformer Network for Recommendation** (Graph + CL)

     WWW 2024, [[PDF]](https://arxiv.org/pdf/2405.20565), [[Code]](https://github.com/CCIIPLab/KGTN)

116. **QAGCF: Graph Collaborative Filtering for Q&A Recommendation** (Graph + CL)

     arXiv 2024, [[PDF]](https://arxiv.org/pdf/2406.04828)

117. **Balancing Embedding Spectrum for Recommendation** (Graph + CL)

     arXiv 2024, [[PDF]](https://arxiv.org/pdf/2406.12032), [[Code]](https://github.com/tanatosuu/directspec)

118. **Meta Graph Learning for Long-tail Recommendation** (Graph + CL)

     KDD 2023, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3580305.3599428), [[Code]](https://github.com/weicy15/MGL)

119. **Heterogeneous Hypergraph Embedding for Recommendation Systems** (Graph + CL)

     arXiv 2024, [[PDF]](https://arxiv.org/pdf/2407.03665), [[Code]](https://github.com/viethungvu1998/KHGRec)

120. **Consistency and Discrepancy-Based Contrastive Tripartite Graph Learning for Recommendations** (Graph + CL)

     KDD 2024, [[PDF]](https://arxiv.org/pdf/2407.05126), [[Code]](https://github.com/foodfaust/CDR)

121. **Towards Robust Recommendation via Decision Boundary-aware Graph Contrastive Learning** (Graph + CL)

     KDD 2024, [[PDF]](https://arxiv.org/pdf/2407.10184), [[Code]](https://cl4rec.github.io/RGCL)

122. **Graph Augmentation Empowered Contrastive Learning for Recommendation** (Graph + DA + CL)

     TOIS 2024, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3677377)

123. **L2CL: Embarrassingly Simple Layer-to-Layer Contrastive Learning for Graph Collaborative Filtering** (Graph + CL)

     arXiv 2024, [[PDF]](https://arxiv.org/pdf/2407.14266), [[Code]](https://github.com/downeykking/L2CL)

124. **RevGNN: Negative Sampling Enhanced Contrastive Graph Learning for Academic Reviewer Recommendation** (Graph + CL)

     TOIS 2024, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3679200), [[Code]](https://github.com/THUDM/Reviewer-Rec)

125. **Intent-Guided Heterogeneous Graph Contrastive Learning for Recommendation** (Graph + CL)

     arXiv 2024, [[PDF]](https://arxiv.org/pdf/2407.17234), [[Code]](https://github.com/wangyu0627/IHGCL)

126. **Your Graph Recommender is Provably a Single-view Graph Contrastive Learning** (Graph + CL)

     arXiv 2024, [[PDF]](https://arxiv.org/pdf/2407.17723)

127. **High-Order Fusion Graph Contrastive Learning for Recommendation** (Graph + CL)

     arXiv 2024, [[PDF]](https://arxiv.org/pdf/2407.19692)

128. **Feedback Reciprocal Graph Collaborative Filtering** (Graph + CL)

     CIKM 2024, [[PDF]](https://arxiv.org/pdf/2408.02404)

129. **Symmetric Graph Contrastive Learning against Noisy Views for Recommendation** (Graph + CL)

     arXiv 2024, [[PDF]](https://arxiv.org/pdf/2408.02691), [[Code]](https://github.com/user683/SGCL)

130. **Dual-Channel Latent Factor Analysis Enhanced Graph Contrastive Learning for Recommendation** (Graph + DA + CL)

     arXiv 2024, [[PDF]](https://arxiv.org/pdf/2408.04838)

131. **Meta-optimized Structural and Semantic Contrastive Learning for Graph Collaborative Filtering** (Graph + DA + CL)

     ICDE 2024, [[PDF]](https://ieeexplore.ieee.org/abstract/document/10597955), [[Code]](https://github.com/YongjingHao/Meta-SSCL)

132. **Unveiling Vulnerabilities of Contrastive Recommender Systems to Poisoning Attacks** (Graph + Attack + CL)

     KDD 2024, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3637528.3671795), [[Code]](https://github.com/CoderWZW/ARLib)

133. **Enhancing Graph Contrastive Learning with Reliable and Informative Augmentation for Recommendation** (Graph + CL)

     arXiv 2024, [[PDF]](https://arxiv.org/pdf/2409.05633)

134. **Multi-view Hypergraph-based Contrastive Learning Model for Cold-Start Micro-video Recommendation** (Graph + CL)

     arXiv 2024, [[PDF]](https://arxiv.org/pdf/2409.09638)

135. **TwinCL: A Twin Graph Contrastive Learning Model for Collaborative Filtering** (Graph + CL)

     arXiv 2024, [[PDF]](https://arxiv.org/pdf/2409.19169), [[Code]](https://github.com/chengkai-liu/TwinCL)

136. **Firzen: Firing Strict Cold-Start Items with Frozen Heterogeneous and Homogeneous Graphs for Recommendation** (Graph + CL)

     ICDE 2024, [[PDF]](https://arxiv.org/pdf/2410.07654), [[Code]](https://github.com/PKU-ICST-MIPL/Firzen_ICDE2024)

137. **Firzen: Firing Strict Cold-Start Items with Frozen Heterogeneous and Homogeneous Graphs for Recommendation** (Graph + CL)

     ICWS 2024, [[PDF]](https://arxiv.org/pdf/2410.10296), [[Code]](https://github.com/ItsukiFujii/AttrGAU)

138. **Adaptive Fusion of Multi-View for Graph Contrastive Recommendation** (Graph + DA + CL)

     RecSys 2024, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3640457.3688153), [[Code]](https://github.com/Du-danger/AMGCR)

139. **Simplify to the Limit! Embedding-less Graph Collaborative Filtering for Recommender Systems** (Graph + CL)

     TOIS 2024, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3701230), [[Code]](https://github.com/BlueGhostYi/ID-GRec)

140. **FairDgcl: Fairness-aware Recommendation with Dynamic Graph Contrastive Learning** (Graph + DA + CL)

     arXiv 2024, [[PDF]](https://arxiv.org/pdf/2410.17555), [[Code]](https://github.com/cwei01/FairDgcl)

141. **Decoupled Behavior-based Contrastive Recommendation** (Graph + CL)

     CIKM 2024, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3627673.3679636), [[Code]](https://github.com/Du-danger/DBCR)

142. **Mixed Supervised Graph Contrastive Learning for Recommendation** (Graph + DA + CL)

     arXiv 2024, [[PDF]](https://arxiv.org/pdf/2404.15954)

143. **Bi-Level Graph Structure Learning for Next POI Recommendation** (Graph + POI Rec + CL)

     TKDE 2024, [[PDF]](https://arxiv.org/pdf/2411.01169)

144. **Bi-Level Graph Structure Learning for Next POI Recommendation** (Graph + Multi-Modal + CL)

     arXiv 2024, [[PDF]](https://arxiv.org/pdf/2411.01169)

145. **Mitigating Matthew Effect: Multi-Hypergraph Boosted Multi-Interest Self-Supervised Learning for Conversational Recommendation** (Graph + CL)

     EMNLP 2024, [[PDF]](https://aclanthology.org/2024.emnlp-main.86.pdf), [[Code]](https://github.com/zysensmile/HiCore)

146. **Unifying Graph Convolution and Contrastive Learning in Collaborative Filtering** (Graph + CL)

     KDD 2024, [[PDF]](https://arxiv.org/pdf/2406.13996), [[Code]](https://github.com/wu1hong/SCCF)

147. **DeBaTeR: Denoising Bipartite Temporal Graph for Recommendation** (Graph + CL)

     arXiv 2024, [[PDF]](https://arxiv.org/pdf/2411.09181)

148. **Next Point-of-Interest Recommendation with Adaptive Graph Contrastive Learning** (Graph + CL)

     TKDE 2024, [[PDF]](https://ieeexplore.ieee.org/document/10772008)

149. **Graph-Sequential Alignment and Uniformity: Toward Enhanced Recommendation Systems** (Graph + Sequential + CL)

     arXiv 2024, [[PDF]](https://arxiv.org/pdf/2412.04276), [[Code]](https://github.com/YuweiCao-UIC/GSAU)


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

    RecSys 2023, [[PDF]](https://arxiv.org/pdf/2211.05290.pdf), [[Code]](https://github.com/Tokkiu/ECL)

14. **Explanation Guided Contrastive Learning for Sequential Recommendation** (Sequential + DA + CL)

    CIKM 2022, [[PDF]](https://arxiv.org/pdf/2209.01347.pdf), [[Code]](https://github.com/demoleiwang/EC4SRec)

15. **Intent Contrastive Learning for Sequential Recommendation** (Sequential + DA + CL)

    WWW 2022, [[PDF]](https://arxiv.org/pdf/2202.02519.pdf), [[Code]](https://github.com/salesforce/ICLRec)

16. **Dual Contrastive Network for Sequential Recommendation** (Sequential + CL)

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

32. **Contrastive Cross-Domain Sequential Recommendation** (Cross-Domain + Sequential + CL)

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

46. **Sequential Recommendation with Multiple Contrast Signals** (Sequential + CL)

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

52. **Dual Contrastive Transformer for Hierarchical Preference Modeling in Sequential Recommendation** (Sequential + CL)

    SIGIR 2023, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3539618.3591672)

53. **Leveraging Negative Signals with Self-Attention for Sequential Music Recommendation** (Sequential + CL)

    RecSys 2023, [[PDF]](https://arxiv.org/pdf/2309.11623.pdf)

54. **RUEL: Retrieval-Augmented User Representation with Edge Browser Logs for Sequential Recommendationn** (Sequential + DA + CL)

    CIKM 2023, [[PDF]](https://arxiv.org/pdf/2309.10469.pdf)

55. **FedDCSR: Federated Cross-domain Sequential Recommendation via Disentangled Representation Learning** (Sequential + DA + CL)

    arXiv 2023, [[PDF]](https://arxiv.org/pdf/2309.08420.pdf), [[Code]](https://github.com/orion-orion/FedDCSR)

56. **Unbiased and Robust: External Attention-enhanced Graph Contrastive Learning for Cross-domain Sequential Recommendation** (Sequential + Graph + DA + CL)

    arXiv 2023, [[PDF]](https://arxiv.org/pdf/2310.04633.pdf), [[Code]](https://github.com/HoupingY/EA-GCL)

57. **Dual-Scale Interest Extraction Framework with Self-Supervision for Sequential Recommendation** (Sequential + CL)

    arXiv 2023, [[PDF]](https://arxiv.org/pdf/2310.10025.pdf)

58. **Intent Contrastive Learning with Cross Subsequences for Sequential Recommendation** (Sequential + DA + CL)

    WSDM 2024, [[PDF]](https://arxiv.org/pdf/2310.14318.pdf), [[Code]](https://github.com/QinHsiu/ICSRec)

59. **Meta-optimized Joint Generative and Contrastive Learning for Sequential Recommendation** (Sequential + DA + CL)

    ICDE 2024, [[PDF]](https://arxiv.org/pdf/2310.13925.pdf), [[Code]](https://github.com/YongjingHao/Meta-SGCL)

60. **Model-enhanced Contrastive Reinforcement Learning for Sequential Recommendation** (Sequential + CL)

    arXiv 2023, [[PDF]](https://arxiv.org/pdf/2310.16566.pdf)

61. **Periodicity May Be Emanative: Hierarchical Contrastive Learning for Sequential Recommendation** (Sequential + DA + CL)

    CIKM 2023, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3583780.3615007), [[Code]](https://github.com/RUCAIBox/RecBole)

62. **APGL4SR: A Generic Framework with Adaptive and Personalized Global Collaborative Information in Sequential Recommendation** (Sequential + Graph + CL)

    CIKM 2023, [[PDF]](https://arxiv.org/pdf/2311.02816.pdf), [[Code]](https://github.com/Graph-Team/APGL4SR)

63. **Towards Open-world Cross-Domain Sequential Recommendation: A Model-Agnostic Contrastive Denoising Approach** (Sequential + CL)

    arXiv 2023, [[PDF]](https://arxiv.org/pdf/2311.04760.pdf)

64. **Feature-Level Deeper Self-Attention Network With Contrastive Learning for Sequential Recommendation** (Sequential + CL)

    TKDE 2023, [[PDF]](https://ieeexplore.ieee.org/document/10059216)

65. **Learnable Model Augmentation Contrastive Learning for Sequential Recommendation** (Sequential + CL)

    TKDE 2023, [[PDF]](https://ieeexplore.ieee.org/document/10313990)

66. **Learnable Model Augmentation Contrastive Learning for Sequential Recommendation** (Sequential + DA + CL)

    WSDM 2023, [[PDF]](https://www.atailab.cn/seminar2023Spring/pdf/2023_WSDM_Multi-Intention%20Oriented%20Contrastive%20Learning%20for%20Sequential%20Recommendation.pdf), [[Code]](https://github.com/LFM-bot/IOCRec)

67. **Collaborative Word-based Pre-trained Item Representation for Transferable Recommendation** (Sequential + CL)

    arXiv 2023, [[PDF]](https://arxiv.org/pdf/2311.10501.pdf), [[Code]](https://github.com/ysh-1998/CoWPiRec)

68. **Cracking the Code of Negative Transfer:A Cooperative Game Theoretic Approach for Cross-Domain Sequential Recommendation** (Sequential + Cross-Domain + CL)

    CIKM 2023, [[PDF]](https://arxiv.org/pdf/2311.13188.pdf)

69. **Contrastive Multi-View Interest Learning for Cross-Domain Sequential Recommendation** (Sequential + Cross-Domain + CL)

    TOIS 2023, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3632402), [[Code]](https://github.com/ZSHKJWBY/CMVCDR)

70. **E4SRec: An Elegant Effective Efficient Extensible Solution of Large Language Models for Sequential Recommendation** (Sequential + LLM + CL)

    arXiv 2023, [[PDF]](https://arxiv.org/pdf/2312.02443.pdf), [[Code]](https://github.com/HestiaSky/E4SRec/)

71. **TFCSRec: Time-Frequency Consistency Based Contrastive Learning for Sequential Recommendation** (Sequential + CL)

    Expert Systems with Applications 2024, [[PDF]](https://www.sciencedirect.com/science/article/pii/S0957417423036229)

72. **A Relevant and Diverse Retrieval-enhanced Data Augmentation Framework for Sequential Recommendation** (Sequential + DA + CL)

    CIMK 2022, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3511808.3557071), [[Code]](https://github.com/RUCAIBox/ReDA)

73. **high-level preferences as positive examples in contrastive learning for multi-interest sequential recommendation** （Sequential + CL）

    Arxiv 2023, [[PDF]](https://assets.researchsquare.com/files/rs-3825823/v1_covered_773bc524-1cf2-454b-88cb-52e5bf0386b0.pdf?c=1704709556)

74. **Feature-Aware Contrastive Learning with Bidirectional Transformers for Sequential Recommendation** (Sequential + CL)

    TKDE 2023, [[PDF]](https://ieeexplore.ieee.org/abstract/document/10375742/)

75. **End-to-end Learnable Clustering for Intent Learning in Recommendation** (Sequential + CL)

    arXiv 2024, [[PDF]](https://arxiv.org/pdf/2401.05975.pdf)
    
76. **Contrastive Learning with Frequency-Domain Interest Trends for Sequential Recommendation** (Sequential + DA + CL)

    RecSys 2023, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3604915.3608790), [[Code]](https://github.com/zhangyichi1Z/CFIT4SRec)

77. **Sequential Recommendation on Temporal Proximities with Contrastive Learning and Self-Attention** (Sequential + CL)

    WWW 2024, [[PDF]](https://arxiv.org/pdf/2402.09784.pdf), [[Code]](https://github.com/TemProxRec)

78. **End-to-end Graph-Sequential Representation Learning for Accurate Recommendations** (Sequential + Graph + CL)

    WWW 2024, [[PDF]](https://arxiv.org/pdf/2403.00895.pdf), [[Code]](https://github.com/NonameUntitled/MRGSRec)

79. **Multi-Sequence Attentive User Representation Learning for Side-information Integrated Sequential Recommendation** (Sequential + CL)

    WSDM 2024, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3616855.3635815), [[Code]](https://github.com/xiaolLIN/MSSR)

80. **Empowering Sequential Recommendation from Collaborative Signals and Semantic Relatedness** (Sequential + CL)

    arxiv 2024, [[PDF]](https://arxiv.org/pdf/2403.07623.pdf)

81. **Collaborative Sequential Recommendations via Multi-View GNN-Transformers** (Sequential + Graph + CL)

    TOIS 2024, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3649436)

82. **Is Contrastive Learning Necessary? A Study of Data Augmentation vs Contrastive Learning in Sequential Recommendation** (Sequential + DA + CL)

    WWW 2024, [[PDF]](https://arxiv.org/pdf/2403.11136.pdf), [[Code]](https://github.com/AIM-SE/DA4Rec)

83. **Diversifying Sequential Recommendation with Retrospective and Prospective Transformers** (Sequential + CL)

    TOIS 2024, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3653016), [[Code]](https://github.com/chaoyushi/TRIER)

84. **A Large Language Model Enhanced Sequential Recommender for Joint Video and Comment Recommendation** (Sequential + CL)

    arXiv 2024, [[PDF]](https://arxiv.org/pdf/2403.13574.pdf), [[Code]](https://github.com/RUCAIBox/LSVCR/)

85. **Efficient Noise-Decoupling for Multi-Behavior Sequential Recommendation** (Sequential + CL)

    WWW 2024, [[PDF]](https://arxiv.org/pdf/2403.17603.pdf), [[Code]](https://github.com/huschbsd/END4REC)

86. **Temporal Graph Contrastive Learning for Sequential Recommendation** (Sequential + Graph + CL)

    AAAI 2024, [[PDF]](https://ojs.aaai.org/index.php/AAAI/article/download/28789/29511)

87. **Sparse Enhanced Network: An Adversarial Generation Method for Robust Augmentation in Sequential Recommendation** (Sequential + DA + CL)

    AAAI 2024, [[PDF]](https://ojs.aaai.org/index.php/AAAI/article/download/28669/29299), [[Code]](https://github.com/junyachen/SparseEnNet)

88. **Sequential Recommendation for Optimizing Both Immediate Feedback and Long-term Retention** (Sequential + CL)

    SIGIR 2024, [[PDF]](https://arxiv.org/pdf/2404.03637.pdf), [[Code]](https://anonymous.4open.science/r/DT4IER-5837)

89. **Beyond the Sequence: Statistics-Driven Pre-training for Stabilizing Sequential Recommendation Model** (Sequential + DA + CL)

    RecSys 2023, [[PDF]](https://arxiv.org/pdf/2404.05342.pdf)

90. **Leave No One Behind: Online Self-Supervised Self-Distillation for Sequential Recommendation** (Sequential + DA + CL)

    WWW 2024, [[PDF]](https://arxiv.org/pdf/2404.07219.pdf), [[Code]](https://github.com/xjaw/S4Rec)

91. **UniSAR: Modeling User Transition Behaviors between Search and Recommendation** (Sequential + CL)

    SIGIR 2024, [[PDF]](https://arxiv.org/pdf/2404.09520.pdf), [[Code]](https://github.com/TengShi-RUC/UniSAR)

92. **Multi-Level Sequence Denoising with Cross-Signal Contrastive Learning for Sequential Recommendation** (Sequential + CL)

    arXiv 2024, [[PDF]](https://arxiv.org/pdf/2404.13878.pdf), [[Code]](https://github.com/lalunex/MSDCCL/tree/main)

93. **Contrastive Learning Method for Sequential Recommendation based on Multi-Intention Disentanglement** (Sequential + CL)

    arXiv 2024, [[PDF]](https://arxiv.org/pdf/2404.18214)

94. **CALRec: Contrastive Alignment of Generative LLMs For Sequential Recommendation** (Sequential + LLM + CL)

    arXiv 2024, [[PDF]](https://arxiv.org/pdf/2405.02429)

95. **ID-centric Pre-training for Recommendation** (Sequential + CL)

    arXiv 2024, [[PDF]](https://arxiv.org/pdf/2405.03562)

96. **Diffusion-based Contrastive Learning for Sequential Recommendation** (Sequential + DA + CL)

    arXiv 2024, [[PDF]](https://arxiv.org/pdf/2405.09369)

97. **Soft Contrastive Sequential Recommendation** (Sequential + DA + CL)

    TOIS 2024, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3665325)

98. **Modeling User Fatigue for Sequential Recommendation** (Sequential + DA + CL)

    SIGIR 2024, [[PDF]](https://arxiv.org/pdf/2405.11764), [[Code]](https://github.com/tsinghua-fib-lab/SIGIR24-FRec)

99. **Aligned Side Information Fusion Method for Sequential Recommendation** (Sequential + CL)

    WWW 2024, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3589335.3648308)

100. **Learning Partially Aligned Item Representation for Cross-Domain Sequential Recommendation** (Sequential + CL)

     arXiv 2024, [[PDF]](https://arxiv.org/pdf/2405.12473), [[Code]](https://anonymous.4open.science/r/KDD2024-58E8/)

101. **SelfGNN: Self-Supervised Graph Neural Networks for Sequential Recommendation** (Sequential + Graph + CL)

     SIGIR 2024, [[PDF]](https://arxiv.org/pdf/2405.20878), [[Code]](https://github.com/HKUDS/SelfGNN)

102. **Exploring User Retrieval Integration towards Large Language Models for Cross-Domain Sequential Recommendation** (Sequential + CL)

     arXiv 2024, [[PDF]](https://arxiv.org/pdf/2406.03085), [[Code]](https://github.com/TingJShen/URLLM)

103. **PTF-FSR: A Parameter Transmission-Free Federated Sequential Recommender System** (Sequential + CL)

     arXiv 2024, [[PDF]](https://arxiv.org/pdf/2406.05387)

104. **Pacer and Runner: Cooperative Learning Framework between Single- and Cross-Domain Sequential Recommendation** (Sequential + CL)

     SIGIR 2024, [[PDF]](https://arxiv.org/pdf/2407.11245), [[Code]](https://github.com/cpark88/SyNCRec)

105. **Scaling Sequential Recommendation Models with Transformers** (Sequential + CL)

     SIGIR 2024, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3626772.3657816), [[Code]](https://github.com/mercadolibre/srt)

106. **CMCLRec: Cross-modal Contrastive Learning for User Cold-start Sequential Recommendation** (Sequential + CL)

     SIGIR 2024, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3626772.3657839)

107. **Multimodal Pre-training for Sequential Recommendation via Contrastive Learning** (Sequential + CL)

     TORS 2024, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3682075)

108. **Beyond Inter-Item Relations: Dynamic Adaptive Mixture-of-Experts for LLM-Based Sequential Recommendation** (Sequential + LLM + CL)

     arXiv 2024, [[PDF]](https://arxiv.org/pdf/2408.07427)

109. **Contrastive Learning on Medical Intents for Sequential Prescription Recommendation** (Sequential + CL)

     CIKM 2024, [[PDF]](https://arxiv.org/pdf/2408.10259), [[Code]](https://github.com/aryahm1375/ARCI)

110. **Disentangled Multi-interest Representation Learning for Sequential Recommendation** (Sequential + CL)

     KDD 2024, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3637528.3671800)

111. **Multi-intent Aware Contrastive Learning for Sequential Recommendation** (Sequential + CL)

     arXiv 2024, [[PDF]](https://arxiv.org/pdf/2409.08733)

112. **Large Language Model Empowered Embedding Generator for Sequential Recommendation** (Sequential + LLM + CL)

     arXiv 2024, [[PDF]](https://arxiv.org/pdf/2409.08733), [[Code]](https://github.com/liuqidong07/LLMEmb)

113. **FELLAS: Enhancing Federated Sequential Recommendation with LLM as External Services** (Sequential + LLM + CL)

     arXiv 2024, [[PDF]](https://arxiv.org/pdf/2410.04927)

114. **Sequential Recommendation with Collaborative Explanation via Mutual Information Maximization** (Sequential + CL)

     SIGIR 2024, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3626772.3657770), [[Code]](https://github.com/yiyualt/SCEMIM)

115. **Intent-Enhanced Data Augmentation for Sequential Recommendation** (Sequential + DA + CL)

     arXiv 2024, [[PDF]](https://arxiv.org/pdf/2410.08583), [[Code]](https://github.com/yiyualt/SCEMIM)

116. **Relative Contrastive Learning for Sequential Recommendation with Similarity-based Positive Sample Selection** (Sequential + CL)

     CIKM 2024, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3627673.3679681), [[Code]](https://github.com/Cloudcatcher888/RCL)

117. **Context Matters: Enhancing Sequential Recommendation with Context-aware Diffusion-based Contrastive Learning** (Sequential + DA + CL)

     CIKM 2024, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3627673.3679655), [[Code]](https://github.com/ziqiangcui/CaDiRec)

118. **Momentum Contrastive Bidirectional Encoding with Self-Distillation for Sequential Recommendation** (Sequential + CL)

     CIKM 2024, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3627673.3679965)

119. **AuriSRec: Adversarial User Intention Learning in Sequential Recommendation** (Sequential + CL)

     EMNLP 2024 (Findings), [[PDF]](https://aclanthology.org/2024.findings-emnlp.735.pdf)

120. **AuriSRec: Adversarial User Intention Learning in Sequential Recommendation** (Sequential + LLM + CL)

     EMNLP 2024 (Findings), [[PDF]](https://aclanthology.org/2024.findings-emnlp.423.pdf)

121. **LLM-assisted Explicit and Implicit Multi-interest Learning Framework for Sequential Recommendation** (Sequential + LLM + CL)

     arXiv 2024, [[PDF]](https://arxiv.org/pdf/2411.09410)


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

6. **Improving Micro-video Recommendation via Contrastive Multiple Interests** (Micro Video + CL)

    SIGIR 2022, [[PDF]](https://arxiv.org/pdf/2205.09593.pdf)

7. **Exploiting Negative Preference in Content-based Music Recommendation with Contrastive Learning** (Music Rec + CL)

    RecSys 2022, [[PDF]](https://arxiv.org/pdf/2103.09410.pdf), [[Code]](https://github.com/Spijkervet/CLMR)

8. **Self-supervised Learning for Large-scale Item Recommendations** (Industry + CL + DA)

    CIKM 2021, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3459637.3481952)

9. **CrossCBR: Cross-view Contrastive Learning for Bundle Recommendation** (Bundle Rec + CL)

    KDD 2023, [[PDF]](https://arxiv.org/pdf/2206.00242.pdf), [[Code]](https://github.com/mysbupt/CrossCBR)

10. **Contrastive Learning for Cold-start Recommendation** (Cold Start + CL)

    ACM MM (ACM International Conference on Multimedia) 2021, [[PDF]](https://arxiv.org/pdf/2107.05315v1.pdf), [[Code]](https://github.com/weiyinwei/CLCRec)

11. **Socially-aware Dual Contrastive Learning for Cold-Start Recommendation** (Cold Start + CL)

    SIGIR 2022, [[PDF]](https://dl.acm.org/doi/10.1145/3477495.3531780)

12. **Multi-modal Graph Contrastive Learning for Micro-video Recommendation** (Cold Start + Graph + CL)

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

21. **Contrastive Cross-Domain Sequential Recommendation** (Cross-Domain + Sequential + CL)

     CIKM 2022, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3511808.3557262), [[Code]](https://github.com/cjx96/C2DSR)

22. **Prototypical Contrastive Learning and Adaptive Interest Selection for Candidate Generation in Recommendations** (Industry + CL + DA)

     CIKM 2022, [[PDF]](https://arxiv.org/pdf/2211.12893.pdf), [[Code]](https://github.com/cjx96/C2DSR)

23. **Spatio-Temporal Contrastive Learning Enhanced GNNs for Session-based Recommendation** (GNN + CL)

     TOIS 2022, under review, [[PDF]](https://arxiv.org/pdf/2209.11461v2.pdf)
    
24. **Disentangled Causal Embedding With Contrastive Learning For Recommender System** (Causal + CL)

     arXiv 2023, [[PDF]](https://arxiv.org/pdf/2302.03248.pdf), [[Code]](https://github.com/somestudies/DCCL)

25. **Contrastive Collaborative Filtering for Cold-Start Item Recommendation** (CF + Cold Start +  CL)

     WWW 2023, [[PDF]](https://arxiv.org/pdf/2302.02151.pdf), [[Code]](https://github.com/zzhin/CCFCRec)
    
26. **Cross-domain recommendation via user interest alignment** (Cross-Domain Rec + CL)

     WWW 2023, [[PDF]](https://arxiv.org/pdf/2301.11467.pdf), [[Code]](https://github/anonymous/COAST)
    
27. **Multi-Modal Self-Supervised Learning for Recommendation** (Multi-Modal Rec + CL)

     WWW 2023, [[PDF]](https://arxiv.org/pdf/2302.10632.pdf), [[Code]](https://github.com/HKUDS/MMSSL)
    
28. **Efficient On-Device Session-Based Recommendation** (Session + DA + CL)

     TOIS 2023, [[PDF]](https://arxiv.org/pdf/2209.13422.pdf), [[Code]](https://github.com/xiaxin1998/EODRec)

29. **On-Device Next-Item Recommendation with Self-Supervised Knowledge Distillation** (Session + DA + CL)

     SIGIR 2022, [[PDF]](https://arxiv.org/pdf/2204.11091.pdf), [[Code]](https://github.com/xiaxin1998/OD-Rec)

30. **Modality Matches Modality: Pretraining Modality-Disentangled Item Representations for Recommendation** (Multi-Modal Rec + CL)

     WWW 2022, [[PDF]](https://web.archive.org/web/20220428140054id_/https://dl.acm.org/doi/pdf/10.1145/3485447.3512079), [[Code]](https://github.com/hantengyue/PAMD)

31. **End-to-End Personalized Next Location Recommendation via Contrastive User Preference Modeling** (POI Rec + CL)

     arXiv 2023, [[PDF]](https://arxiv.org/abs/2303.12507)

32. **Bootstrap Latent Representations for Multi-modal Recommendation** (Multi-Modal Rec + CL)

     WWW 2023, [[PDF]](https://arxiv.org/abs/2207.05969), [[Code]](https://github.com/enoche/BM3)

33. **Simplifying Content-Based Neural News Recommendation: On User Modeling and Training Objectives** (News Rec + CL)

     SIGIR 2023, [[PDF]](https://arxiv.org/abs/2304.03112), [[Code]](https://github.com/andreeaiana/simplifying_nnr)

34. **Hierarchically Fusing Long and Short-Term User Interests for Click-Through Rate Prediction in Product Search** (CTR + CL)

     CIKM 2022, [[PDF]](https://arxiv.org/abs/2304.02089)

35. **Cross-Domain Recommendation to Cold-Start Users via Variational Information Bottleneck** (Cross-Domain + CL)

     ICDE 2022, [[PDF]](https://arxiv.org/abs/2304.02089), [[Code]](https://github.com/cjx96/CDRIB)

36. **DisenCDR: Learning Disentangled Representations for Cross-Domain Recommendation** (Cross-Domain + CL)

     SIGIR 2022, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3477495.3531967), [[Code]](https://github.com/cjx96/DisenCDR)

37. **Towards Universal Cross-Domain Recommendation** (Cross-domain + CL)

     WSDM 2023, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3539597.3570366), [[Code]](https://github.com/cjx96/UniCDR)

38. **Dual-Ganularity Contrastive Learning for Session-based Recommendation** (Session + CL)

    arXiv 2023, [[PDF]](https://arxiv.org/pdf/2304.08873.pdf)

39. **Discreetly Exploiting Inter-session Information for Session-based Recommendation** (Session Rec + CL)

    arXiv 2023, [[PDF]](https://arxiv.org/pdf/2304.08894.pdf)

40. **PerCoNet: News Recommendation with Explicit Persona and Contrastive Learning** (News Rec + CL)

    arXiv 2023, [[PDF]](https://arxiv.org/pdf/2304.07923.pdf)

41. **Hierarchical and Contrastive Representation Learning for Knowledge-aware Recommendation** (Knowledge Aware + CL)

    ICME 2023, [[PDF]](https://arxiv.org/pdf/2304.07506.pdf)

42. **Attention-guided Multi-step Fusion: A Hierarchical Fusion Network for Multimodal Recommendation** (Multi-Modal + CL)

    SIGIR 2023, [[PDF]](https://arxiv.org/pdf/2304.11979.pdf)

43. **PerFedRec++: Enhancing Personalized Federated Recommendation with Self-Supervised Pre-Training** (Fed Rec + CL)

    arXiv 2023, [[PDF]](https://arxiv.org/abs/2305.06622)

44. **UniTRec: A Unified Text-to-Text Transformer and Joint Contrastive Learning Framework for Text-based Recommendation** (Text Based Rec + CL)

    ACL 2023, [[PDF]](https://arxiv.org/pdf/2305.15756.pdf), [[Code]](https://github.com/Veason-silverbullet/UniTRec)

45. **Attention-guided Multi-step Fusion: A Hierarchical Fusion Network for Multimodal Recommendation** (Multi-Behavior + CL)

    SIGIR 2023, [[PDF]](https://arxiv.org/pdf/2305.18238v1.pdf), [[Code]](https://github.com/Scofield666/MBSSL)

46. **Learning Similarity among Users for Personalized Session-Based Recommendation from hierarchical structure of User-Session-Item** (Session Rec + CL)

    arXiv 2023, [[PDF]](https://arxiv.org/pdf/2306.03040.pdf)

47. **Securing Visually-Aware Recommender Systems: An Adversarial Image Reconstruction and Detection Framework** (Visually Rec + CL)

    arXiv 2023, [[PDF]](https://arxiv.org/pdf/2306.07992.pdf)

48. **Disentangled Contrastive Learning for Cross-Domain Recommendation** (Cross-Domain + CL)

    DASFAA 2023, [[PDF]](https://link.springer.com/chapter/10.1007/978-3-031-30672-3_11)

49. **ContentCTR: Frame-level Live Streaming Click-Through Rate Prediction with Multimodal Transformer** (CTR + CL)

     arXiv 2023, [[PDF]](https://arxiv.org/pdf/2306.14392.pdf)

50. **Contrastive Learning for Conversion Rate Prediction** (CVR + CL)

     SIGIR 2023, [[PDF]](https://arxiv.org/pdf/2307.05974.pdf), [[Code]](https://github.com/DongRuiHust/CL4CVR)

51. **Language-Enhanced Session-Based Recommendation with Decoupled Contrastive Learning** (Session Rec + CL)

     KDD 2023, [[PDF]](https://arxiv.org/pdf/2307.10650.pdf), [[Code]](https://github.com/gaozhanfire//KDDCup2023)

52. **Multi-view Hypergraph Contrastive Policy Learning for Conversational Recommendation** (Conversational Rec + CL)

     SIGIR 2023, [[PDF]](https://arxiv.org/pdf/2307.14024.pdf), [[Code]](https://github.com/Snnzhao/MH)

53. **Gaussian Graph with Prototypical Contrastive Learning in E-Commerce Bundle Recommendation** (Bundle Rec + CL)

     arXiv 2023, [[PDF]](https://arxiv.org/abs/2307.13468), [[Code]](https://github.com/Snnzhao/MH)

54. **Contrastive Learning for Conversion Rate Prediction** (CVR + CL)

     SIGIR 2023, [[PDF]](https://arxiv.org/pdf/2307.05974.pdf), [[Code]](https://github.com/DongRuiHust/CL4CVR)

55. **Review-based Multi-intention Contrastive Learning for Recommendation** (Review + CL)

     SIGIR 2023, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3539618.3592053)

56. **CSPM: A Contrastive Spatiotemporal Preference Model for CTR Prediction  in On-Demand Food Delivery Services** (CTR + CL)

     CIKM 2023, [[PDF]](https://arxiv.org/pdf/2308.08446.pdf)

57. **MISSRec: Pre-training and Transferring Multi-modal Interest-aware Sequence Representation for Recommendation** (Multi-Modal + CL)

     MM 2023, [[PDF]](https://arxiv.org/pdf/2308.11175.pdf), [[Code]](https://github.com/gimpong/MM23-MISSRec)

58. **MUSE: Music Recommender System with Shuffle Play Recommendation Enhancement** (Music Rec + DA + CL)

     CIKM 2023, [[PDF]](https://arxiv.org/pdf/2308.09649.pdf), [[Code]](https://github.com/yunhak0/MUSE)

59. **Multi-aspect Graph Contrastive Learning for Review-enhanced Recommendation** (Review + CL)

     TOIS 2023, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3618106)

60. **Interpretable User Retention Modeling in Recommendation** (User Modelling + CL)

     RecSys 2023, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3604915.3608818), [[Code]](https://github.com/dinry/IURO)

61. **Beyond Co-occurrence: Multi-modal Session-based Recommendation** (Session Rec + CL)

     TKDE 2023, [[PDF]](https://arxiv.org/pdf/2309.17037.pdf), [[Code]](https://github.com/Zhang-xiaokun/MMSBR)

62. **Representation Learning with Large Language Models for Recommendation** (LLM + CL)

     arXiv 2023, [[PDF]](https://arxiv.org/pdf/2310.15950.pdf), [[Code]](https://github.com/HKUDS/RLMRec)

63. **Universal Multi-modal Multi-domain Pre-trained Recommendation** (Pre-trained + CL)

     arXiv 2023, [[PDF]](https://arxiv.org/pdf/2311.01831.pdf)

64. **Towards Hierarchical Intent Disentanglement for Bundle Recommendation** (Bundle Rec + CL)

     TKDE 2023, [[PDF]](https://ieeexplore.ieee.org/abstract/document/10304376)

65. **ControlRec: Bridging the Semantic Gap between Language Model and Personalized Recommendation** (LLM + CL)

     arXiv 2023, [[PDF]](https://arxiv.org/pdf/2311.16441.pdf)

66. **Enhancing Item-level Bundle Representation for Bundle Recommendation** (Bundle Rec + CL)

     arXiv 2023, [[PDF]](https://arxiv.org/pdf/2311.16892.pdf), [[Code]](https://github.com/answermycode/EBRec)

67. **MultiCBR: Multi-view Contrastive Learning for Bundle Recommendation** (Bundle Rec + CL)

     arXiv 2023, [[PDF]](https://arxiv.org/pdf/2311.16751.pdf), [[Code]](https://github.com/HappyPointer/MultiCBR)

68. **Poisoning Attacks Against Contrastive Recommender Systems** (Attack Rec + CL)

     arXiv 2023, [[PDF]](https://arxiv.org/pdf/2311.18244.pdf)

69. **PEACE: Prototype lEarning Augmented transferable framework for Cross-domain rEcommendation** (Cross-domain + CL)

     WSDM 2024, [[PDF]](https://arxiv.org/pdf/2312.01916.pdf)

70. **(Debiased) Contrastive Learning Loss for Recommendation (Technical Report)** (Analysis + CL)

     arXiv 2023, [[PDF]](https://arxiv.org/pdf/2312.08517.pdf)

71. **Revisiting Recommendation Loss Functions through Contrastive Learning (Technical Report)** (Analysis + CL)

     arXiv 2023, [[PDF]](https://arxiv.org/pdf/2312.08520.pdf)

72. **Hierarchical Alignment With Polar Contrastive Learning for Next-Basket Recommendation** (Next Basket + CL)

     TKDE 2023, [[PDF]](https://ieeexplore.ieee.org/document/10144403)

73. **CETN: Contrast-enhanced Through Network for CTR Prediction** (CTR + CL)

     arXiv 2023, [[PDF]](https://arxiv.org/pdf/2312.09715.pdf)

74. **Multi-Modality is All You Need for Transferable Recommender Systems** (Transferable Rec + CL)

     ICDE 2024, [[PDF]](https://arxiv.org/pdf/2312.09602.pdf), [[Code]](https://github.com/ICDE24/PMMRec)

75. **RIGHT: Retrieval-augmented Generation for Mainstream Hashtag Recommendation** (Hashtag Rec + CL)

     ECIR 2024, [[PDF]](https://arxiv.org/pdf/2312.10466.pdf), [[Code]](https://github.com/ict-bigdatalab/RIGHT)

76. **AT4CTR: Auxiliary Match Tasks for Enhancing Click-Through Rate Prediction** (CTR + CL)

     AAAI 2024, [[PDF]](https://arxiv.org/pdf/2312.06683.pdf)

77. **Attribute-driven Disentangled Representation Learning for Multimodal Recommendation** (Multi-Modal + CL)

     arXiv 2023, [[PDF]](https://arxiv.org/pdf/2312.14433.pdf)

78. **TopicVAE: Topic-aware Disentanglement Representation Learning for Enhanced Recommendation** (Multi-Modal + CL)

     MM 2022, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3503161.3548294), [[Code]](https://github.com/georgeguo-cn/TopicVAE)

79. **Disentangled CVAEs with Contrastive Learning for Explainable Recommendation** (Explainable + CL)

     AAAI 2023, [[PDF]](https://ojs.aaai.org/index.php/AAAI/article/view/26604/26376)

80. **DualVAE: Dual Disentangled Variational AutoEncoder for Recommendation** (Rec + CL)

     SIAM 2024, [[PDF]](https://arxiv.org/pdf/2401.04914.pdf), [[Code]](https://github.com/georgeguo-cn/DualVAE)

81. **Self-Supervised Learning for User Sequence Modeling** (Rec + CL)

     arXiv 2023, [[PDF]](https://sslneurips23.github.io/paper_pdfs/paper_39.pdf)

82. **RA-Rec: An Efficient ID Representation Alignment Framework for LLM-based Recommendation** (LLM + CL)

     arXiv 2024, [[PDF]](https://arxiv.org/pdf/2402.04527.pdf)

83. **CounterCLR: Counterfactual Contrastive Learning with Non-random Missing Data in Recommendation** (Counterfactual + CL)

     arXiv 2024, [[PDF]](https://arxiv.org/pdf/2402.05740.pdf)

84. **Non-autoregressive Generative Models for Reranking Recommendation** (Reranking + CL)

     arXiv 2024, [[PDF]](https://arxiv.org/pdf/2402.06871.pdf)

85. **Modeling Balanced Explicit and Implicit Relations with Contrastive Learning for Knowledge Concept Recommendation in MOOCs** (MOOC Rec + CL)

     arXiv 2024, [[PDF]](https://arxiv.org/pdf/2402.08256.pdf)

86. **MENTOR: Multi-level Self-supervised Learning for Multimodal Recommendation** (Multi-Modal + CL)

     arXiv 2024, [[PDF]](https://arxiv.org/pdf/2402.19407.pdf), [[Code]](https://github.com/Jinfeng-Xu/MENTOR)

87. **NoteLLM: A Retrievable Large Language Model for Note Recommendation** (Note Rec + CL)

     WWW 2024, [[PDF]](https://arxiv.org/pdf/2403.01744.pdf)

88. **A Privacy-Preserving Framework with Multi-Modal Data for Cross-Domain Recommendation** (Cross-Domain + CL)

     arXiv 2024, [[PDF]](https://arxiv.org/pdf/2403.03600.pdf)

89. **PPM : A Pre-trained Plug-in Model for Click-through Rate Prediction** (CTR + CL)

     WWW 2024, [[PDF]](https://arxiv.org/pdf/2403.10049.pdf)

90. **An Aligning and Training Framework for Multimodal Recommendations** (Multi-Modal + CL)

     arXiv 2024, [[PDF]](https://arxiv.org/pdf/2403.12384.pdf)

91. **Reinforcement Learning-based Recommender Systems with Large Language Models for State Reward and Action Modeling** (RL Rec + CL)

     arXiv 2024, [[PDF]](https://arxiv.org/pdf/2403.16948.pdf)

92. **Enhanced Generative Recommendation via Content and Collaboration Integration** (Generative Rec + CL)

     arXiv 2024, [[PDF]](https://arxiv.org/pdf/2403.18480.pdf)

93. **End-to-End Personalized Next Location Recommendation via Contrastive User Preference Modeling** (POI Rec + CL)

     arXiv 2023, [[PDF]](https://arxiv.org/pdf/2303.12507.pdf)

94. **Preference Aware Dual Contrastive Learning for Item Cold-Start Recommendation** (Cold Start + CL)

     AAAI 2024, [[PDF]](https://ojs.aaai.org/index.php/AAAI/article/view/28763/29465)

95. **Tail-STEAK: Improve Friend Recommendation for Tail Users via Self-Training Enhanced Knowledge Distillation** (Friend Rec + CL)

     AAAI 2024, [[PDF]](https://ojs.aaai.org/index.php/AAAI/article/view/28737/29421), [[Code]](https://github.com/antman9914/Tail-STEAK)

96. **Aiming at the Target: Filter Collaborative Information for Cross-Domain Recommendation** (Cross-Domain + CL)

     SIGIR 2024, [[PDF]](https://arxiv.org/pdf/2403.20296.pdf), [[Code]](https://anonymous.4open.science/r/CUT_anonymous-9815)

97. **Robust Federated Contrastive Recommender System against Model Poisoning Attack** (Fed Rec + CL)

     arXiv 2024, [[PDF]](https://arxiv.org/pdf/2403.20107.pdf), [[Code]](https://anonymous.4open.science/r/CUT_anonymous-9815)

98. **Bridging Language and Items for Retrieval and Recommendation** (Multi-Modal + CL)

     arXiv 2024, [[PDF]](https://arxiv.org/pdf/2403.03952.pdf), [[Code]](https://github.com/hyp1231/AmazonReviews2023)

99. **DRepMRec: A Dual Representation Learning Framework for Multimodal Recommendation** (Multi-Modal + CL)

     arXiv 2024, [[PDF]](https://arxiv.org/pdf/2404.11119.pdf)

100. **Knowledge-Aware Multi-Intent Contrastive Learning for Multi-Behavior Recommendation** (Multi-Behavior + CL)

      arXiv 2024, [[PDF]](https://arxiv.org/pdf/2404.11993.pdf)

101. **General Item Representation Learning for Cold-start Content Recommendations** (Cold Start + CL)

      arXiv 2024, [[PDF]](https://arxiv.org/pdf/2404.13808.pdf)

102. **MARec: Metadata Alignment for Cold-start Recommendation** (Cold Start + CL)

      arXiv 2024, [[PDF]](https://arxiv.org/pdf/2404.13298.pdf)

103. **Contrastive Quantization based Semantic Code for Generative Recommendation** (Generative Rec + CL)

      CIKM 2023, [[PDF]](https://arxiv.org/pdf/2404.14774.pdf)

104. **Retrieval-Oriented Knowledge for Click-Through Rate Prediction** (CTR + CL)

      arXiv 2024, [[PDF]](https://arxiv.org/pdf/2404.18304)

105. **Denoising Long-and Short-term Interests for Sequential Recommendation** (Session + DA + CL)

      SDM 2024, [[PDF]](https://epubs.siam.org/doi/pdf/10.1137/1.9781611978032.63), [[Code]](https://github.com/zxyllq/LSIDN)

106. **Learnable Tokenizer for LLM-based Generative Recommendation** (Gen Rec + CL)

      arXiv 2024, [[PDF]](https://arxiv.org/pdf/2405.07314)

107. **MVBIND: Self-Supervised Music Recommendation For Videos Via Embedding Space Binding** (Music Rec + CL)

      arXiv 2024, [[PDF]](https://arxiv.org/pdf/2405.09286)

108. **CELA: Cost-Efficient Language Model Alignment for CTR Prediction** (LLM + CTR + CL)

      arXiv 2024, [[PDF]](https://arxiv.org/pdf/2405.10596)

109. **A Unified Search and Recommendation Framework Based on Multi-Scenario Learning for Ranking in E-commerce** (Search & Rec + CL)

      SIGIR 2024, [[PDF]](https://arxiv.org/pdf/2405.10835)

110. **Learning Structure and Knowledge Aware Representation with Large Language Models for Concept Recommendation** (Concept Rec + CL)

      arXiv 2024, [[PDF]](https://arxiv.org/pdf/2405.12442)

111. **Bilateral Multi-Behavior Modeling for Reciprocal Recommendation in Online Recruitment** (Job Rec + CL)

      TKDE 2024, [[PDF]](https://ieeexplore.ieee.org/abstract/document/10521826/)

112. **Multi-Modal Recommendation Unlearning** (Rec Unlearning + CL)

      arXiv 2024, [[PDF]](https://arxiv.org/pdf/2405.15328)

113. **Your decision path does matter in pre-training industrial recommenders with multi-source behaviors** (Cross-Domain + CL)

      arXiv 2024, [[PDF]](https://arxiv.org/pdf/2405.17132)

114. **NoteLLM-2: Multimodal Large Representation Models for Recommendation** (Multi-Modal + CL)

      arXiv 2024, [[PDF]](https://arxiv.org/pdf/2405.16789)

115. **Multimodality Invariant Learning for Multimedia-Based New Item Recommendation** (Multi-Modal + CL)

      SIGIR 2024, [[PDF]](https://arxiv.org/pdf/2405.15783), [[Code]](https://github.com/HaoyueBai98/MILK)

116. **Cross-Domain LifeLong Sequential Modeling for Online Click-Through Rate Prediction** (CTR + CL)

      arXiv 2024, [[PDF]](https://arxiv.org/pdf/2312.06424)

117. **Medication Recommendation via Dual Molecular Modalities and Multi-Substructure Distillation** (Med Rec + CL)

      arXiv 2024, [[PDF]](https://arxiv.org/pdf/2405.20358)

118. **Item-Language Model for Conversational Recommendation** (Conversational Rec + CL)

      arXiv 2024, [[PDF]](https://arxiv.org/pdf/2406.02844)

119. **Boosting Multimedia Recommendation via Separate Generic and Unique Awareness** (Multi-Modal + CL)

      arXiv 2024, [[PDF]](https://arxiv.org/pdf/2406.08270), [[Code]](https://github.com/bruno686/SAND)

120. **Contextual Distillation Model for Diversified Recommendation** (Diversified Rec + CL)

      KDD 2024, [[PDF]](https://arxiv.org/pdf/2406.09021)

121. **DiffMM: Multi-Modal Diffusion Model for Recommendation** (Multi-Modal + DA + CL)

      MM 2024, [[PDF]](https://arxiv.org/pdf/2406.11781), [[Code]](https://github.com/HKUDS/DiffMM)

122. **Improving Multi-modal Recommender Systems by Denoising and Aligning Multi-modal Content and User Feedback** (Multi-Modal + CL)

      KDD 2024, [[PDF]](https://arxiv.org/pdf/2406.12501), [[Code]](https://github.com/XMUDM/DA-MRS)

123. **EAGER: Two-Stream Generative Recommender with Behavior-Semantic Collaboration** (Gen Rec + CL)

      KDD 2024, [[PDF]](https://arxiv.org/pdf/2406.14017)

124. **Enhancing Collaborative Semantics of Language Model-Driven Recommendations via Graph-Aware Learning** (LLM Rec + CL)

      arXiv 2024, [[PDF]](https://arxiv.org/pdf/2406.13235)

125. **Hyperbolic Knowledge Transfer in Cross-Domain Recommendation System** (Cross-Domain + CL)

      arXiv 2024, [[PDF]](https://arxiv.org/pdf/2406.17289)

126. **MMBee: Live Streaming Gift-Sending Recommendations via Multi-Modal Fusion and Behaviour Expansion** (Gift-Sending Rec + CL)

      KDD 2024, [[PDF]](https://arxiv.org/pdf/2407.00056)

127. **Adapting Job Recommendations to User Preference Drift with Behavioral-Semantic Fusion Learning** (Job Rec + CL)

      KDD 2024, [[PDF]](https://arxiv.org/pdf/2407.00082)

128. **Personalised Outfit Recommendation via History-aware Transformers** (Outfit Rec + CL)

      arXiv 2024, [[PDF]](https://arxiv.org/pdf/2407.00289)

129. **Unified Dual-Intent Translation for Joint Modeling of Search and Recommendation** (Search & Rec + CL)

      KDD 2024, [[PDF]](https://arxiv.org/pdf/2407.00912), [[Code]](https://github.com/17231087/UDITSR)

130. **Language Models Encode Collaborative Signals in Recommendation** (LLM + Graph + CL)

      arXiv 2024, [[PDF]](https://arxiv.org/pdf/2407.05441), [[Code]](https://github.com/LehengTHU/AlphaRec)

131. **GUME: Graphs and User Modalities Enhancement for Long-Tail Multimodal Recommendation** (Multi-Modal + CL)

      arXiv 2024, [[PDF]](https://arxiv.org/pdf/2407.12338), [[Code]](https://github.com/NanGongNingYi/GUME)

132. **A Unified Graph Transformer for Overcoming Isolations in Multi-modal Recommendation** (Multi-Modal + CL)

      arXiv 2024, [[PDF]](https://arxiv.org/pdf/2407.19886)

133. **MOSAIC: Multimodal Multistakeholder-aware Visual Art Recommendation** (Art Rec + CL)

      arXiv 2024, [[PDF]](https://arxiv.org/pdf/2407.21758)

134. **Disentangled Contrastive Hypergraph Learning for Next POI Recommendation** (POI Rec + DA + CL)

      SIGIR 2024, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3626772.3657726), [[Code]](https://github.com/icmpnorequest/SIGIR2024_DCHL)

135. **Modeling User Intent Beyond Trigger: Incorporating Uncertainty for Trigger-Induced Recommendation** (CTR + CL)

      CIKM 2024, [[PDF]](https://arxiv.org/pdf/2408.03091), [[Code]](https://github.com/majx1997/DUIN)

136. **SimCEN: Simple Contrast-enhanced Network for CTR Prediction** (CTR + CL)

      MM 2024, [[PDF]](https://openreview.net/pdf?id=pJHu4hDlLX), [[Code]](https://github.com/salmon1802/SimCEN)

137. **CETN: Contrast-enhanced Through Network for Click-Through Rate Prediction** (CTR + CL)

      TOIS 2024, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3688571), [[Code]](https://github.com/salmon1802/CETN)

138. **Multi-task Heterogeneous Graph Learning on Electronic Health Records** (Drug Rec + CL)

      NN 2024, [[PDF]](https://arxiv.org/pdf/2408.07569), [[Code]](https://github.com/HKU-MedAI/MulT-EHR)

139. **Don’t Click the Bait: Title Debiasing News Recommendation via Cross-Field Contrastive Learning** (News Rec + CL)

      arXiv 2024, [[PDF]](https://arxiv.org/pdf/2408.08538)

140. **EasyRec: Simple yet Effective Language Models for Recommendation** (LLM + CL)

      arXiv 2024, [[PDF]](https://arxiv.org/pdf/2408.08821), [[Code]](https://github.com/HKUDS/EasyRec)

141. **Bundle Recommendation with Item-level Causation-enhanced Multi-view Learning** (Bundle Rec + CL)

      arXiv 2024, [[PDF]](https://arxiv.org/pdf/2408.08906)

142. **Debiased Contrastive Representation Learning for Mitigating Dual Biases in Recommender Systems** (Debias + CL)

      arXiv 2024, [[PDF]](https://arxiv.org/pdf/2408.09646)

143. **LARR: Large Language Model Aided Real-time Scene Recommendation with Semantic Understanding** (CTR + LLM + CL)

      RecSys 2024, [[PDF]](https://arxiv.org/pdf/2408.11523)

144. **Federated User Preference Modeling for Privacy-Preserving Cross-Domain Recommendation** (Cross-Domain + Fed Rec + CL)

      arXiv 2024, [[PDF]](https://arxiv.org/pdf/2408.14689), [[Code]](https://github.com/Lili1013/FUPM)

145. **Mitigating Negative Transfer in Cross-Domain Recommendation via Knowledge Transferability Enhancement** (Cross-Domain + CL)

      KDD 2024, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3637528.3671799)

146. **Federated Prototype-based Contrastive Learning for Privacy-Preserving Cross-domain Recommendation** (Cross-Domain + CL)

      arXiv 2024, [[PDF]](https://arxiv.org/pdf/2409.03294)

147. **A Unified Framework for Cross-Domain Recommendation** (Cross-Domain + CL)

      arXiv 2024, [[PDF]](https://arxiv.org/pdf/2409.04540)

148. **End-to-End Learnable Item Tokenization for Generative Recommendation** (Gen Rec + CL)

      arXiv 2024, [[PDF]](https://arxiv.org/pdf/2409.05546)

149. **Towards Leveraging Contrastively Pretrained Neural Audio Embeddings for Recommender Tasks** (Music Rec + CL)

      RecSys 2024, [[PDF]](https://arxiv.org/pdf/2409.09026)

150. **A Multimodal Single-Branch Embedding Network for Recommendation in Cold-Start and Missing Modality Scenarios** (Multi-Modal + CL)

     RecSys 2024, [[PDF]](https://arxiv.org/pdf/2409.17864), [[Code]](https://github.com/hcai-mms/SiBraR---Single-Branch-Recommender)

151. **The Devil is in the Sources! Knowledge Enhanced Cross-Domain Recommendation in an Information Bottleneck Perspective** (Cross-Domain + CL)

     CIKM 2024, [[PDF]](https://arxiv.org/pdf/2409.19574)

152. **Contrastive Clustering Learning for Multi-Behavior Recommendation** (Multi-Behavior + CL)

     TOIS 2024, [[PDF]](https://dl.acm.org/doi/10.1145/3698192), [[Code]](https://github.com/lanbiolab/MBRCC)

153. **End-to-End Learnable Item Tokenization for Generative Recommendation** (Gen Rec + CL)

      arXiv 2024, [[PDF]](https://arxiv.org/pdf/2410.02939), [[Code]](https://github.com/Jamesding000/SpecGR)

154. **Improving Object Detection via Local-global Contrastive Learning** (OD + CL)

      BMVC 2024, [[PDF]](https://arxiv.org/pdf/2410.05058), [[Code]](https://local-global-detection.github.io/)

155. **DISCO: A Hierarchical Disentangled Cognitive Diagnosis Framework for Interpretable Job Recommendation** (Job Rec + CL)

      ICDM 2024, [[PDF]](https://arxiv.org/pdf/2410.07671), [[Code]](https://github.com/LabyrinthineLeo/DISCO)

156. **Neural Contrast: Leveraging Generative Editing for Graphic Design Recommendations** (Design Rec + CL)

      PRICAI 2024, [[PDF]](https://arxiv.org/pdf/2410.07211)

157. **Pseudo Dataset Generation for Out-of-domain Multi-Camera View Recommendation** (View Rec + CL)

      arXiv 2024, [[PDF]](https://arxiv.org/pdf/2410.13585)

158. **Hyperbolic Contrastive Learning for Cross-Domain Recommendation** (Cross-Domain + CL)

     CIKM 2024, [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3627673.3679572), [[Code]](https://github.com/EnkiXin/hcts)

159. **Enhancing CTR prediction in Recommendation Domain with Search Query Representation** (CTR + CL)

     CIKM 2024, [[PDF]](https://arxiv.org/pdf/2410.21487)

160. **Multi-Modal Correction Network for Recommendation** (Multi-Modal + CL)

     TKDE 2024, [[PDF]](https://ieeexplore.ieee.org/document/10746604)

161. **QARM: Quantitative Alignment Multi-Modal Recommendation at Kuaishou** (Multi-Modal + CL)

     arXiv 2024, [[PDF]](https://arxiv.org/pdf/2411.11739)

162. **Collaborative Contrastive Network for Click-Through Rate Prediction** (CTR + CL)

     arXiv 2024, [[PDF]](https://arxiv.org/pdf/2411.11508)

163. **Hierarchical Denoising for Robust Social Recommendation** (Social Rec + CL)

     TKDE 2024, [[PDF]](https://ieeexplore.ieee.org/document/10771708)
