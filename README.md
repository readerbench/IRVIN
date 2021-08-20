# InterpRetable Vulnerability IdeNtification (IRVIN)

This is the public repository of the study in which we introduce an interpretable approach to the subject of cyberthreat early detection. In particular, an interpretable classification is performed using the Longformer architecture alongside prototypes from the ProSeNet structure, after performing a preliminary analysis on the Transformer’s encoding capabilities. The best interpretable architecture achieves an 88\% F2-Score, arguing for the system's applicability in real-life monitoring conditions of OSINT data.

# Performance

Corpus: 1600 news articles from two collections of labeled articles:
* Iorga, D., Corlatescu, D.-G., Grigorescu, O., Sandescu, C., Dascalu, M., & Rughinis, R. (2020). Early Detection of Vulnerabilities from News Websites using Machine Learning Models. In 19th RoEduNet Conference: Networking in Education and Research Bucharet, Romania (Online): IEEE.
* Iorga, D., Corlatescu, D.-G., Grigorescu, O., C., S., Dascalu, M., & Rughinis, R. (2021). Yggdrasil – Early Detection of Cybernetic Vulnerabilities from Twitter. In 23rd Conference on Control Systems and Computer Science. Bucharest, Romania (Online): IEEE.

| Model  | Accuracy | Precision | Recall | F2-Score | Interpretable |
|---| --- | --- | --- | --- | --- |
| `MNB` | 0.84 | **0.92** | 0.62 | 0.66 | Yes |
| `Longformer (no pre-training)` | **0.87** | 0.76 | 0.95 | 0.90 | No |
| `Longformer (pre-trained)` | 0.86 | 0.73 | **0.98** | **0.92** | No |
| `Longformer+ProSeNet` | **0.87** | 0.78 | 0.91 | 0.88 | Yes |

# BibTeX entry and citation info

@inproceedings{delaForet2021,
    author = {Frode de la Foret, P. and Ruseti, S. and Sandescu, C. and Dascalu, M. and Travadel, S.},  
    title = {Yggdrasil - Explainable Identification of Cybersecurity Vulnerabilities from News Articles},  
    booktitle = {Int. Conf. on Recent Advances in Natural Language Processing (RANLP 2021)},  
    publisher = {ACL},  
    year = {in press},  
    type = {Conference Proceedings}  
}
