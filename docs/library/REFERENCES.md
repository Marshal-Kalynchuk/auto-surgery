# External References

**Purpose:** Canonical upstream links for the models, libraries, and datasets referenced by `docs/ARCHITECTURE.md` and `docs/ARCHITECTURE_THESIS.md`.

**Verified against:** `docs/ARCHITECTURE.md` v2.5 and `docs/ARCHITECTURE_THESIS.md` on 2026-05-07. Updated with comparator/baseline citations 2026-05-07 to support the thesis's positioning sections.

Use this file as the first stop before broad web search. Keep agent and rule files pointed here instead of duplicating long link lists.

---

## Core Foundation Models

### DINOv2
- Repository: [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)
- Paper: [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)

### V-JEPA 2
- Repository: [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2)
- Paper: [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://arxiv.org/abs/2506.09985)
- Project page: [Introducing V-JEPA 2](https://ai.meta.com/research/vjepa/)
- Model docs: [Hugging Face V-JEPA 2](https://huggingface.co/docs/transformers/en/model_doc/vjepa2)

### DINO-WM
- Repository: [gaoyuezhou/dino_wm](https://github.com/gaoyuezhou/dino_wm)
- Project page: [DINO-WM](https://dino-wm.github.io/)
- Paper: [DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning](https://arxiv.org/abs/2411.04983)

### Modern Hopfield Networks
- Canonical paper: [Modern Hopfield Networks and Attention for Immune Repertoire Classification](https://proceedings.neurips.cc/paper/2020/hash/da4902cb0bc38210839714ebdcf0efc3-Abstract.html)

---

## Object-Centric Video / Slot Attention

### DINOSAUR
- Project page: [DINOSAUR](https://dinosaur-paper.github.io/)
- Paper: [Bridging the Gap to Real-World Object-Centric Learning](https://openreview.net/forum?id=b9tUk-f_aG)
- Implementation: [Object-centric Learning Framework](https://github.com/amazon-science/object-centric-learning-framework)

### SAVi++ / Slot Attention Video
- Repository: [google-research/slot-attention-video](https://github.com/google-research/slot-attention-video)
- Project page: [SAVi++: Towards End-to-End Object-Centric Learning from Real-World Videos](https://slot-attention-video.github.io/savi++/)

---

## Geometry, Control, and Robot Learning

### SE(3)-Transformer
- Repository: [FabianFuchsML/se3-transformer-public](https://github.com/FabianFuchsML/se3-transformer-public)
- Project page: [SE(3)-Transformer](https://fabianfuchsml.github.io/se3transformer/)
- Paper: [SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks](https://arxiv.org/abs/2006.10503)

### Equiformer
- Repository: [atomicarchitects/equiformer](https://github.com/atomicarchitects/equiformer)
- Paper: [Equiformer: Equivariant Graph Attention Transformer for 3D Atomistic Graphs](https://arxiv.org/abs/2206.11990)

### BRIDGE Data V2
- Project page: [BridgeData V2](https://bridgedata-v2.github.io/)
- Paper: [BridgeData V2: A Dataset for Robot Learning at Scale](https://proceedings.mlr.press/v229/walke23a.html)
- Code: [rail-berkeley/bridge_data_robot](https://github.com/rail-berkeley/bridge_data_robot)

### RoboNet
- Project page: [RoboNet](https://www.robonet.wiki/)
- Paper: [RoboNet: Large-Scale Multi-Robot Learning](https://proceedings.mlr.press/v100/dasari20a.html)

### OpenVLA
- Project page: [OpenVLA](https://openvla.github.io/)
- Repository: [openvla/openvla](https://github.com/openvla/openvla)
- Paper: [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246)

---

## Simulation and Safety Infrastructure

### SOFA
- Documentation: [SOFA Documentation](https://sofa-framework.github.io/doc/)
- Homepage: [SOFA - Simulation Open Framework Architecture](https://www.sofa-framework.org/)
- Repository: [sofa-framework/sofa](https://github.com/sofa-framework/sofa)
- Inverse model reference: [SOFA inverse model for soft-robot control](https://www.sofa-framework.org/applications/plugins/inverse-model-for-soft-robot-control/)

### Target Robot Reference
- Robot summary: [`docs/TARGET_ROBOT_KUKA_KR6.md`](./TARGET_ROBOT_KUKA_KR6.md)

---

## Surgical Robotics Autonomy and Levels-of-Autonomy

### Levels of Autonomy taxonomies
- Yang, Cambias, Cleary, Daimler, Drake, Dupont, Hata, Kazanzides, Martel, Patel, Santos, Taylor — *Medical robotics — Regulatory, ethical, and legal considerations for increasing levels of autonomy*. Science Robotics, 2017. (Defines the 0–5 surgical autonomy scale that this thesis positions itself against.)
- Haidegger — *Autonomy for Surgical Robots: Concepts and Paradigms*. IEEE Trans. Med. Robot. Bionics, 2019.
- Attanasio, Scaglioni, De Momi, Fiorini, Valdastri — *Autonomy in Surgical Robotics*. Annu. Rev. Control Robot. Auton. Syst., 2021.

### Research and clinical surgical robots referenced for positioning
- Smart Tissue Autonomous Robot (STAR) — Saeidi et al., *Autonomous robotic laparoscopic surgery for intestinal anastomosis*, Science Robotics, 2022.
- da Vinci Research Kit (dVRK) — [JHU dVRK](https://research.intusurg.com/dvrk/).
- da Vinci SP, Auris/Monarch, CMR Versius, Stryker Mako, ROSA (Zimmer Biomet), CyberKnife (Accuray), ROBODOC — commercial platforms; cited descriptively for paradigm positioning, no canonical paper.
- NeuroArm (University of Calgary) — [project page](https://www.neuroarm.org/).

---

## Brain-Shift Compensation and Image-Guided Neurosurgery

- Roberts, Hartov, Kennedy, Miga, Paulsen — *Intraoperative brain shift and deformation: A quantitative analysis of cortical displacement in 28 cases*. Neurosurgery, 1998.
- Miga, Sun, Chen, Clements, Pheiffer, Simpson, Thompson — *Clinical evaluation of a model-updated image-guidance approach to brain shift compensation*. Int. J. Comput. Assist. Radiol. Surg.
- Bayer, Maier-Hein, Frenzel, Lasso, Tonetti, Wood, Maier-Hein — *Intraoperative imaging modalities and compensation for brain shift in tumor resection: a comprehensive review*. Comput. Math. Methods Med., 2017.
- Heinrich, Jenkinson, Brady, Schnabel — *MRF-Based Deformable Registration and Ventilation Estimation of Lung CT*. IEEE Trans. Med. Imaging, 2013. (MRF deformable registration baseline.)
- Heinrich, Jenkinson — *MIND: Modality independent neighbourhood descriptor for multi-modal deformable registration*. Med. Image Anal., 2012.
- Commercial IGNS systems referenced descriptively: Medtronic StealthStation, Brainlab Curve / Cranial Navigation.

---

## Formal Safety, Runtime Assurance, and Conformal Prediction for Control

### Control Barrier Functions and reachability
- Ames, Coogan, Egerstedt, Notomista, Sreenath, Tabuada — *Control Barrier Functions: Theory and Applications*. ECC, 2019. [arXiv:1903.11199](https://arxiv.org/abs/1903.11199).
- Bansal, Chen, Herbert, Tomlin — *Hamilton-Jacobi Reachability: A Brief Overview and Recent Advances*. CDC, 2017. [arXiv:1709.07523](https://arxiv.org/abs/1709.07523).

### Runtime assurance / Simplex
- Sha — *Using Simplicity to Control Complexity*. IEEE Software, 2001.
- Schierman, DeVore, Richards, Clark — *Runtime Assurance Framework Development for Highly Adaptive Flight Control Systems*. AIAA, 2015.

### Surgical safety primitives
- Abbott, Marayong, Okamura — *Haptic Virtual Fixtures for Robot-Assisted Manipulation*. ISRR, 2007.
- Davies, Rodriguez y Baena, Barrett, Gomes, Harris, Jakopec, Cobb — *Acrobot: a robot for cooperative orthopaedic surgery*. ICAR, 2007.
- Bowyer, Davies, Rodriguez y Baena — *Active Constraints/Virtual Fixtures: A Survey*. IEEE Trans. Robot., 2014.
- Marayong, Bettini, Okamura — *Effect of virtual fixture compliance on human-machine cooperative manipulation*. IROS, 2002.

### Conformal prediction for control and safe robot learning
- Vovk, Gammerman, Shafer — *Algorithmic Learning in a Random World*. Springer, 2005. (Canonical text.)
- Angelopoulos, Bates — *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*. [arXiv:2107.07511](https://arxiv.org/abs/2107.07511), 2021.
- Lindemann, Cleaveland, Shim, Pappas — *Safe Planning in Dynamic Environments using Conformal Prediction*. RA-L, 2023. [arXiv:2210.10254](https://arxiv.org/abs/2210.10254).
- Tibshirani, Foygel Barber, Candès, Ramdas — *Conformal Prediction Under Covariate Shift*. NeurIPS, 2019. (Coverage under distribution shift.)

---

## World Models (Comparators)

- Hafner, Pasukonis, Ba, Lillicrap — *Mastering Diverse Domains through World Models* (DreamerV3). [arXiv:2301.04104](https://arxiv.org/abs/2301.04104), 2023.
- Schrittwieser et al. — *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model* (MuZero). Nature, 2020. [arXiv:1911.08265](https://arxiv.org/abs/1911.08265).
- Micheli, Alonso, Fleuret — *Transformers are Sample-Efficient World Models* (IRIS). ICLR, 2023. [arXiv:2209.00588](https://arxiv.org/abs/2209.00588).
- Wu, Escontrela, Hafner, Goldberg, Abbeel — *DayDreamer: World Models for Physical Robot Learning*. CoRL, 2022.

---

## Vision-Language-Action (VLA) Models (Comparators)

- Brohan et al. — *RT-1: Robotics Transformer for Real-World Control at Scale*. RSS, 2023. [arXiv:2212.06817](https://arxiv.org/abs/2212.06817).
- Brohan et al. — *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*. CoRL, 2023. [arXiv:2307.15818](https://arxiv.org/abs/2307.15818).
- Octo Model Team — *Octo: An Open-Source Generalist Robot Policy*. [arXiv:2405.12213](https://arxiv.org/abs/2405.12213), 2024.
- Black et al. — *π0: A Vision-Language-Action Flow Model for General Robot Control*. [arXiv:2410.24164](https://arxiv.org/abs/2410.24164), 2024.
- Open X-Embodiment Collaboration — *Open X-Embodiment: Robotic Learning Datasets and RT-X Models*. [arXiv:2310.08864](https://arxiv.org/abs/2310.08864), 2023.
- Baker et al. — *Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos*. NeurIPS, 2022. [arXiv:2206.11795](https://arxiv.org/abs/2206.11795). (Inverse-dynamics-from-video reference.)

---

## Memory Architectures (Comparators)

- Graves et al. — *Hybrid computing using a neural network with dynamic external memory* (Differentiable Neural Computer). Nature, 2016.
- Wu, Rabe, Hutchins, Szegedy — *Memorizing Transformers*. ICLR, 2022. [arXiv:2203.08913](https://arxiv.org/abs/2203.08913).
- Borgeaud et al. — *Improving Language Models by Retrieving from Trillions of Tokens* (RETRO). ICML, 2022. [arXiv:2112.04426](https://arxiv.org/abs/2112.04426).
- Ramsauer et al. — *Hopfield Networks Is All You Need*. ICLR, 2021. [arXiv:2008.02217](https://arxiv.org/abs/2008.02217). (Modern Hopfield capacity reference.)

---

## Object-Centric / Tracking / Surgical-Scene Perception (Baselines)

- Locatello et al. — *Object-Centric Learning with Slot Attention*. NeurIPS, 2020. [arXiv:2006.15055](https://arxiv.org/abs/2006.15055).
- Kipf et al. — *Conditional Object-Centric Learning from Video* (SAVi). ICLR, 2022. [arXiv:2111.12594](https://arxiv.org/abs/2111.12594).
- Elsayed et al. — *SAVi++: Towards End-to-End Object-Centric Learning from Real-World Videos*. NeurIPS, 2022. [arXiv:2206.07764](https://arxiv.org/abs/2206.07764).
- Carion et al. — *End-to-End Object Detection with Transformers* (DETR). ECCV, 2020.
- Zeng et al. — *MOTR: End-to-End Multiple-Object Tracking with Transformer*. ECCV, 2022.
- Campos et al. — *ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map SLAM*. IEEE Trans. Robot., 2021.
- Wang et al. — *Neural Rendering for Stereo 3D Reconstruction of Deformable Tissues in Robotic Surgery* (EndoNeRF). MICCAI, 2022.
- Özsoy et al. — *4D-OR: Semantic Scene Graphs for OR Domain Modeling*. MICCAI, 2022.
- Twinanda et al. — *EndoNet: A Deep Architecture for Recognition Tasks on Laparoscopic Videos* (Cholec80). IEEE Trans. Med. Imaging, 2017.

---

## Surgical Datasets and Benchmarks

- JIGSAWS — Gao et al., *JHU-ISI Gesture and Skill Assessment Working Set*, MICCAI Workshop M2CAI, 2014.
- Cholec80 — Twinanda et al., 2017 (see above).
- CATARACTS Challenge — Al Hajj et al., *CATARACTS: Challenge on Automatic Tool Annotation for cataRACT Surgery*. Med. Image Anal., 2019.
- M2CAI Workshop — see [Endoscopic Vision Challenge series](https://endovissub-instrument.grand-challenge.org/).
- EndoVis Challenge — yearly MICCAI sub-challenges on instrument segmentation, scene segmentation, and tool tracking.
- SurgT — Cartucho et al., *SurgT: Soft-Tissue Tracking for Robotic Surgery*. (Used in this thesis for tracking thresholds.)
- FLS Peg Transfer — Fundamentals of Laparoscopic Surgery, [SAGES](https://www.sages.org/).

---

## Surgical Foundation / Vision-Language Models

- Ma et al. — *Segment Anything in Medical Images* (MedSAM). Nat. Commun., 2024. [arXiv:2304.12306](https://arxiv.org/abs/2304.12306).
- Ramesh et al. — *Surgical-VQLA: Transformer with Gated Vision-Language Embedding for Visual Question Localized-Answering in Robotic Surgery*. ICRA, 2023.
- Yuan et al. — *SurgicalGPT: End-to-End Language-Vision GPT for Visual Question Answering in Surgery*. MICCAI, 2023.
- Yuan, Bhatia, Karargyris, et al. — *EndoFM*: surgical-domain video foundation model (where applicable).

---

## Regulatory and Ethical Standards

- IEC 62304 — *Medical device software — Software life cycle processes*.
- IEC 60601-2-77 — *Medical electrical equipment — Particular requirements for the basic safety and essential performance of robotically assisted surgical equipment*.
- FDA Software as a Medical Device (SaMD) — [FDA SaMD guidance](https://www.fda.gov/medical-devices/digital-health-center-excellence/software-medical-device-samd).
- ISO 14971 — *Medical devices — Application of risk management to medical devices*.
- IDEAL-D Framework — McCulloch et al., *Stages of innovation in surgery: a long evaluation cycle*. Lancet, 2009; IDEAL-D for medical devices, 2018.

---

## Deferred or Optional References

### SAM 2
- Repository: [facebookresearch/sam2](https://github.com/facebookresearch/SAM2)
- Paper: [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714)
- Model docs: [Hugging Face SAM 2](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/sam2.md)

### VL-JEPA
- Paper: [VL-JEPA: Joint Embedding Predictive Architecture for Vision-language](https://arxiv.org/abs/2512.10942)
- Paper page: [Hugging Face paper page](https://huggingface.co/papers/2512.10942)

### Mamba-2
- Paper: [Mamba-2](https://arxiv.org/abs/2405.21060)
- Docs: [Hugging Face Transformers Mamba-2](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/mamba2.md)

### Jamba
- Research page: [Jamba: A Hybrid Transformer-Mamba Language Model](https://www.ai21.com/research/jamba-a-hybrid-transformer-mamba-language-model)
- Announcement: [Introducing Jamba](https://ai21.com/blog/announcing-jamba)
- Paper page: [Hugging Face paper page](https://hf.co/papers/2403.19887)

---

## Update Rule

If a new model, dataset, or library becomes load-bearing for this architecture, add it here first, then reference this file from the relevant rule or agent definition.
