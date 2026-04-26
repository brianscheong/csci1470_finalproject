## CSCI 1470 Final Project Check-in \#2

Brian Cheong, Laura McCallion, Kirill Vesialou  
Week of 4/19/26

### Introduction

This project critically replicates and analyzes the VoxelMorph framework for deformable medical image registration, focusing on stress-testing its core claims and understanding its failure modes. We evaluate whether VoxelMorph achieves accuracy comparable to classical optimization-based methods as claimed and whether its amortized (globally trained) predictions approximate per-instance optimal deformation fields. To do so, we measure registration quality (via Dice Score), quantify the amortization gap (the difference with per-instance registration), and stress test performance under distribution shifts such as intensity scaling and modality-shift (i.e. testing it on CT instead of MRI). In parallel, we conduct controlled ablation studies on key design choices, including smoothness regularization in the loss function, to assess its impact on alignment accuracy and robustness. Together, these experiments aim to move beyond benchmarking performance and provide a more rigorous characterization of when and why learned registration methods succeed or break down.

### Challenges

The most challenging aspects of the project have been largely practical rather than conceptual. Initially, there was a learning curve associated with setting up the Oscar cluster environment, configuring a remote IDE, and correctly submitting SLURM jobs. While these issues required time to resolve, they were not directly related to the core intellectual goals of the project.

A more substantive challenge arose in determining appropriate preprocessing and training procedures, given the limited methodological detail provided in the original paper. Although the authors mention steps such as affine alignment, the implementation specifics are left largely unspecified. As a result, we had to make several design decisions independently, including how to crop, center, and optimize affine transformations. These choices introduced some ambiguity, particularly regarding whether our preprocessing pipeline might be overly strong and inadvertently simplify the registration task.

Despite this uncertainty, the resulting Dice scores are reasonable and remain below those reported in the paper for the target modality and anatomy, suggesting that our approach has not artificially inflated performance and is broadly consistent with expected outcomes.

### Insights

Are there any concrete results you can show at this point? How is your model performing compared with expectations?

Yes. After preprocessing the KiTS dataset (kidney CT volumes), we completed an initial end-to-end training and evaluation run using the baseline configuration—specifically, the recommended similarity loss and smoothness regularization (two components we plan to systematically vary in later experiments).

The resulting Dice scores fall in a reasonable range: they indicate meaningful alignment performance, yet remain below the values reported in the original paper for its target modality and anatomy. This is expected given differences in data and setup, and importantly suggests that our pipeline is functioning correctly without artificially inflating results.

These preliminary results validate our implementation and provide a stable baseline, giving us confidence to proceed with the planned ablation studies and parameter sweeps.

### Plan

What do you need to dedicate more time to? What are you thinking of changing, if anything?

We are on track overall and have made substantial progress in establishing the workflow. In particular, we have become comfortable working with the Oscar cluster and have successfully built and validated our training and testing pipeline. Our initial round of experiments confirms that the setup is functioning as intended and that the implementation choices we made during preprocessing and training are reasonable.

Although we have not yet begun the full set of planned experiments, this setup phase was expected to be one of the most time-intensive parts of the project. Completing it represents a meaningful milestone and positions us well for more consistent progress moving forward.

At this stage, our primary focus will be on refining the training and testing scripts to ensure we can reliably generate and track Dice score outcomes across different configurations. There are no major changes required to the overall approach; the priority is to build on the current foundation and proceed into systematic experimentation and analysis.
