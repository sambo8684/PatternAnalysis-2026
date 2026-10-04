# Analysing HipMRI using 2D U-Net for Prostate Radiotherapy Planning

U-Net is a type of convolutional neural network (CNN) developed to assist with biomedical imaging tasks. The encoder-decoder structure is useful for 2D images and isolating specific anatomical structures; in this case it is 2D MRI scans of hips that require radiotherapy for prostate tumours. Radiotherapy requires millimeter level precision so that tumours are accurately targeted and critical organs are spared. Specifically with 2D U-Net, under-contouring risks recoccuring tumours whilst over-conturing risks radiation toxicity. 

structure from spec (delete before final submission):
• how it works in a paragraph, and a figure/visualisation.
• Your one-page written feasibility review from Section 3.
• Any dependencies required, including exact versions, and address reproducibility of results.
• Example inputs, outputs, and visual plots of your algorithm.
• Description of any specific pre-processing used, with references if any. Justify your training,
validation, and testing splits of the data.
• Investigation of the Open Research Dilemma:
– Quantitative benchmarking of your chosen model against your implemented baseline model under
identical evaluation splits.
– Resource profiling table documenting computational efficiency (peak GPU VRAM, parameter
count, and inference runtime/latency).
– In-depth qualitative error autopsies analyzing 3–5 representative failure cases from the dataset
(diagnosing root triggers and failure modes).
– Actionable engineering recommendation and trade-off synthesis for your project manager.
• The mandatory Artificial Intelligence Usage Disclosure subsection (see Section 6).