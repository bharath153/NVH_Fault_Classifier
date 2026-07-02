# Interview Q&A — NVH Drivetrain Fault Classifier from Vibration Signals
## Random Forest on FFT-band + statistical features | NVH & Predictive Maintenance

### Q1: Why use frequency domain features for fault classification?

Mechanical faults generate characteristic frequency signatures. Bearing outer-race faults produce impulses at the Ball Pass Frequency Outer (BPFO), gear defects modulate the mesh frequency, and imbalance produces a 1× shaft frequency peak. Time-domain signals are noisy and hard to classify; the frequency domain separates these signatures clearly, making classification much easier.

---

### Q2: What is a CNN spectrogram approach and how does it compare?

A CNN on spectrograms (Short-Time Fourier Transform images) treats the 2D time-frequency representation as an image classification problem. CNNs can learn spatial patterns in spectrograms that correspond to fault signatures. The advantage over hand-crafted FFT features is automatic feature learning. The disadvantage is data hunger and longer training times. For small datasets (< 10,000 samples), hand-crafted features + Random Forest often matches or beats CNN performance.

---

### Q3: What are real-world NVH datasets?

CWRU Bearing Dataset (Case Western Reserve University) — most widely used bearing fault dataset. MFPT Bearing Dataset (Machinery Failure Prevention Technology). PHM Society 2009 Challenge dataset (gearbox). Paderborn University Bearing Dataset. These are all freely available and referenced in hundreds of ML papers.

---

### Q4: What is Crest Factor and why is it used for bearing diagnostics?

Crest Factor = Peak value / RMS value. For a pure sine wave it is √2 ≈ 1.41. Impulsive signals (like bearing faults) have high crest factors because the peaks (impact events) are much larger than the overall vibration level. A Crest Factor > 6 is a traditional bearing fault indicator in ISO 13373.

---

### Q5: How would you deploy this at an OEM assembly line?

End-of-Line (EOL) NVH testing is already standard at plants like Stellantis Mirafiori, BMW Leipzig, and Mercedes-Benz Sindelfingen. A vibration sensor is mounted on the gearbox/axle, a 10-second run-up sequence is performed, FFT features are extracted, and the classifier makes a pass/fail decision in under 1 second. The ML model replaces or augments human NVH engineers listening with headsets.

---

