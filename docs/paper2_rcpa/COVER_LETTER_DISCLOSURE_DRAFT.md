# Cover Letter Disclosure Draft (Paper 2)

> **Do not submit now.** Choose Version A or B based on Paper 1 status at submission time.

---

## Version A — Paper 1 accepted or in final publication

Dear Editor,

We submit the manuscript entitled *"Diagnosing and Mitigating Receiver-Induced OOB Feature Entanglement in LoRa Radio-Frequency Fingerprinting"* for consideration as a regular paper in the IEEE Internet of Things Journal.

This manuscript is **related to our prior IoTJ work** on OOB-guided cross-attentive RF-HSTU hybrid modeling for LoRa device authentication [cite upon publication]. That prior work focuses on **backbone architecture design** and **source-only same-receiver cross-day robustness**, and reports strict cross-receiver transfer as an open limitation.

The present manuscript addresses a **distinct problem and contribution**:
1. We diagnose receiver-induced OOB feature entanglement under cross-receiver shift.
2. We show source-trained classifier/prototype non-transferability.
3. We propose RCPA, a **post-hoc target-receiver prototype calibration protocol** using $K$ labeled calibration windows per device on a **frozen backbone**.

**No new backbone architecture is claimed.** Overlapping background on LoRa RFFI and the hybrid extractor is rewritten; experiments, figures, tables, and conclusions are distinct. We are happy to provide the prior published manuscript if helpful.

Sincerely,  
[Authors]

---

## Version B — Paper 1 under review / decision pending

Dear Editor,

We submit the manuscript entitled *"Diagnosing and Mitigating Receiver-Induced OOB Feature Entanglement in LoRa Radio-Frequency Fingerprinting"* for consideration as a regular paper in the IEEE Internet of Things Journal.

We disclose that this manuscript is **related to a concurrently submitted manuscript** to the IEEE Internet of Things Journal entitled:

> *"OOB-Guided Cross-Attentive RF-HSTU Hybrid Modeling for Robust LoRa Device Authentication"*
> Status: **[Under Review / Minor Revision / placeholder]**

The concurrent manuscript proposes an OOB-guided hybrid **architecture** and evaluates **source-only same-receiver cross-day robustness**.
The present manuscript **does not propose a new architecture**. Instead, it:
- diagnoses cross-receiver OOB entanglement on a frozen instance of that hybrid model;
- introduces a target-receiver prototype calibration protocol (RCPA);
- reports new cross-receiver experiments, ablations, and unsupervised adaptation negative baselines.

We have rewritten all shared background text to minimize overlap and can provide the related manuscript upon request.

Sincerely,  
[Authors]
