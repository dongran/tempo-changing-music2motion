# JoruriPuppet: Learning Tempo-Changing Mechanisms Beyond the Beat for Music-to-Motion Generation with Expressive Metrics

This repository accompanies **SIGGRAPH Asia 2025** and provides:
- **SMPL-based** motion capture / dataset processing pipeline
- **Tempo-changing** music features for music-to-motion
- **Expressive motion metrics** (Jo–Ha–Kyu, S-curve, Head–Hand Contrast)

![Representative Image](asset/Representative_Image.jpg)

This repository provides companion code for the paper **“JoruriPuppet: Learning Tempo‑Changing Mechanisms Beyond the Beat for Music‑to‑Motion Generation with Expressive Metrics”** (SIGGRAPH Asia 2025) and the associated project page:  
[JoruriPuppet project page](https://dr-lab.org/projects/joruripuppet/).

The public dataset (motions, SMPL/NPZ/TRC, and audio) is distributed from the project page.  
This repository focuses on **how the data are processed and evaluated**, and how **tempo‑changing music features** are constructed for neural music‑to‑motion models.

### Repository layout

- `dataset/` – Minimal examples showing how raw puppet BVH motions are
  1. prepared for MotionBuilder,
  2. retargeted to an SMPL skeleton inside MotionBuilder, and
  3. converted into SMPL‑style `.npz` files for training, together with notes on TRC export.
- `metrics/` – Implementations of the three proposed metrics:
  - **Jo‑Ha‑Kyu score** (tempo‑change synchronization, using SMPL BVH / NPZ),
  - **Motion aesthetic (S‑curve) score** (using SMPL TRC / positions),
  - **Head–hand contrast score** (using SMPL TRC / positions),
  plus small example scripts.
- `tempo_features/` – Examples of constructing **tempo‑changing music features**:
  - a **global** version (instantaneous BPM and its change interpolated over the audio),
  - a **frame‑aligned** version (tempo features aligned to motion frames).

### Quick start & Examples

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the provided examples** to see how everything works:

   - **Dataset Pipeline Verification**:
     The dataset pipeline is split into three steps to allow for the manual MotionBuilder process.
     
     **Step 1: Prepare BVH for MotionBuilder**
     ```bash
     python dataset/examples/step1_prepare_for_mb.py
     ```
     (Creates `exampleData/bvh_to_smpl_example/bvhForC` with T-pose inserted)
     
     **Step 2: MotionBuilder Retargeting**
     Open Autodesk MotionBuilder and run the script:
     `dataset/motionbuilder/PuppetToSmpl.py`
     (Reads from `bvhForC`, outputs to `bvhForC/output`)
     
     **Step 3: Finalize to NPZ**
     ```bash
     python dataset/examples/step3_finalize_npz.py
     ```
     (Converts MB output to standardized SMPL NPZ in `exampleData/bvh_to_smpl_example/npz`)

   - **Metrics Calculation**:
     ```bash
     python metrics/examples/run_metrics_example.py
     ```
     (Computes Jo-Ha-Kyu, S-curve, and Contrast scores on sample data in `exampleData`)

   - **Music Feature Extraction**:
     ```bash
     python tempo_features/examples/extract_features_example.py
     ```
     (Extracts beat-aligned tempo features using Madmom from `exampleData/wav`)

---

## Dataset pipeline: from puppet BVH to SMPL BVH / NPZ / TRC

This repository shows how the JoruriPuppet motion data are processed, following the methodology described in the paper and appendix:

![JoruriPuppet Dataset](asset/JoruriPuppet.png)

- Raw puppet BVH clips (IMU skeleton)  
  → MotionBuilder retargeting to SMPL BVH  
  → SMPL BVH standardized and converted to SMPL `.npz` (axis‑angle + root translation)  
  → optional export to SMPL‑based TRC for metrics (S‑curve, head–hand contrast).

The official downloadable dataset already contains (see appendix for details):

- `bvh/` – raw captured motion (puppet skeleton),
- `bvhSMPL/` – motions retargeted to the SMPL skeleton,
- `npz/` – SMPL `.npz` for training (`trans` + `poses`),
- `trcSMPL/` – joint positions on the SMPL skeleton,
- `wav/` – corresponding audio.

### Recommended Train/Test Splits

We provide recommended train/test split files in `dataset/`:

- `dataset/train_all.txt` & `dataset/test_all.txt`
- `dataset/train_shamisen.txt` & `dataset/test_shamisen.txt`

These splits are designed based on the **Jo-Ha-Kyu score** to ensure that both the training and test sets have similar score distributions. Users can refer to these for their experiments or define their own splits as needed.

The scripts in `dataset/` are provided so that others can **rebuild similar pipelines for their own motion capture data**.

### Step 1 – Prepare puppet BVH clips for MotionBuilder

Goal: insert a **T‑pose frame at the beginning** of each puppet BVH clip and normalize filenames.

- Script: `dataset/python/add_tpose_and_rename_clips.py`
- Example Usage (via `step1_prepare_for_mb.py`):
  
  ```bash
  python dataset/examples/step1_prepare_for_mb.py
  ```

This will:
- Read raw BVH clips from `exampleData/bvh` (or your source).
- Insert a single T-pose frame at the start.
- Save them to `bvhForC/` ready for MotionBuilder.

This corresponds to the appendix step where IMU skeleton BVH files are aligned to an SMPL T‑pose.

### Step 2 – Retarget puppet BVH to SMPL in MotionBuilder

Goal: retarget each `bvhForC` puppet BVH clip to an SMPL skeleton and export SMPL BVH clips.

![MotionBuilder Process](asset/motionbuilder.png)

- Script (to be run **inside MotionBuilder**): `dataset/motionbuilder/PuppetToSmpl.py`
- Before running, ensure `dataset/motionbuilder/smpl-male-T-pose.fbx` exists (provided in repo).

For each clip, the script will:
- open the SMPL T‑pose FBX,
- import the BVH (with T‑pose first frame),
- create and characterize a new Character,
- retarget and bake the motion onto the SMPL skeleton,
- export the result as `output/<clip_name>Re.bvh`.

### Step 3 – Convert SMPL BVH to SMPL NPZ

Goal: convert the retargeted SMPL BVH into standardized SMPL BVH and `.npz` suitable for training.

- Script: `dataset/python/smpl_bvh_to_smpl_npz.py`
- Example Usage (via `step3_finalize_npz.py`):

  ```bash
  python dataset/examples/step3_finalize_npz.py
  ```

The script performs:

1. **SMPL BVH standardization**
   - reads BVH files from `bvhForC/output/`,
   - drops the first (T‑pose) frame,
   - reduces from 6 channels per joint (position + rotation) to 3 rotation channels per joint,
   - writes canonical SMPL BVH files to `bvhSMPL/` using the header from `smpl-T.bvh`.

2. **SMPL NPZ export**
   - reads SMPL BVH from `bvhSMPL/`,
   - splits into root translation (`trans`) and joint rotations,
   - reshapes and reorders joints to match the SMPL joint list,
   - converts Euler rotations to axis‑angle using quaternions,
   - saves `npz/*.npz` with:
     - `trans` – shape `(T, 3)`, root translation,
     - `poses` – shape `(T, 24, 3)`, SMPL axis‑angle rotations.

These NPZ files match the format described in the appendix and used for training in the paper.

### Step 4 – (Optional) Export TRC for position‑based metrics

For metrics that operate on positions (S‑curve and head–hand contrast), motions are also exported as TRC files with the SMPL skeleton:
- export TRC from MotionBuilder with joint names consistent with the SMPL skeleton,
- organize them under `trcSMPL/`, as in the official dataset.

---

## Metrics: Jo–Ha–Kyu, S‑curve, and head–hand contrast

We provide implementations of the three metrics proposed in the paper (Section 3.3 “Metrics for Evaluating Tempo‑changing Motion”) and detailed in the appendix:

![Metrics Overview](asset/metrics.png)

- **Jo–Ha–Kyu score** – measures the correlation between **music tempo changes** and **motion speed** over beat‑based segments, using Pearson correlation and Fisher’s z‑transform for averaging across sequences.
- **Motion aesthetic (S‑curve) score** – evaluates the presence of desirable S‑shaped trajectories in **head and hand motion**, computed on a PCA “motion characteristic plane”.
- **Head–hand contrast score** – measures the theatrical opposite movements between head and hand, based on relative phase and spatial difference, mapped through a Gaussian contrast function.

### Metric Implementation Examples

The `metrics/` package is organized as:

- `jo_ha_kyu.py`
  - functions such as `compute_jo_ha_kyu_from_smpl_bvh(bvh_path, audio_path, ...)`
    and `compute_jo_ha_kyu_from_smpl_npz(npz_path, audio_path, ...)`,
  - internally:
    - uses Madmom (`DBNBeatTrackingProcessor` + `RNNBeatProcessor`) for beat times and instantaneous BPM,
    - segments motion by music beats,
    - computes motion speed per beat segment,
    - returns the Pearson correlation between tempo and motion speed as the Jo–Ha–Kyu score, with optional Fisher z‑averaging across multiple clips (as in the paper).

- `s_curve.py`
  - functions to compute S‑curve scores from SMPL‑based TRC or reconstructed SMPL joint positions (via FK from NPZ),
  - implements the curvature‑percentage‑based S‑curve metric:
    - project the head / hand trajectories onto a PCA plane,
    - compute curvature percentage using sagitta‑based geometry,
    - count segments whose curvature falls into the empirically determined “desirable” range (e.g., 20–60%) and average over segments, as in Equation (3) and the appendix.

- `head_hand_contrast.py`
  - functions to compute head–hand contrast from SMPL‑based TRC or reconstructed positions,
  - follows the paper’s definition:
    - analyze relative phase between head and hand trajectories (PCA + angle difference),
    - compute contrast values and map them with a Gaussian weighting around the reference contrast mean and standard deviation learned from ground truth (as in Equation (4) and the appendix).

- `tempo_utils.py`
  - shared utilities for beat and tempo extraction using Madmom (DBN beat tracker + RNN beat processor),
  - used consistently in Jo–Ha–Kyu and the tempo‑segmented S‑curve / contrast metrics.

### Using the metrics (high‑level)

At a high level, the metrics can be used as follows:

- Given a motion clip in SMPL BVH or NPZ format and its corresponding audio:
  - call the Jo–Ha–Kyu function to obtain the tempo‑change synchronization score,
  - call the S‑curve function (on TRC or reconstructed positions) to evaluate aesthetic quality of head/hand trajectories,
  - call the head–hand contrast function to evaluate theatrical contrast between head and hands.

For detailed mathematical definitions and interpretation, please refer to:

- Main paper, Section 3.3 “Metrics for Evaluating Tempo‑changing Motion”,
- Appendix sections “Qualitative Evaluations for Proposed Metrics” and “Metric Details in Analysis”.

Example script: `metrics/examples/run_metrics_example.py` demonstrates end‑to‑end usage on one JoruriPuppet sequence.

### Multi-sequence Jo–Ha–Kyu averaging (template only)

The paper reports an averaged Jo–Ha–Kyu score across multiple sequences using **Fisher’s z-transform** and also reports a **95% confidence interval**.
This repository keeps the runnable examples focused on a single sequence, but `metrics/jo_ha_kyu.py` also includes the corresponding utility functions:

- `fishers_z_transform(r_values, confidence_level=0.95)`
- `compute_jo_ha_kyu_avg(results, confidence_level=0.95)`

If you have multiple sequences locally, the typical workflow is:

- compute one `JoHaKyuResult` per sequence using `compute_jo_ha_kyu_from_bvh_and_audio(...)`,
- collect the `r` values (or pass the list of results),
- apply Fisher z averaging to obtain `r_avg` and the 95% CI.

See `metrics/examples/jo_ha_kyu_avg_example.py` for a minimal template script (requires you to provide your own multi-sequence file list).

---

## Tempo‑changing music features (global and frame‑aligned)

We provide example implementations of the **tempo‑changing music features** used in the paper and appendix:

- Baseline music features (e.g., envelope / spectral flux, MFCC, chroma, onset peaks, beat positions),
- plus **two additional dimensions**:
  - instantaneous BPM for each beat interval,
  - ΔBPM (the rate of BPM change between neighboring beat intervals).

Beat positions and tempo are estimated using **Madmom** (DBN beat tracker + RNN beat processor), following the experimental setup described in the paper and in Appendix Section 10 “Implementation Details for Performance Gain”.

### Two usage modes

- **Global tempo‑changing features** (audio‑time interpolation)
  - Implemented in `tempo_features/global_tempo_features.py` (based on `extract_musicfea37_tempoChanges.py`),
  - Extract baseline audio features on a fixed hop length (e.g., Librosa),
  - Compute beat times with Madmom, then instantaneous BPM and ΔBPM per beat segment:
    - `BPM_k = 60 / Δt_k`
    - `ΔBPM_k = BPM_k - BPM_{k-1}`
  - Linearly interpolate tempo and Δtempo onto the audio feature frame grid,
  - Used in the paper when integrating tempo‑changing features into existing models on datasets such as AIST++ and FineDance.

- **Frame‑aligned tempo‑changing features** (motion‑time alignment)
  - Implemented in `tempo_features/beat_aligned_tempo_features.py` (based on `bvhFeatureExportAllmotorica-tempoNewAlignT.py`),
  - Given motion frame rate and number of frames:
    - map Madmom beat times to motion frame indices,
    - build beat activation sequences aligned to motion frames,
    - interpolate instantaneous BPM and ΔBPM onto the motion frame timeline,
    - Used in the paper for datasets with strong and frequent tempo changes (e.g., Motorica), where fine frame‑wise alignment yields better performance.

### Example usage (conceptual)

- **Global features** – for each `.wav` file:
  - compute `(35 + 2)`‑D features (baseline 35‑D + 2‑D tempo features),
  - save them as `.npy` (e.g., compatible with LODGE’s 35‑D baseline extended to 37‑D).

- **Frame‑aligned features** – for a specific motion/audio pair:
  - given motion frame rate and number of frames,
  - compute beat‑aligned tempo features with one value per motion frame,
  - use them as additional channels in multi‑modal music‑to‑motion models.

These examples correspond to the configurations described in:

- Main paper, Section 3 (Tempo‑aware Methodology),
- Appendix Section 10 “Implementation Details for Performance Gain”.

### License and citation

The code is released under the license specified in `LICENSE`.  
If you use this code or the JoruriPuppet dataset in academic work, please cite the paper:

```bibtex
@inproceedings{dong2025joruripuppet,
  title     = {JoruriPuppet: Learning Tempo-Changing Mechanisms Beyond the Beat for Music-to-Motion Generation},
  author    = {Dong, Ran and Ni, Shaowen and Yang, Xi},
  booktitle = {Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers},
  pages     = {1--11},
  year      = {2025},
  doi       = {10.1145/3757377.3764006}
}
```
