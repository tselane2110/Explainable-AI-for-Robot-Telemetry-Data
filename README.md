# Explainable-AI-for-Robot-Telemetry-Data (Assignment Report)

---

## **TABLE OF CONTENTS**
1. [Executive Summary](#executive-summary)
2. [Introduction](#introduction)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Evaluation](#model-evaluation)
5. [Explainable AI Analysis](#explainable-ai-analysis)
6. [Appendix](#appendix)

---

## **EXECUTIVE SUMMARY**

This project implements a comprehensive machine learning pipeline for anomaly detection in robot/drone telemetry data. In accordance with the assignment requirements for **even-numbered roll numbers**, this report focuses on the implementation and evaluation of three specific models: **1D-CNN, XGBoost, and Feedforward Neural Network (FNN)**.

The system is designed to classify telemetry data into three distinct categories: **Normal Operation**, **DoS Attack**, and **Malfunction**. Despite facing significant data quality challenges—specifically **95% sparsity** across most sensor readings—the project successfully utilized aggressive feature selection to train robust classifiers.

**Key Achievements & Findings:**

* **High Performance:** The **FNN model** emerged as the top performer, achieving approximately **97.6% accuracy** on the test set.
* **XAI Insights:** Explainable AI analysis (focused on XGBoost) revealed that the models rely heavily on **GPS coordinates (Longitude/Latitude)** to detect anomalies. This indicates the system effectively identified *where* attacks occurred, but may require additional system health metrics (CPU, RAM) to generalize to new locations.
* **Data Handling:** Successfully processed a heterogeneous dataset with extreme missing values by narrowing the focus to high-reliability features like GPS and timestamps.

This report details the preprocessing steps, model architectures, and a critical analysis of why the models prioritized spatial data over system performance metrics.

---

## **INTRODUCTION**

### **1.1 Project Background**
Autonomous robots and drones are increasingly deployed in critical applications, making their security and reliability paramount. Telemetry data provides a rich source of information for detecting anomalies, but manual monitoring is impractical. This project addresses the need for automated, explainable anomaly detection systems.

### **1.2 Problem Statement**
Detect and classify anomalies in robot telemetry data into:
- **Normal Operation**: Standard system behavior
- **DoS Attack**: Malicious denial-of-service attempts
- **Malfunction**: System failures or degraded performance

### **1.3 Objectives**
1. Preprocess sparse, heterogeneous telemetry data
2. Implement 3 ML/DL models (XGBoost, 1D-CNN and FNN) as per assignment requirements
3. Apply XAI techniques to explain model predictions
4. Compare model performance and interpretability
5. Provide actionable insights for robotics security

### **1.4 Dataset Description**
- **Source**: Explainable-AI-for-Robot-Telemetry-Data GitHub repository
- **Classes**: Normal (4 files), DoS Attack (2 files), Malfunction (2 files)
- **Initial Size**: 87,417 samples × 79 features
- **Key Features**: GPS coordinates, battery status, IMU data, CPU/RAM usage, RSSI signals

---

## **DATA PREPROCESSING**

### **2.1 Data Loading and Initial Exploration**
```python
# Key Statistics
Initial dataset shape: (87417, 79)
Classes: ['Normal', 'DoS_Attack', 'Malfunction']
Class distribution: Normal (XX%), DoS (XX%), Malfunction (XX%)
```

### **2.2 Missing Data Analysis**
```
                                               null_count  null_percent
setpoint_raw-global_Time                                0      0.000000
setpoint_raw-global_header.seq                          0      0.000000
setpoint_raw-global_header.stamp.secs                   0      0.000000
setpoint_raw-global_latitude                            0      0.000000
setpoint_raw-global_longitude                           0      0.000000
setpoint_raw-global_altitude                            0      0.000000
battery_Time                                        83068     95.024995
battery_header.seq                                  83068     95.024995
battery_header.stamp.secs                           83068     95.024995
battery_voltage                                     83068     95.024995
battery_current                                     83068     95.024995
battery_temperature                                 83068     95.024995
battery_percentage                                  83068     95.024995
global_position-local_Time                          82889     94.820229
global_position-local_header.seq                    82889     94.820229
global_position-local_header.stamp.secs             82889     94.820229
global_position-local_pose.pose.position.x          82889     94.820229
global_position-local_pose.pose.position.y          82889     94.820229
global_position-local_pose.pose.position.z          82889     94.820229
global_position-local_pose.pose.orientation.x       82889     94.820229
global_position-local_pose.pose.orientation.y       82889     94.820229
global_position-local_pose.pose.orientation.z       82889     94.820229
global_position-local_twist.twist.linear.x          82889     94.820229
global_position-local_twist.twist.linear.y          82889     94.820229
global_position-local_twist.twist.linear.z          82889     94.820229
imu-data_Time                                       82912     94.846540
imu-data_header.seq                                 82912     94.846540
imu-data_header.stamp.secs                          82912     94.846540
imu-data_orientation.x                              82912     94.846540
imu-data_orientation.y                              82912     94.846540
imu-data_orientation.z                              82912     94.846540
imu-data_orientation.w                              82912     94.846540
imu-data_angular_velocity.x                         82912     94.846540
imu-data_angular_velocity.y                         82912     94.846540
imu-data_angular_velocity.z                         82912     94.846540
rc-out_Time                                         83036     94.988389
rc-out_header.seq                                   83036     94.988389
rc-out_header.stamp.secs                            83036     94.988389
rc-out_channels_0                                   83036     94.988389
rc-out_channels_1                                   83036     94.988389
rc-out_channels_2                                   83036     94.988389
rc-out_channels_3                                   83036     94.988389
rc-out_channels_4                                   83036     94.988389
vfr_hud_Time                                        83094     95.054738
vfr_hud_header.seq                                  83094     95.054738
vfr_hud_header.stamp.secs                           83094     95.054738
vfr_hud_airspeed                                    83094     95.054738
vfr_hud_groundspeed                                 83094     95.054738
vfr_hud_heading                                     83094     95.054738
vfr_hud_throttle                                    83094     95.054738
vfr_hud_altitude                                    83094     95.054738
vfr_hud_climb                                       83094     95.054738
global_position-global_header.stamp.secs            82887     94.817942
global_position-global_altitude                     82887     94.817942
global_position-global_longitude                    82887     94.817942
global_position-global_latitude                     82887     94.817942
global_position-raw-satellites_data                 82885     94.815654
setpoint_raw-target_global_Time                     84622     96.802681
setpoint_raw-target_global_header.seq               84622     96.802681
setpoint_raw-target_global_header.stamp.secs        84622     96.802681
setpoint_raw-target_global_latitude                 84622     96.802681
setpoint_raw-target_global_longitude                84622     96.802681
setpoint_raw-target_global_altitude                 84622     96.802681
setpoint_raw-target_global_yaw                      84622     96.802681
state_Time                                          86951     99.466923
state_header.seq                                    86951     99.466923
state_connected                                     86951     99.466923
state_armed                                         86951     99.466923
state_guided                                        86951     99.466923
state_manual_input                                  86951     99.466923
state_system_status                                 86951     99.466923
RSSI_Time                                           87332     99.902765
RSSI_Quality                                        87332     99.902765
RSSI_Signal                                         87332     99.902765
CPU_Time                                            87235     99.791803
CPU_Percent                                         87235     99.791803
RAM_Time                                            86864     99.367400
Used_RAM_MB                                         86864     99.367400
class                                                   0      0.000000
```
- **Challenge**: Most sensor columns had 90-95% missing values
- **Solution**: Aggressive feature selection keeping only columns with <90% nulls (better solution would be data interpolation but lack of time hurdled in attempting that)
- **Result**: Reduced to 7 usable features

### **2.3 Feature Engineering**
Should've performed to create the following features or more, but didn't:
1. **Battery Drain Rate**: Δvoltage/Δtime
2. **Position Stability**: Variance in GPS coordinates
3. **Signal Quality Index**: Combined RSSI metrics
4. **System Load Ratio**: CPU/RAM normalized usage

### **2.4 Data Normalization/Standardization**
Should've performed but didn't due to lack of time

### **2.5 Feature Correlation Analysis**
Should've perfomed but didn't due to time constraints

### **2.6 Data Splitting**
- **Split Ratio**: 80% Train, 10% Validation, 10% Test
- **Method**: Stratified sampling to preserve class distribution
- **Final Sizes**: Train (XX), Validation (XX), Test (XX)

---
## **MODEL EVALUATION**

### **3.1 Performance Metrics (Class-wise) Comparison**
* **XGBoost**: Doesnt show the best results but will improve later </br>
<img width="432" height="197" alt="image" src="https://github.com/user-attachments/assets/9e34b356-bd76-4353-b929-58406f684198" /> </br>
* **1D-CNN**: </br>
<img width="439" height="197" alt="image" src="https://github.com/user-attachments/assets/c69f078d-e574-4b9b-a223-d93aa8c0a882" /> </br>
* **FNN**:</br>
<img width="435" height="189" alt="image" src="https://github.com/user-attachments/assets/3c3f52fc-2427-41c0-a7a1-cccaf40a18d5" /> </br>

---

## EXPLAINABLE AI ANALYSIS
Based on the analysis in the provided notebook, here is the report for the Explainable AI (XAI) section, followed by a conclusion of the overall project results.

* **Models Analyzed:** XGBoost
* **Classification Task:** 3 Classes (Normal, DoS Attack, Malfunction)
* **Features Available:** 6 (after preprocessing)

### 4.1 Most Important Features
The analysis identified the top features driving the model's predictions, ranked by importance:

1. **`setpoint_raw-global_longitude`** (Importance: ~0.497)
2. **`setpoint_raw-global_latitude`** (Importance: ~0.292)
3. **`setpoint_raw-global_Time`** (Importance: ~0.136)
4. **`setpoint_raw-global_altitude`** (Importance: ~0.058)
5. **`setpoint_raw-global_header.seq`** (Importance: ~0.016)

### 4.2 Key Findings
* **Dominance of GPS:** GPS coordinates (longitude and latitude) are the strongest predictors, accounting for the vast majority of the model's decision-making power.
* **Temporal Relevance:** Temporal features (Time) show moderate importance, suggesting the timing of events provides necessary context.
* **Positional Reliance:** The model heavily discriminates between classes based on position data.

### 4.3 Interpretation of Results
**The dominance of GPS coordinates suggests that:**
* **Location Patterns:** Specific location patterns are key differentiators between normal flight, attacks, and malfunctions.
* **Spatial Correlations:** Denial of Service (DoS) attacks and malfunctions may be occurring in specific geographic locations or along specific flight paths.
* **Context:** Time stamps likely help sequence these events, distinguishing isolated incidents from sustained attacks.

### 4.4 Model Decision-Making Process

1. **Check Position:** It first evaluates position data (Latitude/Longitude).
2. **Check Time:** It then considers the temporal context (Timestamp/Sequence).
3. **Check Altitude:** Finally, it evaluates altitude and other secondary parameters.

### 4.5 Practical Implications
* **For Anomaly Detection:**
* Monitor flight paths for unusual concentrations of activity in specific locations.
* Use position data as the primary indicator for flagging potential threats.


* **For System Design:**
* GPS/position sensors are critical components for this security application.
* Ensuring the accuracy of timestamps is vital for the secondary validation layer.


### 4.6 Limitations & Recommendations
* **Limitations:** The model relies heavily on position data, which may limit its ability to generalize to new locations. It currently lacks visibility into system metrics like CPU or RAM usage.
* **Recommendations:**
* **Collect More Data:** Incorporate sensor data such as CPU load, RAM usage, battery levels, and network traffic to create a more robust profile.
* **Location Independence:** Implement anomaly detection that isn't solely tied to specific coordinates to improve generalization.
* **Ensemble Methods:** Combine position-based models with system-performance models for multi-source verification.

---

## 5. Conclusion of Overall Results
The project successfully trained and evaluated machine learning models to detect anomalies in robot telemetry data.

5.1 **Model Performance:**
* The **Feedforward Neural Network (FNN)** achieved high performance, reaching approximately **97.6% accuracy** on the test set and **98% accuracy** on the validation set.
* The **XGBoost** model served as the core for the Explainable AI analysis, demonstrating high confidence (e.g., 99.99%) in its classifications.


5.2 **Critical Insights:**
* The primary differentiator for detecting DoS attacks and malfunctions in this specific dataset is **spatial data** (Longitude/Latitude).
* **DoS Attacks** and **Malfunctions** exhibited distinct feature averages compared to **Normal** flights. For instance, DoS attacks were associated with specific average longitude/latitude values significantly different from normal patterns.


5.3 **Final Verdict:**
While the models are highly accurate, their heavy reliance on GPS coordinates indicates they are learning *where* attacks happen rather than *how* they manifest in system performance (like CPU spikes). To make the system robust against attacks in *new* locations, future iterations must prioritize system health metrics (CPU, RAM, Battery) over raw geospatial coordinates.

## **APPENDIX**

### **A.1 Code Repository Structure**
```
super simple 
```

### **A.2 Environment Setup**
```
no setup required.
Just run the commands in the notebook in order 
```

### **A.4 Hardware Specifications**
Colab's GPU :)

### **A.5 Contact Information**
- **Student**: [Tanzeela Sehar]
- **Email**: [muc.555@gmail.com]
- **GitHub**: [tselane2110]

---

**END OF REPORT**
