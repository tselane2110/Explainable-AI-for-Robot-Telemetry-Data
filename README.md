# Explainable-AI-for-Robot-Telemetry-Data (Assignment Report)

---

## **TABLE OF CONTENTS**
1. [Executive Summary](#executive-summary)
2. [Introduction](#introduction)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Implementation](#model-implementation)
5. [Model Evaluation](#model-evaluation)
6. [Explainable AI Analysis](#explainable-ai-analysis)
7. [Conclusion](#conclusion)
8. [References](#references)
9. [Appendix](#appendix)

---

## **EXECUTIVE SUMMARY**

This project implements a comprehensive machine learning pipeline for anomaly detection in robot/drone telemetry data. Following assignment guidelines, we implemented and evaluated **six machine learning models** (LSTM, 1D-CNN, SVM, XGBoost, VAE, and FNN) and performed extensive **Explainable AI (XAI)** analysis using SHAP, LIME, and other interpretability techniques. The system successfully classifies telemetry data into three categories: **Normal Operation**, **DoS Attack**, and **Malfunction**, achieving [X]% average accuracy across models while providing interpretable insights into model decisions.

**Key Achievements:**
- Implemented 3 even-numbered models with hyperparameter tuning
- Handled challenging data issues (95% sparsity in some sensors)
- Applied multiple XAI techniques (SHAP, LIME, PDP, Feature Importance)
- Provided actionable insights for robotics security and maintenance

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
* **XGBoost**: Doesnt show the best results but will improve later
<img width="432" height="197" alt="image" src="https://github.com/user-attachments/assets/9e34b356-bd76-4353-b929-58406f684198" />
* **1D-CNN**:
<img width="439" height="197" alt="image" src="https://github.com/user-attachments/assets/c69f078d-e574-4b9b-a223-d93aa8c0a882" />
* **FNN**:
<img width="435" height="189" alt="image" src="https://github.com/user-attachments/assets/3c3f52fc-2427-41c0-a7a1-cccaf40a18d5" />

---

## **EXPLAINABLE AI ANALYSIS**
## **APPENDIX**

### **A.1 Code Repository Structure**
```
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
