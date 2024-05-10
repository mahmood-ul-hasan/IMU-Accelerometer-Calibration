# IMU-Accelerometer-Calibration
 ## A Field Calibration Method for Low-Cost MEMS Accelerometer Based on the Generalized Nonlinear Least Square Method

## Description
This repository contains the source code implementation of the paper titled "A Field Calibration Method for Low-Cost MEMS Accelerometer Based on the Generalized Nonlinear Least Square Method" by Hassan, M.u., and Bao, Q., published in Multiscale Sci. Eng. 2, 135–142 (2020). [DOI: 10.1007/s42493-020-00045-2] It includes the necessary data set and main files to run the calibration process.

## Data Set
Two accelerometers were utilized in the verification process of the proposed algorithm:
- Innalabs AHRS M3
- Smartphone accelerometer STMicroelectronics LIS3DH

The provided data set includes the following files:
1. CalibrationOfLowCostTriaxialInertialSensors.mat
2. MobileAccData30orientationRawSample01.mat

## Running the Code
To execute the calibration process, use the main file:
- `CombinedAlgorithm.m`

Supporting functions included are:
- `Funct_Gauss.m`: Function to solve the nonlinear equation.
- `LMFnlsq.m`: Function to solve the nonlinear equation.
- A solution is obtained by a Fletcher's version of the Levenberg-Maquardt algoritm for minimization of a sum of squares of equation residuals. The main domain of LMFnlsq applications is in curve fitting during processing of experimental data.

## Usage
1. Clone this repository to your local machine.
2. Ensure MATLAB is installed.
3. Open MATLAB and navigate to the repository directory.
4. Run `CombinedAlgorithm.m` to perform the calibration process.

## Citation
If you use this code or data set in your research, please cite the following paper:
Hassan, M.u., Bao, Q. A Field Calibration Method for Low-Cost MEMS Accelerometer Based on the Generalized Nonlinear Least Square Method. Multiscale Sci. Eng. 2, 135–142 (2020). [DOI: 10.1007/s42493-020-00045-2]


## Overview of paper
This paper proposes a field calibration method for an accelerometer without the need of having any external devices for calibration. In the proposed calibration method, generalized nonlinear least-square (GNLS) is used to estimate deterministic errors. A novel sensor’s data collection procedure is developed to collect data of an accelerometer along all three axes and all possible orientations where the expectation of influence of all possible errors is very high. The proposed calibration method is verified by applying it to two different accelerometers. The proposed calibration method achieved an accurate estimation of calibration parameters. The results of the proposed GNLS based calibration method are compared with two other commonly used algorithms, such as Levenberg–Marquardt (LM) and Gauss–Newton (GN). Simulation and experimental results show that the proposed GNLS-based calibration method is slightly more accurate than the LM and GN. The GNLS convergence rate for estimating the calibration parameters is also faster than the LM and GN.

